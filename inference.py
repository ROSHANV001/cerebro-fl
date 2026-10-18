import os
import torch
import nibabel as nib
import numpy as np
from monai.inferers import SlidingWindowInferer
from dataset import get_brats_dataloader
import config

def run_inference(model, data_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    loader = get_brats_dataloader(data_dir, is_train=False, has_label=False)
    
    # --- FIX 1: Gaussian Blending & Higher Overlap ---
    inferer = SlidingWindowInferer(
        roi_size=config.PATCH_SIZE, 
        sw_batch_size=1, 
        overlap=0.5, 
        mode="gaussian"
    )
    
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            inputs = batch["image"].to(config.DEVICE)
            
            meta = batch["image"].meta
            raw_path = meta["filename_or_obj"]
            while isinstance(raw_path, (list, tuple)):
                raw_path = raw_path[0]
            
            original_path = str(raw_path)
            patient_id = os.path.basename(os.path.dirname(original_path))
            
            print(f"Segmenting full volume for {patient_id}...")
            with torch.amp.autocast("cuda", enabled=config.USE_AMP and torch.cuda.is_available()):
                logits = inferer(inputs, model)
                probs = torch.softmax(logits, dim=1)
                pred_mask = torch.argmax(probs, dim=1).squeeze().cpu().numpy().astype("uint8")
            
            # --- FIX 2: Background Masking ---
            ref_img = nib.load(original_path)
            ref_data = ref_img.get_fdata()
            
            # Force prediction to 0 (background) wherever the original MRI is black
            pred_mask[ref_data == 0] = 0
            
            pred_mask[pred_mask == 3] = 4  # Map back to BraTS standard
            
            pred_nifti = nib.Nifti1Image(pred_mask, affine=ref_img.affine, header=ref_img.header)
            out_path = os.path.join(output_dir, f"{patient_id}_prediction.nii.gz")
            nib.save(pred_nifti, out_path)
            print(f"Saved: {out_path}")