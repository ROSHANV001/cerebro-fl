import os
import shutil
import tempfile
import threading
import torch
import nibabel as nib
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.background import BackgroundTask
from monai.inferers import SlidingWindowInferer
from model import build_model
import config
from dataset import get_brats_dataloader

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading model into memory...")
model = build_model()
model.load_state_dict(torch.load(os.getenv("MODEL_PATH", "./model_final.pth"), map_location=config.DEVICE))
model.eval()
inferer = SlidingWindowInferer(
    roi_size=config.PATCH_SIZE, sw_batch_size=1, overlap=0.5, mode="gaussian"
)

gpu_lock = threading.Lock()  # 4GB GPU pe ek time pe ek hi inference


@app.post("/predict")
def predict_tumor(
    flair: UploadFile = File(...),
    t1: UploadFile = File(...),
    t1ce: UploadFile = File(...),
    t2: UploadFile = File(...),
):
    # Per-request isolated temp dir: <tmp>/upload/scan_*.nii.gz
    root = tempfile.mkdtemp(prefix="predict_")
    try:
        patient_dir = os.path.join(root, "upload")
        os.makedirs(patient_dir)

        for f, suffix in zip([flair, t1, t1ce, t2], ["flair", "t1", "t1ce", "t2"]):
            with open(os.path.join(patient_dir, f"scan_{suffix}.nii.gz"), "wb") as buf:
                shutil.copyfileobj(f.file, buf)

        loader = get_brats_dataloader(root, is_train=False, has_label=False)
        batch = next(iter(loader))
        inputs = batch["image"].to(config.DEVICE)

        with gpu_lock, torch.no_grad(), torch.amp.autocast(
            "cuda", enabled=config.USE_AMP and torch.cuda.is_available()
        ):
            logits = inferer(inputs, model)
            pred_mask = (
                torch.argmax(torch.softmax(logits, dim=1), dim=1)
                .squeeze().cpu().numpy().astype("uint8")
            )

        ref_img = nib.load(os.path.join(patient_dir, "scan_flair.nii.gz"))
        pred_mask[ref_img.get_fdata() == 0] = 0   # mask background first
        pred_mask[pred_mask == 3] = 4             # back to BraTS labels

        out_path = os.path.join(root, "tumor_mask.nii.gz")
        nib.save(nib.Nifti1Image(pred_mask, affine=ref_img.affine, header=ref_img.header), out_path)

        # Response bhejne ke baad temp dir auto-delete
        return FileResponse(
            out_path,
            media_type="application/gzip",
            filename="tumor_mask.nii.gz",
            background=BackgroundTask(shutil.rmtree, root, ignore_errors=True),
        )
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)