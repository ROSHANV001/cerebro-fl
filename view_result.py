import nibabel as nib
try:
    import matplotlib.pyplot as plt  # type: ignore[import-not-found]
except ImportError:
    plt = None
import numpy as np

if plt is None:
    raise SystemExit("matplotlib is required to display the MRI results.")

# 1. Load the original brain scan and your AI's predicted mask
# (Change the ID if you want to look at a different patient)
patient_id = "BraTS2021_00204"
flair_path = f"./data/holdout_test/{patient_id}/{patient_id}_flair.nii.gz"
mask_path = f"./frontend_masks/{patient_id}_prediction.nii.gz"

flair_volume = nib.load(flair_path).get_fdata()
mask_volume = nib.load(mask_path).get_fdata()

# 2. Pick a cross-section slice in the middle of the brain (Z-axis)
# BraTS images are typically 155 slices deep.
slice_idx = 75

plt.figure(figsize=(12, 6))

# Plot Original MRI
plt.subplot(1, 2, 1)
plt.title(f"Original MRI ({patient_id})")
plt.imshow(flair_volume[:, :, slice_idx].T, cmap="gray", origin="lower")
plt.axis('off')

# Plot AI Overlay
plt.subplot(1, 2, 2)
plt.title("AI Tumor Prediction Overlay")
plt.imshow(flair_volume[:, :, slice_idx].T, cmap="gray", origin="lower")

# Filter out the empty background (0) so only the tumor is colored
tumor_only = np.ma.masked_where(mask_volume[:, :, slice_idx] == 0, mask_volume[:, :, slice_idx])
plt.imshow(tumor_only.T, cmap="jet", alpha=0.6, origin="lower")
plt.axis('off')

plt.tight_layout()
plt.show()