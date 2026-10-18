import os
import glob
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, NormalizeIntensityd,
    RandCropByPosNegLabeld, RandFlipd, EnsureTyped, MapLabelValued
)
from monai.data import Dataset, DataLoader
import config


def get_brats_dataloader(data_dir: str, is_train: bool = True, has_label: bool = True):
    patient_dirs = [
        os.path.join(data_dir, d) for d in sorted(os.listdir(data_dir))
        if os.path.isdir(os.path.join(data_dir, d))
    ]
    data_dicts = []

    for p_dir in patient_dirs:
        flair = glob.glob(os.path.join(p_dir, "*flair.nii.gz"))
        t1 = glob.glob(os.path.join(p_dir, "*t1.nii.gz"))
        t1ce = glob.glob(os.path.join(p_dir, "*t1ce.nii.gz"))
        t2 = glob.glob(os.path.join(p_dir, "*t2.nii.gz"))
        seg = glob.glob(os.path.join(p_dir, "*seg.nii.gz"))

        required = [flair, t1, t1ce, t2] + ([seg] if has_label else [])
        if all(required):
            item = {"image": [flair[0], t1[0], t1ce[0], t2[0]]}
            if has_label:
                item["label"] = seg[0]
            data_dicts.append(item)

    if not data_dicts:
        raise RuntimeError(f"No valid patient folders found in {data_dir}")

    # Training always needs labels
    if is_train:
        has_label = True

    keys = ["image", "label"] if has_label else ["image"]

    transforms_list = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ]
    if has_label:
        transforms_list.insert(
            2,
            MapLabelValued(keys="label", orig_labels=[0, 1, 2, 4], target_labels=[0, 1, 2, 3]),
        )

    if is_train:
        transforms_list.extend([
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=config.PATCH_SIZE,
                pos=1, neg=1,
                num_samples=config.NUM_SAMPLES,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        ])

    transforms_list.append(EnsureTyped(keys=keys))

    ds = Dataset(data=data_dicts, transform=Compose(transforms_list))
    return DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=is_train, num_workers=0)