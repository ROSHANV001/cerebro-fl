from monai.networks.nets import UNet
import config

def build_model():
    """Builds the 4GB VRAM-optimized 3D-UNet."""
    return UNet(
        spatial_dims=3,
        in_channels=config.IN_CHANNELS,
        out_channels=config.OUT_CHANNELS,
        channels=config.CHANNELS,
        strides=(2, 2, 2, 2),
        num_res_units=1,
        norm="instance"  # CRITICAL: InstanceNorm required because Batch_Size=1
    ).to(config.DEVICE)