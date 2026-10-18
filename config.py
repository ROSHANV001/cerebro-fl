import torch

# Hardware Constraints
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = True

# Geometry & VRAM Protection
PATCH_SIZE = (96, 96, 96)
BATCH_SIZE = 1
NUM_SAMPLES = 1          # patches per volume -> real batch = BATCH_SIZE * NUM_SAMPLES
GRAD_ACCUM_STEPS = 4     # effective batch = 4 patches

# Training Hyperparameters
BASE_LR = 2e-4
WEIGHT_DECAY = 1e-5
EPOCHS_PER_ROUND = 60   # smoke test only -> wapas 60 kar dena

# Model Architecture
IN_CHANNELS = 4             # FLAIR, T1, T1ce, T2
OUT_CHANNELS = 4            # Background(0), NCR(1), ED(2), ET(3)
CHANNELS = (16, 32, 64, 128, 256)