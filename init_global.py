import torch
from model import build_model

torch.manual_seed(42)
model = build_model()
torch.save(model.state_dict(), "global_round0.pth")
print("Saved global_round0.pth")