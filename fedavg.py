# fedavg.py
import torch

base = torch.load("global_round0.pth", map_location="cpu")
deltas = [torch.load(f"handoff_{n}/weight_delta.pth", map_location="cpu") for n in "ABC"]

new_global = {}
for k, v in base.items():
    if v.is_floating_point():
        new_global[k] = v + sum(d[k] for d in deltas) / len(deltas)
    else:
        new_global[k] = v

torch.save(new_global, "global_round1.pth")
print("Saved global_round1.pth")