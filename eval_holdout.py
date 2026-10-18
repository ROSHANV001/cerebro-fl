import sys, torch
from monai.inferers import SlidingWindowInferer
from model import build_model
from dataset import get_brats_dataloader
import config

def dice(p, g):
    if p.sum() == 0 and g.sum() == 0:
        return 1.0
    return (2 * (p & g).sum() / (p.sum() + g.sum() + 1e-8)).item()

REGIONS = {
    "WT": lambda t: t > 0,
    "TC": lambda t: (t == 1) | (t == 3),
    "ET": lambda t: t == 3,
}

def evaluate(weights):
    model = build_model()
    model.load_state_dict(torch.load(weights, map_location=config.DEVICE))
    model.eval()
    inferer = SlidingWindowInferer(roi_size=config.PATCH_SIZE, sw_batch_size=1, overlap=0.5, mode="gaussian")
    loader = get_brats_dataloader("./data/holdout_test", is_train=False, has_label=True)
    scores = {k: [] for k in REGIONS}
    leak = []
    with torch.no_grad():
        for b in loader:
            x = b["image"].to(config.DEVICE)
            y = b["label"].squeeze().to(config.DEVICE).long()
            with torch.amp.autocast("cuda", enabled=config.USE_AMP and torch.cuda.is_available()):
                pred = inferer(x, model).argmax(1).squeeze()

            brain = x[0, 0] != 0
            leak.append((pred[~brain] > 0).float().mean().item())
            pred[~brain] = 0

            for name, fn in REGIONS.items():
                scores[name].append(dice(fn(pred), fn(y)))
    return {k: sum(v) / len(v) for k, v in scores.items()}, sum(leak) / len(leak)

DEFAULT = [
    "handoff_A/local_model.pth",
    "handoff_B/local_model.pth",
    "handoff_C/local_model.pth",
    "global_round1.pth",
]

for w in (sys.argv[1:] or DEFAULT):
    r, leak = evaluate(w)
    print(w, {k: round(v, 3) for k, v in r.items()},
          "mean:", round(sum(r.values()) / 3, 3),
          "| outside-brain FP:", f"{leak:.1%}", flush=True)