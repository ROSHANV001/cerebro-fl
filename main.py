import argparse
import torch
import json
from model import build_model
from trainer import LocalNodeTrainer
from inference import run_inference
from dataset import get_brats_dataloader

def main():
    parser = argparse.ArgumentParser(description="Federated 3D-UNet Edge Node Pipeline")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "infer"], help="Run mode")
    parser.add_argument("--data", type=str, required=True, help="Path to BraTS patient folders")
    parser.add_argument("--out", type=str, default="./outputs", help="Output directory for masks/deltas")
    parser.add_argument("--weights", type=str, default=None, help="Path to .pth checkpoint (optional)")
    args = parser.parse_args()

    # 1. Initialize
    model = build_model()
    if args.weights:
        model.load_state_dict(torch.load(args.weights, map_location="cpu"))
        print(f"Loaded weights from {args.weights}")

    # 2. Execute Mode
    if args.mode == "train":
        print("\n--- Starting Local Node Training ---")
        loader = get_brats_dataloader(args.data, is_train=True)
        trainer = LocalNodeTrainer(model)
        
        loss = trainer.train(loader)
        print(f"\n✅ Training Complete. Final Loss: {loss:.4f}")
        
        # 3. Generate Integration Artifacts for Teammates
        print("\n--- Generating Teammate Artifacts ---")
        delta = trainer.extract_weight_delta()
        l2_norm = trainer.compute_zk_l2_norm(delta)
        
        import os
        os.makedirs(args.out, exist_ok=True)
        
        torch.save(model.state_dict(), f"{args.out}/local_model.pth")
        torch.save(delta, f"{args.out}/weight_delta.pth")
        with open(f"{args.out}/zk_metadata.json", "w") as f:
            json.dump({"l2_norm_scaled": int(l2_norm * 10000)}, f)
            
        print(f"📦 Artifacts exported to {args.out}/")
        print(f"   ↳ ZK L2 Norm: {l2_norm:.6f}")

    elif args.mode == "infer":
        print("\n--- Starting 3D Volume Segmentation ---")
        run_inference(model, args.data, args.out)

if __name__ == "__main__":
    main()