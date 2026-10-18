import torch
from torch.amp import GradScaler, autocast
from monai.losses import DiceCELoss
import copy
from tqdm import tqdm
import config


class LocalNodeTrainer:
    def __init__(self, model):
        self.model = model
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=config.BASE_LR, 
            weight_decay=config.WEIGHT_DECAY
        )
        # Calculate loss without penalizing background class
        self.loss_fn = DiceCELoss(to_onehot_y=True, softmax=True, include_background=True)
        self.scaler = GradScaler("cuda", enabled=config.USE_AMP and torch.cuda.is_available())
        self.global_weights_cache = None

    def train(self, dataloader, epochs=config.EPOCHS_PER_ROUND):
        self.model.train()
        self.global_weights_cache = {k: v.clone() for k, v in self.model.state_dict().items()}
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            self.optimizer.zero_grad()
            
            # Use TQDM for clean console output
            progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs}")
            
            for step, batch in progress_bar:
                inputs, labels = batch["image"].to(config.DEVICE), batch["label"].to(config.DEVICE)
                
                with autocast("cuda", enabled=config.USE_AMP and torch.cuda.is_available()):
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, labels) / config.GRAD_ACCUM_STEPS
                
                self.scaler.scale(loss).backward()
                
                # Gradient Accumulation Execution
                if (step + 1) % config.GRAD_ACCUM_STEPS == 0 or (step + 1) == len(dataloader):
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                
                epoch_loss += (loss.item() * config.GRAD_ACCUM_STEPS)
                progress_bar.set_postfix({"loss": f"{loss.item() * config.GRAD_ACCUM_STEPS:.4f}"})
                
        return epoch_loss / len(dataloader)

    # --- BRIDGE FUNCTIONS FOR YOUR TEAMMATES ---

    def extract_weight_delta(self) -> dict:
        """For the Federated Learning Lead (FedAvg computation)"""
        if not self.global_weights_cache:
            raise ValueError("Must run train() before extracting delta.")
        delta = {}
        local_state = self.model.state_dict()
        for key in local_state.keys():
            delta[key] = (local_state[key] - self.global_weights_cache[key].to(local_state[key].device)).cpu()
        return delta

    def compute_zk_l2_norm(self, delta_dict: dict) -> float:
        """For the Zero-Knowledge Engineer (zk-SNARK poisoning proof)"""
        sum_sq = sum(torch.sum(param ** 2).item() for param in delta_dict.values())
        return float(sum_sq ** 0.5)