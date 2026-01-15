import os
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from itertools import cycle
import logging
from tqdm import tqdm
from tqdm import trange
from model import Transformer
from model import EarlyStopping
from model import DataManager
from model import Hyperparameters
from model import PlateauScheduler

logging.basicConfig(filename='AutoEve\model.log', level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

# 4. Layer unfreezing
class LayerUnfreezer:
    def __init__(self, model, unfreeze_schedule = None, block_att = 'blocks', strategy = 'top-down'):
        self.model = model

        self.unfreeze_schedule = (
            unfreeze_schedule
            if unfreeze_schedule is not None
            else self.auto_schedule(Hyperparameters.max_iters, Hyperparameters.n_layer)
        )
        self.block_att = block_att
        self.strategy = strategy
        self.unfrozen_layers = 0
        self.total_layers = Hyperparameters.n_layer
        assert self.total_layers > 0, "Model must have at least one block to unfreeze"

    def auto_schedule(self, total_steps, num_layers, warmup_percentage=0):
        logging.info("Auto generating unfreeze schedule...")
        if warmup_percentage < 0 or warmup_percentage > 1:
            raise ValueError("Warmup percentage must be between 0 and 1")
        if warmup_percentage == 0:
            logging.warning("No warmup, the model will begin unfreezing from the first layer")
        warmup_steps = int(total_steps * warmup_percentage)
        remaining_steps = total_steps - warmup_steps
        interval = remaining_steps // num_layers
        unfreeze_schedule = [warmup_steps + (i + 1) * interval for i in range(num_layers)]
        return unfreeze_schedule

    def to_unfreeze(self, current_step):
        if self.unfrozen_layers < self.total_layers:
            if current_step in self.unfreeze_schedule:
                block_list = getattr(self.model, self.block_att)

                if self.strategy == 'top-down':
                    block_idx = self.total_layers - 1 - self.unfrozen_layers

                elif self.strategy == 'bottom-up':
                    block_idx = self.unfrozen_layers

                else:
                    raise ValueError(f"Invalid unfreeze strategy: {self.strategy}")

                for name, param in block_list[block_idx].named_parameters():
                    if 'adapter' in name:
                        param.requires_grad = True

                self.unfrozen_layers += 1

                logging.info(f"Unfreezing adapter layer in block: {block_idx}, at step: {current_step}")
                self.log_params()
    
    def log_params(self):
        logging.info(f"{str(sum(p.numel() for p in self.model.parameters() if p.requires_grad)/1e6)} Million trainable parameters")

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split, loader in [('train', DataManager.train_loader), ('val', DataManager.val_loader)]:
        losses = []
        for i, (X, Y) in enumerate(loader):
            if i >= Hyperparameters.eval_iters:
                break

            X, Y = X.to(Hyperparameters.device, non_blocking=True), Y.to(Hyperparameters.device, non_blocking=True)
            _, loss = model(X, Y)
            losses.append(loss.item())
        out[split] = torch.tensor(losses).mean().item()
    model.train()
    return out

# 5. Fine-tuning loop similar to the pre training loop.
if __name__ == "__main__":
    # initalize the dataloader
    DataManager.load('Conversational')
    train_iterator = cycle(DataManager.train_loader)
    model = Transformer().to(Hyperparameters.device)
    model.load_state_dict(torch.load('AutoEve\\model.pth'), strict=False)
    # freeze the adapters
    for name, param in model.named_parameters():
        if 'adapter' in name:
            param.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=Hyperparameters.learning_rate, weight_decay=Hyperparameters.weight_decay)
    early_stopping = EarlyStopping(patience=10, verbose=True)
    scheduler = PlateauScheduler(
        optimizer,
        steps_per_epoch=Hyperparameters.eval_interval, # every 1000 epochs
        T_0=Hyperparameters.eval_interval * 5,  # this is the number of epochs to warm up for
        patience=2,
        max_lr=Hyperparameters.learning_rate,
        min_lr=1e-6       )
    scaler = GradScaler()

    logging.info(f"{str(sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6)} Million trainable parameters")

    for iter in tqdm(range(Hyperparameters.max_iters), desc="Working 9 to 5!"):
        if iter % Hyperparameters.eval_interval == 0 or iter == Hyperparameters.max_iters - 1:
            losses = estimate_loss()
            logging.info(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            logging.info(f"Step: {iter}, LR: {optimizer.param_groups[0]['lr']:.9f}")
            early_stopping(losses['val'], model)
            os.system('cls')
            context = (torch.tensor((DataManager.tokenizer.encode("<start> <user> How are you? <eve>")), dtype=torch.long).unsqueeze(0).to(Hyperparameters.device))
            generated_text = DataManager.tokenizer.decode(model.generate(context, max_new_tokens=100)[0].tolist())
            logging.info(f"Generated text: {generated_text}")
            
            os.system('cls')

            scheduler.step(val_loss=losses['val'])

            if early_stopping.early_stop:
                break
            if early_stopping.best_model_wts is not None:
                torch.save(early_stopping.best_model_wts, 'AutoEve/temp_model.pth')

        xb, yb = next(train_iterator)

        xb, yb = xb.to(Hyperparameters.device, non_blocking=True), yb.to(Hyperparameters.device, non_blocking=True)
      
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=Hyperparameters.device):
            logits, loss = model(xb, yb)
            #loss = F.cross_entropy(logits.view(-1, DataLoader.vocab_size), Hyperparameters.masked_labels_FULL[yb], ignore_index=-100) # this does not work the reason is because yb is not telling it where to mask I dont want to do it on the fly but may have to

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step_batch()

    if early_stopping.best_model_wts is not None:
        torch.save(early_stopping.best_model_wts, 'AutoEve/fine_tuned_model.pth')

# Save the fine-tuned model
torch.save(early_stopping.best_model_wts, 'AutoEve/fine_tuned_model.pth')

