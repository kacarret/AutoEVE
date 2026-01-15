import os
import json
import logging
import numpy as np
from tqdm import tqdm
from tqdm import trange
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

will_initialize = input("Would you like the program to initalize the Post Trainer on Completion? y/n: ").lower().strip() == 'y'
will_shutdown = input("Would you like the program to Shutdown on Completion? y/n: ").lower().strip() == 'y'

# set up logging
logging.basicConfig(filename='pretrain.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel? 16, 32, 64, 128, 256, 512 These are divisable by the block_size by 8
block_size = 128 # what is the maximum context length for predictions? 128, 256, 512, 1024, 2048, 4096
max_iters = 100000
num_updates = 5 # the number of updates will dynamically change the learning rate depending on the number of max iters this is set to 10 while max iters are 10000 meaning every 1000 iters the learning rate will decrease by a multipule of 0.1 (not exact changed the gamma to 0.5)
eval_interval = 100
learning_rate = 1e-5 #dont touch anymore use num update instead 
device = 'cuda' if torch.cuda.is_available() else 'cpu' 
eval_iters = 200
n_embd = 384 #384, 512, or even 768
n_head = 6 # set these to equal and 6 and 8 are nomral
n_layer = 6
dropout = 0.3
weight_decay = 0.005
raise_lr_factor = 2.0
min_improvment = max(1e-5, 0.01)

#print(f"current device in use: {device}") #this is for debugging the cuda/cpu usage.

with open('DataCollection\input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    """Encode a string into a list of integers."""
    return [stoi[c] for c in s]

def decode(l):
    """Decode a list of integers into a string."""
    return ''.join(itos[i] for i in l)

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss_for_model(model, split):
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in trange(eval_iters, desc=f"Estimating loss for {split}", leave=False):
        X, Y = get_batch(split)
        _, loss = model(X, Y)
        losses[k] = loss.item()
    model.train()
    return losses.mean()

@torch.no_grad()
def estimate_loss():
    losses = {}
    for split in ['train', 'val']:
        split_losses = torch.stack(estimate_loss_for_model(model, split))
        losses[split] = split_losses.mean().item()
    return losses

class EarlyStopping:
    def __init__(self, patience=10, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_model_wts = None

    def __call__(self, val_loss, model):
        logging.info(f"val score: {val_loss:.4f}")

        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_wts = model.state_dict()  # Save initial weights
        elif val_loss < self.best_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model_wts = model.state_dict()  # Save new best weights
            self.counter = 0

    def improved(self, val_loss, model):
        logging.info(f"New best score: {val_loss:.4f}")

        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_wts = model.state_dict()  # Save initial weights
            return True#imrovment 
        elif val_loss < self.best_loss:
            self.best_loss= val_loss
            self.best_model_wts = model.state_dict()  # Save new best weights
            self.counter = 0
            return True #imrovemnt made
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logging.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            return False #No Improvement

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_wts)

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class EveLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.fc3 = nn.Linear(n_embd, n_embd)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        x = self.fc3(x)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in tqdm(range(max_new_tokens), desc="Generating tokens"):
            logits_list = []
            logits, _ = model(idx)
            logits_list.append(logits[:, -1, :])
            #time to average the logits
            logits = torch.mean(torch.stack(logits_list), dim=0)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# create a PyTorch optimizer
#ensemble_models = [EveLanguageModel().to(device) for _ in range(ensemble_size)]
#for model in ensemble_models:
    #model.apply(model._init_weights)

model = EveLanguageModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay) 
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, threshold=1e-6, verbose=False)
early_stopping = EarlyStopping(patience=10, verbose=True)
scalers = GradScaler()
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

for iter in range(max_iters):
    # every so often (every eval_interval iterations) we evaluate the loss on the train and val sets
    # and save the model if it's better than the best so far
    # this is so we can keep track of how well the model is doing and make sure it's not overfitting
    with tqdm(total = 1, desc = f"Evaluating iteration {iter}", leave = False) as pbar:

        # if we're at the end of the training loop or at an eval interval, evaluate the loss on the train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            # estimate the loss on the train and val sets
            losses = {split: torch.mean(estimate_loss_for_model(model, split)) for split in ['train', 'val']}
            
            # get the validation loss
            val_loss = [losses['val'].item()]
            
            # print the validation loss
            print("val losses: ", -val_loss[0])
            
            # check if the model is better than the best so far
            # if it is, save the model and update the best loss
            early_stopping(val_loss[0], model)
            
            # clear the screen
            os.system('cls') #window clearing
            
            # log the current step and the average train and val loss
            logging.info(f"Processing batch {int(iter/eval_interval)} out of {int(max_iters/eval_interval)}")
            logging.info(f"Step {iter}: Average Train Loss {losses['train']:.4f}, Average Val Loss {losses['val']:.4f}")
            
            # get the current learning rate
            lr = optimizer.param_groups[0]['lr']
            
            # log the current learning rate
            logging.info(f"Step {iter}, learning rate: {lr}")
            
            # if we've reached the patience limit, stop training
            if early_stopping.early_stop:
                print("Early stopping")
                break
            # if the model is better than the best so far, save it
            if early_stopping.best_model_wts is not None:
                torch.save(early_stopping.best_model_wts, 'DataCollection/pretrained_model.pth')

            # increase the learning rate if the model is improving
            if iter % eval_interval == 0:
                # calculate how much the model has improved
                improvement = early_stopping.best_loss - val_loss[0]
                # if the improvement is small, increase the learning rate
                if improvement < min_improvment:
                    for param_group in optimizer.param_groups:
                        if param_group['lr'] is not None:
                            # increase the learning rate by a factor of raise_lr_factor
                            # but don't go above 1e-3
                            param_group['lr'] = min(param_group['lr'] * raise_lr_factor, 1e-3) 
                            logging.info(f"Increased learning rate to {param_group['lr']}")
                        
        # sample a batch of data and move it to the GPU
        try:
            xb, yb = get_batch('train')
            xb, yb = xb.to(device), yb.to(device)
        except Exception as e:
            logging.error(f"Error sampling batch: {e}")
            continue

        # set the model to training mode
        model.train()
        
        # zero the gradients of the model's parameters
        optimizer.zero_grad(set_to_none=True)

        # evaluate the loss with autocast
        with autocast():
            try:
                logits, loss = model(xb, yb)
            except Exception as e:
                logging.error(f"Error evaluating model: {e}")
                continue

        # check for nans
        if torch.isnan(loss) or torch.isinf(loss):
            logging.error(f"Loss is nan or inf: {loss}")
            continue

        # backward pass with grad scaler
        try:
            scaler.scale(loss).backward()
        except Exception as e:
            logging.error(f"Error scaling loss: {e}")
            continue

        # check for nans this is more for debugging
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    logging.warning(f"NaN or Inf in gradients for {name}")
                    break
        try:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        except Exception as e:
            logging.error(f"Error clipping gradients: {e}")
            continue

        # step the optimizer
        try:
            scaler.step(optimizer)
            scaler.update()
            scheduler.step(losses['val'])
        except Exception as e:
            logging.error(f"Error stepping optimizer: {e}")
            continue

        # update the tqdm bar
        pbar.update(1)
if early_stopping.best_model_wts is not None:
    torch.save(early_stopping.best_model_wts, 'DataCollection/pretrained_model.pth')

def save_params(params, filename):
    with open(filename, 'w') as f:
        json.dump(params, f, indent=4)

params = {
    "batch_size": batch_size,
    "block_size": block_size,
    "max_iters": max_iters,
    "num_updates": num_updates,
    "eval_interval": eval_interval,
    "learning_rate": learning_rate,
    "device": device,
    "eval_iters": eval_iters,
    "n_embd": n_embd,
    "n_head": n_head,
    "n_layer": n_layer,
    "dropout": dropout,
    "chars": chars,
    "vocab_size": vocab_size,
    "stoi": stoi,
    "itos": itos
}

save_params(params, 'DataCollection/params.json')

if will_initialize:
    import post_trainer
elif will_shutdown:
    #os.system('shutdown /s /t 0')
    pass
else:
    logging.info("Training completed!... Doing Nothing... As requested...")