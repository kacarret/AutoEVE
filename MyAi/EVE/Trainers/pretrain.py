import os
import logging
import json
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
import torch.utils
from tqdm import tqdm
from tqdm import trange
from sklearn.model_selection import KFold

logging.basicConfig(filename='EVE\Logging\pretrain.log', level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

def clear_file(filename):
    open(filename, 'w').close()
clear_file('EVE\Logging\pretrain.log')

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 128 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 1000
learning_rate = 0.001
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 8
n_layer = 8
dropout = 0.2
weight_decay = 0.005
n_splits = 2 # number of folds min is 2 try keeping the max at 5 can go to 10 or even 15

with open('EVE\Data_storage\input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
print(encode(text))


# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
kf = KFold(n_splits=n_splits, shuffle=True)
dataset = data.tolist()

class EarlyStopping:
    def __init__(self, patience=10, verbose=False):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_wts = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model_wts = model.state_dict()
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                logging.warning("Early Stopping Activated!")
                logging.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
                logging.info(f"Best score: {self.best_score}")
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_wts = model.state_dict()
            self.counter = 0

    def load_best_model_wts(self, model):
        model.load_state_dict(self.best_model_wts)

# data loading
def get_batch(split, block_size, batch_size):
    # generate a small batch of data of inputs x and targets y
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in trange(eval_iters, desc=f"Estimate Loss For: {split}", leave=False):
            X, Y = get_batch(train_loader, block_size, batch_size)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

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

class FeedFoward(nn.Module):
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
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class EVE(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

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
        for _ in trange(max_new_tokens, desc="Generating tokens"):
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    logging.info(f"Fold {fold+1}/{n_splits}")

    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)

    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    model = EVE().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    early_stopping = EarlyStopping(patience=10, verbose=True)
    #if the problem seemed to be with the amount of folds then i suggest that we change the lr scheduler to CosineAnnealingLR 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=1e-6)
    scaler = GradScaler()

    logging.info(f"{str(sum(p.numel() for p in model.parameters())/1e6)}M parameters")

    for iter in tqdm(range(max_iters), desc=f"Fold {fold+1} Training"):
        with tqdm(total=1, desc=f"Evaluating Iteration {iter}", leave=False) as pbar:

            if iter % eval_interval == 0 or iter == max_iters - 1:
                losses = estimate_loss()
                early_stopping(losses['val'], model)
                os.system('cls')
                logging.info(f"Fold {fold + 1}; Processing batch {int((iter/eval_interval)+1)} out of {int(max_iters/eval_interval)}")
                logging.info(f"Fold {fold + 1}; Step {iter}; Train Loss: {losses['train']:.4f}, Val Loss: {losses['val']:.4f}, LR: {optimizer.param_groups[0]['lr']:.4f}")

                if early_stopping.early_stop:
                    break
                if early_stopping.best_model_wts is not None:
                    torch.save(early_stopping.best_model_wts, 'EVE\Data_storage\pretrained_model.pth')

            xb, yb = get_batch('train', block_size, batch_size)
            
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                logits, loss = model(xb, yb)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            pbar.update(1)

    if early_stopping.best_model_wts is not None:
        torch.save(early_stopping.best_model_wts, 'EVE\Data_storage\pretrained_model.pth')

#save to json

def save_params(params, filename):
    with open(filename, 'w') as f:
        json.dump(params, f, indent=4)

params = {
    'batch_size': batch_size,
    'block_size': block_size,
    'max_iters': max_iters,
    'eval_interval': eval_interval,
    'learning_rate': learning_rate,
    'device': device,
    'eval_iters': eval_iters,
    'n_embd': n_embd,
    'n_head': n_head,
    'n_layer': n_layer,
    'dropout': dropout,
    'chars': chars,
    'vocab_size': vocab_size,
    'stoi': stoi,
    'itos': itos,
    'n_splits': n_splits
}

save_params(params, 'EVE\Data_storage\pre_trained_params.json')

context = torch.zeros((1, 1), dtype=torch.long, device=device)
test_out = decode(model.generate(context, max_new_tokens=block_size)[0].tolist())
print(f"Test Output: {test_out}")
logging.info(f"Test Output: {test_out}")