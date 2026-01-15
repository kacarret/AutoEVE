import os
import time
import re
import json
import logging
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import KFold
from tqdm import tqdm

scaler = GradScaler()
logging.basicConfig(filename='EVE\Logging\post_trainer.log', level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

def clear_file(filename):
    open(filename, 'w').close()
clear_file('EVE\Logging\post_trainer.log')

def load_params(filename):
    with open(filename, 'r') as f:
        params = json.load(f)
        #print(params)
    return params

params = load_params('EVE\Data_storage\pre_trained_params.json')

batch_size = params['batch_size']
block_size = params['block_size']
max_iters = params['max_iters']
eval_interval = params['eval_interval']
learning_rate = params['learning_rate']
device = params['device']
eval_iters = params['eval_iters']
n_embd = params['n_embd']
n_head = params['n_head']
n_layer = params['n_layer']
dropout = params['dropout']
chars = params['chars']
vocab_size = params['vocab_size']
stoi = params['stoi']
itos = params['itos']
n_splits = params['n_splits']

encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda l: ''.join([itos[str(i)] for i in l])
weight_decay = 0.01

USER_TOKEN = '[user]'
EVE_TOKEN = '[eve]'

stoi[USER_TOKEN] = len(stoi)
stoi[EVE_TOKEN] = len(stoi)
itos[stoi[USER_TOKEN]] = USER_TOKEN
itos[stoi[EVE_TOKEN]] = EVE_TOKEN

kf = KFold(n_splits=n_splits, shuffle=True)

def encode_with_role(s, role):
    default_value = stoi.get('UNKOWN', 0)
    return [stoi[role]] + [stoi.get(c, default_value) for c in s]

pretrained_model_path = 'EVE\Data_storage\pretrained_model.pth'
pretrained_model_weights = torch.load(pretrained_model_path, map_location=device)
personality_dataset = 'EVE\Data_storage\Synth_data.txt'

#read text file and process the new tokens that I made cuz im a bitch
def read_text_file(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
    return text

def split_conversation_snippets(text):
    #Regex to extract everything between [start] and [end], including [user] and [eve] parts
    conversation_pattern = r'\[start\](.*?)\[end\]'
    snippets = re.findall(conversation_pattern, text, re.DOTALL)
    return snippets

def extract_user_and_eve_parts(snippet):
    #Ensure the snippet is stripped of start and end markers first
    snippet = snippet.strip()
    
    #Extracting user's parts: content between [user] and [eve]
    user_parts = re.findall(r'\[user\](.*?)\[eve\]', snippet, re.DOTALL)
    
    #Extracting Eve's parts: content between [eve] and either next [user] or the end
    eve_parts = re.findall(r'\[eve\](.*?)(?=\[user\]|\Z)', snippet, re.DOTALL)
    
    return user_parts, eve_parts

def preprocess_snippets(snippets):
    preprocessed_snippets = []
    for snippet in snippets:
        user_parts, eve_parts = extract_user_and_eve_parts(snippet)
        
        preprocessed_snippet = []
        min_len = min(len(user_parts), len(eve_parts))
        
        #Ensure that we pair up user and Eve's parts
        for i in range(min_len):
            preprocessed_snippet.append((user_parts[i], eve_parts[i]))
            #Optional: print each pair of user and Eve's dialogue
            #print(f"User: {user_parts[i]}")
            #print(f"Eve: {eve_parts[i]}")
        
        preprocessed_snippets.append(preprocessed_snippet)
    
    return preprocessed_snippets


def create_dataset(snippets):
    dataset = []
    for snippet in snippets:
        for user_part, eve_part in snippet:
            user_part_encoded = encode_with_role(user_part, USER_TOKEN)
            eve_part_encoded = encode_with_role(eve_part, EVE_TOKEN)
            dataset.append((user_part_encoded, eve_part_encoded))
            #Optional: print encoded values
            #print(f"User Encoded: {user_part_encoded}\nEve Encoded: {eve_part_encoded}")
    
    return dataset


def collate_fn(batch):
    user_parts, eve_parts = zip(*batch)
    
    #Convert lists to tensors (creating new tensors from lists of ints, not tensors)
    user_parts_padded = pad_sequence([torch.tensor(x, dtype=torch.long) if not isinstance(x, torch.Tensor) else x for x in user_parts], batch_first=True, padding_value=0)
    eve_parts_padded = pad_sequence([torch.tensor(x, dtype=torch.long) if not isinstance(x, torch.Tensor) else x for x in eve_parts], batch_first=True, padding_value=0)
    
    return user_parts_padded, eve_parts_padded

text = read_text_file(personality_dataset)
snippets = split_conversation_snippets(text)
preprocessed_snippets = preprocess_snippets(snippets)
dataset = create_dataset(preprocessed_snippets)

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user_part, eve_part = self.data[idx]
        return torch.tensor(user_part, dtype=torch.long), torch.tensor(eve_part, dtype=torch.long)

def prepare_dataloader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


#Prepare datasets
fine_tune_dataset = create_dataset(preprocessed_snippets)
dataset = TextDataset(fine_tune_dataset)
data_loader = prepare_dataloader(dataset, batch_size=batch_size)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

def freeze_layers(model, layer_names, freeze=True):
    layers_frozen = 0
    layers_unfrozen = 0

    for name, param in model.named_parameters():
        time.sleep(10)
        if any(layer_name in name for layer_name in layer_names):
            if freeze:
                param.requires_grad = False
                layers_frozen += 1
                logging.info(f"Frozen layer: {name}")
            else:
                param.requires_grad = True
                layers_unfrozen += 1
                logging.info(f"Unfrozen layer: {name}")

    if freeze:
        logging.info(f"Total layers frozen: {layers_frozen}")
    else:
        logging.info(f"Total layers unfrozen: {layers_unfrozen}")

    logging.info(f"Frozen layers: {layers_frozen}")
    logging.info(f"Unfrozen layers: {layers_unfrozen}")

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in tqdm(range(eval_iters), desc=f"Estimating loss for {split}", leave=False):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class EarlyStopping:
    def __init__(self, patience=10, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.best_score = None
        self.early_stop = False
        self.counter = 0
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

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_wts)

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropoits = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out
    
class FeedForward(nn.Module):
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
    def __init__(self, n_embd, n_head):
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

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
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
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        
        x = self.fc3(x)

        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in tqdm(range(max_new_tokens)):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            if idx.size(1) >= block_size:  #For example, set a max length to avoid too long sequences
                break
        return idx

for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    logging.info(f"Fold {fold+1}/{n_splits}")

    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)

    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    model = Transformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    early_stopping = EarlyStopping(patience=10, verbose=True)
    #if the problem seemed to be with the amount of folds then i suggest that we change the lr scheduler to CosineAnnealingLR 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=1e-6)
    scaler = GradScaler()

    layer_names = {
        'position_embedding_table'
        }
    
    #freeze layers does not seem to work not like it matters as basically position embedding is hyperparameters and we already are passing those through the model.
    #freeze_layers(model, layer_names)

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
                    torch.save(early_stopping.best_model_wts, 'EVE/Data_storage/fine_tuned_model.pth')

            xb, yb = get_batch('train')
            
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
        torch.save(early_stopping.best_model_wts, 'EVE/Data_storage/fine_tuned_model.pth')

def save_params(params, filename):
    with open(filename, 'w') as f:
        json.dump(params, f, indent=4)

params = {
    "batch_size": batch_size,
    "block_size": block_size,
    "max_iters": max_iters,
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
    "itos": itos,
    "n_splits": n_splits
}

save_params(params, 'EVE/Data_storage/fine_tuned_params.json')
