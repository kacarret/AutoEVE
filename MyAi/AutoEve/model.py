from BPETokener import WordPairTokenizer
import math
from collections import Counter
#from gaussian_summation_theory import math_gauss
import os
import subprocess
import logging
import json
import numpy as np
import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
scaler = GradScaler(device='cuda')
from tqdm import tqdm
from tqdm import trange
from itertools import cycle
from sklearn.model_selection import KFold

logging.basicConfig(filename='AutoEve/model.log', level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

def clear_log(filename):
    open(filename, 'w').close()

def load_params(filename):
    with open(filename, 'r') as F:
        params = json.load(F)
    return params

params = load_params('AutoEve/setup.json')
clear_log('AutoEve/model.log')

class Hyperparameters:
    batch_size = params['batch_size']
    block_size = params['block_size']
    max_iters = params['max_iters']
    eval_interval = params['eval_interval']
    learning_rate = params['learning_rate']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.cuda.empty_cache()
    eval_iters = params['eval_iters']
    n_embd = params['n_embd']
    n_head = params['n_head']
    n_layer = params['n_layer']
    dropout = params['dropout']
    weight_decay = params['weight_decay']
    n_splits = params['n_splits']
    fine_tune = params['fine_tune']

class DatasetHandler(torch.utils.data.Dataset):
    """Custom Dataset class for handling sequence data. Stores data on CPU then moves sequence to GPU"""
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.block_size]
        y = self.data[idx+1:idx+self.block_size+1]
        return x, y

class DataManager:
    train_loader = None
    val_loader = None
    vocab_size = None
    tokenizer = WordPairTokenizer()
    encoded_text = None
    stoi = None

    @classmethod
    def load(cls, to_load, val_split=0.2):
        if to_load == 'Full':
            path = "DataSet/openwebtext.txt"
        elif to_load == 'Conversational':
            path = "moreShit.txt"
        else:
            raise ValueError("Invalid dataset")

        cls.tokenizer.load_mappings()
        cls.stoi = cls.tokenizer.stoi

        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()

        cls.encoded_text = torch.tensor(
            cls.tokenizer.encode(text),
            dtype=torch.long
        )

        cls.vocab_size = len(cls.stoi)

        dataset = DatasetHandler(
            cls.encoded_text,
            Hyperparameters.block_size
        )

        split = int(len(dataset) * (1 - val_split))
        train_set, val_set = torch.utils.data.random_split(
            dataset, [split, len(dataset) - split]
        )

        cls.train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=Hyperparameters.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=0
        )

        cls.val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=Hyperparameters.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=0
        )

        return cls.train_loader, cls.val_loader

class EarlyStopping:

    # early stopping should work with k_folds as well, so for eahc training split so a val split should look at val and a train split should look at training
    def __init__(self, patience=3, min_delta=0.025, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_wts = None 

    def __call__(self, loss, model):
        """
        Monitors the validation loss and determines if early stopping should be activated.

        Parameters
        ----------
        val_loss : float
            The current validation loss of the model.
        model : torch.nn.Module
            The model being trained, used to save the best performing model weights.

        Updates
        -------
        self.best_score : float
            Stores the best (lowest) validation score encountered so far.
        self.best_model_wts : dict
            Stores the state dictionary of the model with the best validation score.
        self.counter : int
            Counts the number of consecutive non-improving epochs.
        self.early_stop : bool
            Flag to indicate if early stopping should be triggered.

        Condition
        ---------
        If the validation loss does not improve by at least `min_delta` for `patience`
        consecutive epochs, early stopping is activated.
        """
        score = -loss
        if self.best_score is None:
            self.best_score = score
            self.best_model_wts = model.state_dict()
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                logging.warning("Early Stopping Activated!")
                logging.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
                logging.info(f"Best score: {self.best_score}")
                self.early_stop = True
        else:
            self.best_score = score
            logging.info(f"New best score: {self.best_score}")
            self.best_model_wts = model.state_dict()
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_wts)

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

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(Hyperparameters.n_embd, head_size, bias=False)
        self.query = nn.Linear(Hyperparameters.n_embd, head_size, bias=False)
        self.value = nn.Linear(Hyperparameters.n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(Hyperparameters.block_size, Hyperparameters.block_size)))

        self.dropout = nn.Dropout(Hyperparameters.dropout)

    def forward(self, x):

        B, T, C = x.shape

        k = self.key(x)   # (B, T, hs)
        q = self.query(x) # (B, T, hs)
        v = self.value(x) # (B, T, hs)

        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)

        if T > self.tril.size(0):
            raise ValueError(f"Sequence length {T} exceeds block size {self.tril.size(0)}")
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)

        wei = self.dropout(wei)

        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MHA(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, Hyperparameters.n_embd)
        self.dropout = nn.Dropout(Hyperparameters.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        """
        my math is not good, however look at the old calc 2 book last pages for the math behind 
        why it works like this I will write the comparitive values inside of the summation 
        theroy python file
        """
        # return(0.5 * x * (1 + math_gauss(x)))
        # torch.erf is the gaussian error function ::FACEPALM::
        intergral = torch.erf(x / math.sqrt(2.0))
        return(0.5 * x * (1 + (2 / (math.sqrt(math.pi)) * intergral)))
 
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.linear1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear2 = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(Hyperparameters.dropout)
        self.smartness = nn.GELU()
        #self.smartness = GELU() # i dont need this i dont really like it
        #self.smartness = nn.SiLU()
        #self.smartness = nn.ReLU()
    def forward(self, x):
        x = self.linear1(x)
        x = self.smartness(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MHA(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        self.adapter = LayerAdapter(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        x = self.adapter(x)
        return x

#this is a layer for conversation
class ConversationalHead(nn.Module):
    def __init__(self, n_embd, block_size):
        super().__init__()
        self.linear = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(Hyperparameters.dropout)
    
    def forward(self, x):
        return self.dropout(self.linear(x))

# layer adapters to help alining the model to a given task this task should be converastional
class LayerAdapter(nn.Module):
    def __init__(self, dim, bottleneck_dim=32):
        super().__init__()
        self.down_proj = nn.Linear(dim, bottleneck_dim)
        self.activation = GELU()
        self.up_proj = nn.Linear(bottleneck_dim, dim)
        self.dropout = nn.Dropout(Hyperparameters.dropout)
    
    def forward(self, x):
        return x + self.dropout(self.up_proj(self.activation(self.down_proj(x))))

class Transformer(nn.Module):
    def __init__(self, use_conversational_head=False):
        super().__init__()
        logging.info(f"Initializing Transformer Model with vocab size {DataManager.vocab_size}")
        self.token_embedding_table = nn.Embedding(DataManager.vocab_size, Hyperparameters.n_embd)
        self.position_embedding_table = nn.Embedding(Hyperparameters.block_size, Hyperparameters.n_embd)
        self.blocks = nn.Sequential(*[Block(Hyperparameters.n_embd, Hyperparameters.n_head) for _ in range(Hyperparameters.n_layer)])
        self.ln_f = nn.LayerNorm(Hyperparameters.n_embd)
        self.lm_head = nn.Linear(Hyperparameters.n_embd, DataManager.vocab_size)
        self.conversational_head = ConversationalHead(Hyperparameters.n_embd, Hyperparameters.block_size)
        self.use_conversational_head = use_conversational_head

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
        pos_emb = self.position_embedding_table(torch.arange(T, device=Hyperparameters.device) % Hyperparameters.block_size)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)

        if self.use_conversational_head:
            x = self.conversational_head(x)

        logits = self.lm_head(x)

        if targets is None:
            return logits, None

        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)

        loss = F.cross_entropy(logits, targets)

        #loss = F.cross_entropy(logits, targets, ignore_index=-100) # idk what this is

        return logits, loss

    def generate(self, idx, max_new_tokens, mem_enabled=False, top_k=30, temperature=0.7, penalty=0.3):
        idxout = []
        # once per turn
        idx = torch.cat([torch.tensor([[DataManager.stoi['<eve>']]]).to(idx.device), idx], dim=1)


        def save_short_term_memory(short_term_memory):
            with open('short_term_memory.json', 'w') as f:
                json.dump(short_term_memory[0].tolist(), f)

        def load_short_term_memory():
            with open('short_term_memory.json', 'r') as f:
                return torch.tensor(json.load(f), dtype=torch.long).unsqueeze(0).to(Hyperparameters.device)

        def top_k_filtering(logits, top_k):
            values, _ = torch.topk(logits, top_k)
            min_top_k = values[:, -1].unsqueeze(1)
            logits[logits < min_top_k] = float('-inf')
            return logits

        # ---- load memory ----
        if mem_enabled:
            try:
                short_term_memory = load_short_term_memory()
                idx = torch.cat((short_term_memory, idx), dim=1)
            except Exception as e:
                logging.error(f"Memory load failed: {e}")

        special_tokens = {
            DataManager.stoi['<end>'],
            DataManager.stoi['<user>'],
            DataManager.stoi['<eve>'],
            DataManager.stoi['<start>']
        }

        temperature = max(temperature, 1e-6)

        with torch.no_grad():
            for _ in tqdm(range(max_new_tokens), desc="Generating..."):

                if idx.size(1) >= Hyperparameters.block_size:
                    idx = idx[:, -Hyperparameters.block_size:]

                logits, _ = self(idx)
                logits = logits[:, -1, :]

                # ---- repetition penalty (recent window only) ----
                if penalty > 0 and idx.size(1) > 1:
                    recent_tokens = idx[0, -50:].tolist()
                    for token in set(recent_tokens):
                        logits[0, token] -= penalty

                # ---- temperature ----
                logits = logits / temperature

                # ---- top-k ----
                if top_k is not None:
                    logits = top_k_filtering(logits, top_k)

                # ---- ban special tokens instead of breaking ----
                logits[0, list(special_tokens)] = -float('inf')

                probs = F.softmax(logits, dim=-1)

                # ---- sampling ----
                idx_next = torch.multinomial(probs, num_samples=1)

                idxout.append(idx_next.item())
                idx = torch.cat((idx, idx_next), dim=1)

        if mem_enabled:
            save_short_term_memory(idx[:, -Hyperparameters.block_size:])

        return torch.tensor(idxout, dtype=torch.long).unsqueeze(0).to(Hyperparameters.device)
    
class PlateauScheduler:
    def __init__(self, optimizer, steps_per_epoch, T_0=1000, T_mult=2,
                 patience=1000, max_lr=1e-4, min_lr=1e-6, min_delta=0.05, verbose=False):
        """
        CosineAnnealingWarmRestarts (batch-wise) with validation loss-based resets.

        Args:
            optimizer: The optimizer to schedule.
            steps_per_epoch: Number of steps in one epoch (used to tune T_0).
            T_0: Initial number of steps before restart.
            T_mult: Multiplicative factor for increasing T_i after a restart.
            patience: Validation steps to wait before triggering a restart on plateau.
            max_lr: Reset LR.
            min_lr: Minimum LR in schedule.
            min_delta: Minimum improvement in validation loss to avoid triggering reset.
            verbose: Whether to log resets.
        """
        self.optimizer = optimizer
        self.steps_per_epoch = steps_per_epoch
        self.T_0 = T_0
        self.T_mult = T_mult
        self.patience = patience
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.verbose = verbose

        self.best_val_loss = float('inf')
        self.batches_since_update = 0
        self.global_step = 0

        self._init_scheduler()

    def _init_scheduler(self):
        for group in self.optimizer.param_groups:
            group['lr'] = self.max_lr
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.T_0,
            T_mult=self.T_mult,
            eta_min=self.min_lr
        )
        self.global_step = 0

    def step_batch(self):
        self.scheduler.step(self.global_step)
        self.global_step += 1

    def step(self, val_loss: float):
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.batches_since_update = 0
        else:
            self.batches_since_update += 1

        if self.batches_since_update >= self.patience:
            if self.verbose:
                logging.info(f"[Scheduler] Plateau detected. Resetting LR to {self.max_lr:.2e}")
            self._init_scheduler()
            self.batches_since_update = 0
            self.best_val_loss = val_loss

    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

def memory_debugging():
    if Hyperparameters.device == 'cuda':
        logging.info("_____________________________Start of memory debugging____________________________")
        logging.info(f"GPU memory allocated: {round(torch.cuda.memory_allocated() / 1024**3)} GB")
        logging.info(f"GPU memory reserved: {round(torch.cuda.memory_reserved() / 1024**3)} GB")
        logging.info(f"GPU memory peak: {round(torch.cuda.max_memory_allocated() / 1024**3)} GB")
        logging.info(f"GPU memory peak allocated: {round(torch.cuda.max_memory_allocated() / 1024**3)} GB")
        logging.info(f"GPU memory peak reserved: {round(torch.cuda.max_memory_reserved() / 1024**3)} GB")
        logging.info("______________________________End of memory debugging_____________________________")

if __name__ == "__main__":

    DataManager.load(to_load='Full', val_split=0.2)

    for fold in range(Hyperparameters.n_splits):
        logging.info(f"Fold {fold + 1}/{Hyperparameters.n_splits}")

        model = Transformer().to(Hyperparameters.device)
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

        train_iterator = cycle(DataManager.train_loader)

        logging.info(f"{str(sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6)} Million trainable parameters")

        for it in tqdm(range(Hyperparameters.max_iters), desc="Working 9 to 5!"):
            if it % Hyperparameters.eval_interval == 0 or it == Hyperparameters.max_iters - 1:
                losses = estimate_loss()
                logging.info(f"Step {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                logging.info(f"Step: {it}, LR: {optimizer.param_groups[0]['lr']:.9f}")
                if (fold+1) == 1: # we have no real way to do this
                    early_stopping(losses['train'], model)
                else:
                    early_stopping(losses['val'], model)
                os.system('cls')
                #memory_debugging() # for checking memory management
                context = (torch.tensor((DataManager.tokenizer.encode("<start> <user> How are you? <eve>")), dtype=torch.long).unsqueeze(0).to(Hyperparameters.device))
                generated_tokens = model.generate(context, max_new_tokens=100)[0].tolist()
                generated_text = DataManager.tokenizer.decode(generated_tokens)
                logging.info(f"Generated tokens: {generated_tokens}")
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

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step_batch()

        if early_stopping.best_model_wts is not None:
            torch.save(early_stopping.best_model_wts, 'AutoEve/model.pth')

    def save_params(params, filename):
        with open(filename, 'w') as f:
            json.dump(params, f, indent=4)

    params = {
        'batch_size': Hyperparameters.batch_size,
        'block_size': Hyperparameters.block_size,
        'max_iters': Hyperparameters.max_iters,
        'eval_interval': Hyperparameters.eval_interval,
        'learning_rate': Hyperparameters.learning_rate,
        'device': Hyperparameters.device,
        'eval_iters': Hyperparameters.eval_iters,
        'n_embd': Hyperparameters.n_embd,
        'n_head': Hyperparameters.n_head,
        'n_layer': Hyperparameters.n_layer,
        'dropout': Hyperparameters.dropout,
        'weight_decay': Hyperparameters.weight_decay,
        'n_splits': Hyperparameters.n_splits,
        'vocab_size': DataManager.vocab_size,
        'stoi': DataManager.stoi
    }

    save_params(params, 'AutoEve/params.json')

    if Hyperparameters.fine_tune == True:
        logging.info("Starting fine tuning...\n\n")
        #subprocess.run(['python', 'AutoEve\modelFineTuner.py'])
