from phonemizer import Utils, TextToSpectrogramDataset
import math
from collections import Counter
#from gaussian_summation_theory import math_gauss
import os
import subprocess
import logging
import json
import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
scaler = GradScaler(device='cuda')
from tqdm import tqdm
from tqdm import trange
from sklearn.model_selection import KFold

logging.basicConfig(filename='AutoVoice/Logging/model.log', level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

def clear_log(filename):
    open(filename, 'w').close()

def load_params(filename):
    with open(filename, 'r') as F:
        params = json.load(F)
    return params

params = load_params('AutoVoice/Python Files/setup.json')
clear_log('AutoVoice/Logging/model.log')

class Hyperparameters:
    batch_size = params['batch_size']
    block_size = params['block_size']
    max_iters = params['max_iters']
    eval_interval = params['eval_interval']
    learning_rate = params['learning_rate']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        #logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
    eval_iters = params['eval_iters']
    n_embd = params['n_embd']
    n_head = params['n_head']
    n_layer = params['n_layer']
    dropout = params['dropout']
    weight_decay = params['weight_decay']
    n_splits = params['n_splits']
    fine_tune = params['fine_tune']

class PhonemeDataLoader:
    cmudict = Utils.load_json("AutoVoice\Data\All_Saves\cmudict.json")
    char_to_idx = Utils.load_json("AutoVoice\Data\All_Saves\char_to_idx.json")
    phoneme_to_idx = Utils.load_json("AutoVoice\Data\All_Saves\phoneme_to_idx.json")
    vocab_size = len(phoneme_to_idx)
    data_list = Utils.load_metadata("AutoVoice\Recorder\Logging\metadata.csv")

    dataset = TextToSpectrogramDataset(data_list, cmudict, char_to_idx, phoneme_to_idx)
    loader = DataLoader(dataset=dataset, batch_size=Hyperparameters.batch_size, shuffle=True, collate_fn=Utils.collate_fn)

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.1, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_wts = None

    def __call__(self, val_loss, model):
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

        score = -val_loss
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

def get_batch(split, block_size, batch_size):
    '''
    idx = torch.randint(PhonemeDataLoader.phoneme_to_idx, (batch_size,))

    # idx = torch.tensor([i for i in range(len(PhonemeDataLoader.dataset))])
    print(idx)
    
    x = torch.stack([PhonemeDataLoader.phoneme_to_idx[i:i+block_size] for i in idx])
    y = torch.stack([DataLoader.dataset[i][1][:block_size] for i in idx])

    x = x.to(Hyperparameters.device)
    y = y.to(Hyperparameters.device)
    '''
    dataset = PhonemeDataLoader.loader.dataset
    dataset_size = len(dataset)
    idx = torch.randint(0, dataset_size, (batch_size,))

    x = [dataset[i]['phoneme_ids'][:block_size] for i in idx]
    y = [dataset[i]['mel'][:block_size] for i in idx]

    x = [i[:block_size] for i in x ]
    x = pad_sequence(x, batch_first=True, padding_value=0)
    x = x[:, :block_size]

    y = [i[:block_size] for i in y ]
    y = pad_sequence(y, batch_first=True, padding_value=0.0)
    y = y[:, :block_size, :]

    x = x.to(Hyperparameters.device)
    y = y.to(Hyperparameters.device)

    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(Hyperparameters.eval_iters)
        for k in trange(Hyperparameters.eval_iters, desc=f"Estimate Loss For: {split}", leave=False):
            X, Y = get_batch(split, Hyperparameters.block_size, Hyperparameters.batch_size)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
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

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        #x = self.adapter(x)
        return x

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(PhonemeDataLoader.vocab_size, Hyperparameters.n_embd)
        self.position_embedding_table = nn.Embedding(Hyperparameters.block_size, Hyperparameters.n_embd)
        self.blocks = nn.Sequential(*[Block(Hyperparameters.n_embd, Hyperparameters.n_head) for _ in range(Hyperparameters.n_layer)])
        self.ln_f = nn.LayerNorm(Hyperparameters.n_embd)
        self.spectrogram_head = nn.Linear(Hyperparameters.n_embd, 80)
    
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

        spectrogram_output = self.spectrogram_head(x)

        if targets is None:
            return spectrogram_output, None
        
        #print("spectrogram_output.shape:", spectrogram_output.shape)
        #print("spectrogram_output:", spectrogram_output)

        #print("targets.shape:", targets.shape)
        #print("targets:", targets)

        if targets is not None:
            T_min = min(spectrogram_output.shape[1], targets.shape[1])
            spectrogram_output = spectrogram_output[:, :T_min, :]
            targets = targets[:, :T_min, :]

            loss = F.mse_loss(spectrogram_output, targets)
            return spectrogram_output, loss

    def generate(self, idx):
        self.eval()
        with torch.no_grad():
            idx = idx[:, -Hyperparameters.block_size:]
            spectrogram, _ = self.forward(idx)

            # save the spectrogram directly as a txt file
            try:
                save_path = 'AutoVoice/Logging/output.json'
                with open(save_path, 'w') as f:
                    json.dump(spectrogram.tolist(), f)
            except Exception as e:
                print(e)

            # save the spectrogram as a wav file

            save_path = 'AutoVoice/Logging/output.wav'
            Utils.save_to_wav(Hyperparameters.device, spectrogram, save_path)
        return save_path

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
    for parameter in Hyperparameters.__dict__:
        print(f"{parameter}: {Hyperparameters.__dict__[parameter]}")
    print("Loaded...\n")

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

    logging.info(f"{str(sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6)} Million trainable parameters")

    for iter in tqdm(range(Hyperparameters.max_iters), desc="Working 9 to 5!"):
        if iter % Hyperparameters.eval_interval == 0 or iter == Hyperparameters.max_iters - 1:
            losses = estimate_loss()
            logging.info(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            logging.info(f"Step: {iter}, LR: {optimizer.param_groups[0]['lr']:.9f}")
            early_stopping(losses['val'], model)
            os.system('cls')
            if Hyperparameters.device == 'cuda':
                torch.cuda.synchronize()
            memory_debugging()
            context = (torch.tensor(([PhonemeDataLoader.phoneme_to_idx.get(p, 0) for p in ("This is a test of the new text to speech model.").split()]), dtype=torch.long).unsqueeze(0).to(Hyperparameters.device))
            mel_spec = model.generate(context)
            
            os.system('cls')

            scheduler.step(val_loss=losses['val'])

            if early_stopping.early_stop:
                break
            if early_stopping.best_model_wts is not None:
                torch.save(early_stopping.best_model_wts, 'AutoVoice/Data/Models/temp_model.pth')

        xb, yb = get_batch('train', Hyperparameters.block_size, Hyperparameters.batch_size) # xb = tokens (phoneme ids), yb = mel spectrogram frams
      
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=Hyperparameters.device):
            mel_out, loss = model(xb, yb)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step_batch()

    if early_stopping.best_model_wts is not None:
        torch.save(early_stopping.best_model_wts, 'AutoVoice/Data/Models/model.pth')

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
        'vocab_size': PhonemeDataLoader.vocab_size,
        'stoi': PhonemeDataLoader.phoneme_to_idx
    }

    save_params(params, 'AutoVoice/Python Files/params.json')