import os
from tokener import EnhancedBPETokenizer
import logging
import json
import torch
import torch.nn as nn
import torch.utils
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
from tqdm import tqdm
from tqdm import trange
from sklearn.model_selection import KFold

logging.basicConfig(filename='Redone Eve Model/temp.log', level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

def clear_log(filename):
    open(filename, 'w').close()
clear_log('Redone Eve Model/temp.log')

def load_params(filename):
    with open(filename, 'r') as f:
        params = json.load(f)
    return params

params = load_params('Redone Eve Model/setup.json')

#hyperparameters
batch_size = params['batch_size']
block_size = params['block_size']
max_iters = params['max_iters']
eval_interval = params['eval_interval']
learning_rate = params['learning_rate']
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = params['eval_iters']
n_embd = params['n_embd']
n_head = params['n_head']
n_layer = params['n_layer']
dropout = params['dropout']
weight_decay = params['weight_decay']
n_splits = params['n_splits']

#load the data
path_file = "EVE\Data_storage\Synth_data.txt"
special_tokens = ['<user>', '<eve>', '<start>', '<end>']

tokenizer = EnhancedBPETokenizer(special_tokens=special_tokens)
with open(path_file, 'r', encoding='utf-8') as f:
    text = f.read()

tokenizer.train(text)

stoi = tokenizer.load_stoi()
encoded_text = tokenizer.encode(text)
vocab_size = tokenizer.load_vocab_size(added_special_tokens=len(special_tokens))

# Train and test splits
data = torch.tensor(encoded_text, dtype=torch.long)
kf = KFold(n_splits=n_splits, shuffle=True)
dataset = data.tolist()

#early stopping and saving
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, verbose=False):
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
            self.best_model_wts = model.state_dict()
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_wts)

# data loading
def get_batch(split, block_size, batch_size):
    """
    Generate a small batch of data of inputs x and targets y.

    Parameters
    ----------
    data : list of int
        The dataset, represented as a list of token IDs.
    stoi : dict
        A mapping from token strings to integer IDs.
    split : str
        Which split of the data to use (train or val).
    block_size : int
        Size of each block of text.
    batch_size : int
        Number of blocks to generate.
    device : str
        The device to store the tensors on (default: 'cuda').

    Returns
    -------
    x : torch.tensor, shape=(batch_size, block_size)
        Inputs to the model.
    y : torch.tensor, shape=(batch_size, block_size)
        Targets for the model.
    """
    # Generate random indices for starting points of the blocks
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # Create the input sequences
    x = torch.stack([data[i:i + block_size] for i in ix])

    # Create the target sequences (shifted by 1)
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])

    # Move tensors to the desired device
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
        """
        Performs a forward pass through a single head of self-attention.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, time-step, channels).

        Returns:
            torch.Tensor: Output tensor of shape (batch, time-step, head size) after applying attention and aggregation.
        """
        B,T,C = x.shape

        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        v = self.value(x) # (B,T,hs)

        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)

        if T > self.tril.size(0):
            raise ValueError(f"Sequence length {T} exceeds block size {self.tril.size(0)}")
        mask = self.tril[:T, :T]
        wei = wei.masked_fill(mask == 0, float('-inf')) # (B, T, T)

        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)

        # perform the weighted aggregation of the values
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
        """
        Compute the output of the MultiHeadAttention module.

        Args:
            x: The input of size (batch, time-step, channels)

        Returns:
            The output of size (batch, time-step, channels)
        """
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        """
        Initializes the FeedFoward module.

        Args:
            n_embd (int): The size of the embedding dimension.

        Returns:
            None
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.layers(x)

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
        """
        Transformer block forward pass.

        Args:
            x: Input tensor with shape (batch_size, sequence_length, embedding_dim)

        Returns:
            Output tensor with shape (batch_size, sequence_length, embedding_dim)
        """
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
        """
        Initialize the weights of the EVE model.

        This function is used to initialize the weights of the model. It takes a
        module as input and initializes the weights of the module using the
        following rules:

        - If the module is an instance of nn.Linear, the weights are initialized
          using a normal distribution with mean 0.0 and standard deviation 0.02.
          If the module has a bias, the bias is initialized to 0.
        - If the module is an instance of nn.Embedding, the weights are
          initialized using a normal distribution with mean 0.0 and standard
          deviation 0.02.

        Args:
            module (nn.Module): The module to initialize.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        Compute the output of the EVE model.

        Args:
            idx (torch.Tensor): Input tensor containing token indices with shape (batch_size, sequence_length).
            targets (torch.Tensor or None): Target tensor containing token indices with shape (batch_size, sequence_length).

        Returns:
            tuple: A tuple of two elements. The first is the output logits tensor with shape (batch_size, sequence_length, vocab_size).
                   The second is the loss tensor or None if targets is None. The loss is a scalar value.
        """

        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device) % block_size) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            return logits, None
        
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)

        loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        Generates a sequence of new tokens, ensuring it doesn't just repeat the user's input.

        Args:
            idx (torch.Tensor): The input tensor containing token indices with shape (batch_size, sequence_length).
            max_new_tokens (int): The maximum number of new tokens to generate.

        Returns:
            torch.Tensor: The tensor containing the original and newly generated token indices with shape (batch_size, sequence_length + max_new_tokens).
        """
        for _ in trange(max_new_tokens, desc="Generating tokens"):
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)

            probs[:, stoi['<start>']] = 0  # Set <start> token probability to 0
            probs[:, stoi['<end>']] = 0  # Set <end> token probability to 0
            probs[:, stoi['<user>']] = 0  # Set <user> token probability to 0
            probs[:, stoi['<eve>']] = 0  # Set <eve> token probability to 0


            idx_next = torch.multinomial(probs, num_samples=1, replacement=True)
            #idx_next = torch.argmax(probs, dim=-1, keepdim=True)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    
def loss_function(logits, targets, stoi):
    """
    This is a loss function for the EVE model. This will include additional penalties for the special tokens.

    Args:
        logits (torch.Tensor): The output logits tensor with shape (batch_size, sequence_length, vocab_size) or (batch_size, vocab_size).
        targets (torch.Tensor): The target tensor containing token indices with shape (batch_size, sequence_length).
        stoi (dict): A dictionary mapping tokens to indices.

    Returns:
        torch.Tensor: The loss value.
    """

    # Check the number of dimensions in logits and targets
    if len(logits.shape) == 3:  # (batch_size, sequence_length, vocab_size)
        B, T, C = logits.shape
    elif len(logits.shape) == 2:  # (batch_size, vocab_size) for a single token
        B, C = logits.shape
        T = 1  # In this case, we assume the sequence length is 1
    else:
        raise ValueError(f"Invalid shape for logits: {logits.shape}")

    if len(targets.shape) == 2:  # (batch_size, sequence_length)
        targets = targets.view(B * T)
    elif len(targets.shape) == 1:  # (batch_size,) for a single token
        targets = targets.view(B)  # Flatten targets to a 1D tensor
    else:
        raise ValueError(f"Invalid shape for targets: {targets.shape}")

    logits = logits.view(B * T, C)

    # Mask out special tokens
    for token in special_tokens:
        if token in stoi:
            mask = (targets != stoi[token])

    #mask = (targets != stoi['<start>']) & (targets != stoi['<end>']) & (targets != stoi['<user>']) & (targets != stoi['<eve>'])

    loss = F.cross_entropy(logits, targets, reduction='none')
    loss = loss * mask.float()
    loss = loss.mean()

    # Calculate length penalty if sequence length is > 1
    if T > 1:
        length_penalty = torch.abs(T - logits.shape[1])
        loss += length_penalty * 0.01

    return loss

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
                    torch.save(early_stopping.best_model_wts, 'Redone Eve Model/temp_model.pth')

            xb, yb = get_batch('train', block_size, batch_size)
            
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                logits, loss = model(xb, yb)
                loss = loss_function(logits, yb, stoi)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            pbar.update(1)

    if early_stopping.best_model_wts is not None:
        torch.save(early_stopping.best_model_wts, 'Redone Eve Model/temp_model.pth')

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
    'weight_decay': weight_decay,
    'n_splits': n_splits,
    'vocab_size': vocab_size,
    'stoi': stoi
}

save_params(params, 'Redone Eve Model/params.json')

context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_text = tokenizer.decode(model.generate(context, max_new_tokens=block_size)[0].tolist())
print(f"Generated text: {generated_text}")
logging.info(f"Generated text: {generated_text}")