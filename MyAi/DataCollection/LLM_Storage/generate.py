import logging
import json
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import trange

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load Model parameters
def load_params(filename):
    with open(filename, 'r') as f:
        params = json.load(f)
    return params

params = load_params('DataCollection/params.json')

# Extract parameters
batch_size = params['batch_size']
block_size = params['block_size']
max_iters = params['max_iters']
num_updates = params['num_updates']
eval_interval = params['eval_interval']
learning_rate = params['learning_rate']
device = 'cpu'  # Hardcoded for simplicity; replace with params['device'] if needed
eval_iters = params['eval_iters']
n_embd = params['n_embd']
n_head = params['n_head']
n_layer = params['n_layer']
dropout = params['dropout']
chars = params['chars']
vocab_size = params['vocab_size']
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l])

# Tokens for special roles
USER_TOKEN = '[user]'
EVE_TOKEN = '[eve]'
END = '[end]'
START = '[start]'
stoi[USER_TOKEN] = len(stoi)
stoi[EVE_TOKEN] = len(stoi)
stoi[END] = len(stoi)
stoi[START] = len(stoi)
itos[stoi[USER_TOKEN]] = USER_TOKEN
itos[stoi[EVE_TOKEN]] = EVE_TOKEN
itos[stoi[END]] = END
itos[stoi[START]] = START

# Verify if <eos> is in the vocabulary
if '<eos>' not in stoi:
    logging.error("End of sequence token '<eos>' is missing in vocabulary.")

def encode_with_role(s, role):
    if role not in stoi:
        raise ValueError(f"Role token '{role}' is not in the vocabulary.")
    encoded_role = stoi[role]
    encoded_text = [stoi.get(c, -1) for c in s]
    if -1 in encoded_text:
        raise ValueError("Some tokens are out of vocabulary.")
    return [encoded_role] + encoded_text

def decode_tokens(tokens):
    tokens = [t.item() if isinstance(t, torch.Tensor) else t for t in tokens]
    logging.debug(f"Tokens to decode: {tokens}")
    
    invalid_tokens = [idx for idx in tokens if idx not in itos]
    if invalid_tokens:
        logging.error(f"Invalid tokens encountered: {invalid_tokens}")
        raise ValueError("Invalid token encountered in decoding.")
    
    return ''.join([itos[idx] for idx in tokens])

# Define Model classes
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
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
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

class EveLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
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
        temp_idx = torch.clamp(idx, max=self.token_embedding_table.num_embeddings - 1)
        tok_emb = self.token_embedding_table(temp_idx)
        pos_indices = torch.arange(min(T, block_size), device=device)  # Fixed position indices
        pos_emb = self.position_embedding_table(pos_indices)
        pos_emb = pos_emb[:T]
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
            return logits, loss

def top_k_filtering(logits, top_k):
    """ Apply top-k filtering to logits. """
    if top_k > 0:
        top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
        top_k_probs = F.softmax(top_k_logits, dim=-1)
        return top_k_indices, top_k_probs
    else:
        return None, F.softmax(logits, dim=-1)

def top_p_filtering(logits, p):
    """ Apply top-p (nucleus) filtering to logits. """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    sorted_indices_to_keep = cumulative_probs <= p
    sorted_indices_to_keep[..., 1:] = sorted_indices_to_keep[..., :-1].clone()
    sorted_indices_to_keep[..., 0] = True
    
    logits[~sorted_indices_to_keep] = -float('Inf')
    return logits

def adjust_temperature(logits, temperature):
    logits = logits / temperature
    return F.softmax(logits, dim=-1)

# Instantiate and load model
model = torch.load('DataCollection\pretrained_model.pth', map_location=device)
model.eval()  # Set the model to evaluation mode

def generate_response(prompt, max_new_tokens=block_size, temperature=1.0, top_k=50, top_p=0.9):
    context = encode_with_role(prompt, USER_TOKEN)
    context = torch.tensor(context, dtype=torch.long).unsqueeze(0).to(device)
    sentence_end_tokens = [stoi.get('.', -1), stoi.get('?', -1), stoi.get('!', -1)]

    for _ in range(max_new_tokens):
        logits = model(context)
        logits = logits[:, -1, :]  # Get logits for the last token only
        
        # Apply temperature scaling
        logits = adjust_temperature(logits, temperature)
        
        # Apply top-k filtering
        top_k_indices, top_k_probs = top_k_filtering(logits, top_k)
        if top_k_indices is not None:
            logits = logits.scatter(-1, top_k_indices, top_k_probs)
        
        # Apply top-p filtering
        logits = top_p_filtering(logits, top_p)
        
        # Sample from the logits
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        context = torch.cat((context, next_token), dim=1)
        
        # Check if the generated token is an end-of-sequence token
        if next_token.item() in sentence_end_tokens:
            break

    return decode_tokens(context[0].tolist())

def chat():
    print("Start chatting with the model (type 'exit' to stop)!")
    while True:
        prompt = input("You: ")
        if prompt.lower() == 'exit':
            break
        response = generate_response(prompt)
        print(f"Eve: {response}")

# Run the chat function
chat()
