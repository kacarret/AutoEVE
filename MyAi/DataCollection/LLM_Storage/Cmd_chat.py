import json
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import trange
import json

# load Model parameters
def load_params(filename):
    with open(filename, 'r') as f:
        params = json.load(f)
        print(params)
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

encode = lambda text: [stoi[c] for c in text if c in stoi]
decode = lambda l: ''.join([itos[str(i)] for i in l])

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
        B, T, C = x.shape  # x: (B, T, C)
        
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        v = self.value(x) # (B, T, head_size)
        
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)

        if T > self.tril.size(0):
            raise ValueError(f"Sequence length {T} exceeds block size {self.tril.size(0)}")
        mask = self.tril[:T, :T]
        wei = wei.masked_fill(mask == 0, float('-inf'))

        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        
        out = wei @ v # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # Concatenate outputs from all heads
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

# Define the model architecture
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
        B, T = idx.shape  # idx: (B, T)
        
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_indices = torch.arange(T, device=device).clamp(max=block_size-1)  # Ensure indices are within range
        pos_emb = self.position_embedding_table(pos_indices)  # (T, C)
        pos_emb = pos_emb[:T]  # Adjust to ensure size matches
        x = tok_emb + pos_emb  # (B, T, C)
        
        x = self.blocks(x)  # (B, T, C)
        x = self.ln_f(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss if loss is not None else logits

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            if idx.size(1) > block_size:
                idx = idx[:, -block_size:]
        return idx

# Instantiate the model
model = EveLanguageModel().to(device)
model.load_state_dict(torch.load('EVE\Data_storage\pretrained_model.pth', map_location=device), strict=False)

def generate_response(prompt, max_new_tokens=block_size):
    context = encode_text(prompt)
    context = context.to(device)  # Ensure context is in the shape (1, T)
    
    sentence_end_tokens = [stoi.get('.', -1), stoi.get('?', -1), stoi.get('!', -1)]

    for _ in trange(max_new_tokens, desc="Percent of Tokens Used: "):
        with torch.no_grad():
            logits, _ = model(context)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            context = torch.cat((context, next_token), dim=1)
            
            # Truncate context if it exceeds block_size
            if context.shape[1] > block_size:
                context = context[:, -block_size:]

            if next_token.item() in sentence_end_tokens: # Break if sentence end token is encountered
                break

            if next_token.item() == stoi.get('<eos>', -1):
                break

    prompt_tokens = context[:, len(prompt):] # Remove prompt from context, however we are running into small issues with the output. Due to the discord api it has a ability to auto restart and reinitalze, this will not work in cmd but for now we will leave this. we almost have no choice.

    response_text = decode_tokens(prompt_tokens[0].tolist())
    return response_text

# Define encode_text and decode_tokens
def encode_text(text):
    encoded = encode(text)
    #print(f"ecoded text: {encoded}")
    #print(f"Encoded text shape: {len(encoded)}")  # Debugging line
    return torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)

def decode_tokens(tokens):
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.tolist()
    #print(f"Decoded tokens: {tokens}")  # Debugging line
    return decode(tokens)

# Chat loop
def chat(user_input):#remove user_input for cmd usage
    #store user_input in a file
    with open('dialoague.txt', 'a') as f:
        f.write(f"[user] {user_input}\n")
        f.close()
    
    context = None #initalize the context
    if user_input.lower() == 'exit':
        exit()

    response_text = generate_response(user_input)
    with open('dialoague.txt', 'a') as f:
        f.write(f"[eve] {response_text}\n")
        f.close()
    return response_text#remove for cmd usage

if __name__ == "__main__":
    chat()