import os
import regex as re
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import trange

from model import Transformer 
from model import DataLoader
from model import Hyperparameters

#initalize SOMETHING does not matter what it is
DataLoader.load('Synth')

# Load the generation tool from the model import
model = Transformer(use_conversational_head=False).to(Hyperparameters.device)
#model.load_state_dict(torch.load('Redone Eve Model\\model.pth'), strict=False)
#model.load_state_dict(torch.load('Redone Eve Model\\temp_model.pth'), strict=False)
model.load_state_dict(torch.load('Redone Eve Model\\fine_tuned_model.pth'), strict=False)
model.eval()

def store_user_input(user_input):
    with open('dialoague.txt', 'a') as f:
        f.write(f"<user> {user_input}\n")
        f.close()

def store_response(response_text):
    with open('dialoague.txt', 'a') as f:
        f.write(f"<eve> {response_text}\n")
        f.close()

def sleepmode():
    if os.path.exists("short_term_memory.json"):
        os.remove("short_term_memory.json")
    exit()
# Chat loop
def chat(name, minput, mem_enabled, max_tokens):#remove user_input for cmd usage 
    if name == None:
        name = "Unknown"

    #context = load_context()
    if minput.lower() == 'exit':
        sleepmode()

    minput = minput+"\n<eve> "

    #user_input = (context + ("<user> " + user_input))
    minput = ("<user> " + minput)
    minput = DataLoader.tokenizer.encode(minput)
    context = (torch.tensor(minput, dtype=torch.long).unsqueeze(0).to(Hyperparameters.device))
    response_text = DataLoader.tokenizer.decode(model.generate(idx=context, max_new_tokens=max_tokens, mem_enabled=mem_enabled)[0].tolist())
    if "<exc>" in response_text:
        sleepmode()
    response_text = response_text.replace("<name>", name)
    return response_text#remove for cmd usage

if __name__ == "__main__":
    while True:
        print(chat(None, minput=input("input: "), mem_enabled=True, max_tokens=100))