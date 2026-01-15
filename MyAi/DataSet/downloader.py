from datasets import load_dataset
import os
from tqdm import tqdm

file_dir = "DataSet"
dataset = load_dataset("openwebtext", split="train", streaming=True, trust_remote_code=True)

def delete_dataset():
    if os.path.exists(os.path.join(file_dir, "openwebtext.txt")):
        os.remove(os.path.join(file_dir, "openwebtext.txt"))

def check_dataset_size():
    size_in_bytes = os.path.getsize(os.path.join(file_dir, "openwebtext.txt"))
    return size_in_bytes / (1024 ** 3)

def write_dataset():
    with open(os.path.join(file_dir, "openwebtext.txt"), "w", encoding="utf-8") as f:
        for i, example in tqdm(enumerate(dataset), desc="Writing dataset"):
            text = example["text"].strip()
            f.write(text + "\n")
            if check_dataset_size() >= 0.001: # This is measured in GB so 1 = 1 GB
                break

delete_dataset()
write_dataset()