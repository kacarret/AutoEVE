from difflib import SequenceMatcher
import regex as re
import os
import json
from tqdm import tqdm
from collections import Counter

class WordPairCollector:
    def __init__(self, min_freq=None):
        self.pair_freqs = Counter()
        self.min_freq = min_freq

    def set_min_freq(self):
        """Set min_freq if not provided."""
        max_freq = max(self.pair_freqs.values())
        length_pairs = len(self.pair_freqs)
        self.min_freq = length_pairs // max_freq

    def preprocess(self, text):
        """Extract unique words from text."""
        pattern = re.compile(r"\p{L}+|\p{N}+", re.UNICODE)
        words = set(re.findall(pattern, text.lower()))
        return words
    
    def repetitive_str_checker(self, word, min_block=3, max_block=20, repetitions=2, similarity_threshold=0.85):
        """Check if a string is made of repetitive substrings."""
        n = len(word)

        for size in range(min_block, min(max_block, n // repetitions) + 1):

            repeats = 0

            for i in range(0, n - size + 1, size):
                block = word[i:i + size]
                next_block = word[i + size:i + 2 * size]

                if SequenceMatcher(None, block, next_block).ratio() >= similarity_threshold:
                    repeats += 1
                    if repeats >= repetitions:
                        return True
                    
        return False
        

    def generate_pairs(self, word):
        """Generate all possible consecutive substring pairs for a word."""
        pairs = []
        n = len(word)
        for i in range(n - 1):
            for j in range(i + 1, n + 1):
                sub = word[i:j]
                pairs.append(sub)
        return pairs

    def collect_pairs(self, text):
        """Main method: collect pair frequencies."""
        words = self.preprocess(text)

        for word in tqdm(words, desc="Collecting word pairs"):
            if self.repetitive_str_checker(word):
                continue

            pairs = self.generate_pairs(word)
            self.pair_freqs.update(pairs)

        if self.min_freq is None:
            self.set_min_freq()

        filtered = {pair: freq for pair, freq in self.pair_freqs.items() if freq > self.min_freq}

        self.pair_freqs = sorted(Counter(filtered).items(), key=lambda x: x[1], reverse=True)
        self.pair_freqs = Counter(dict(self.pair_freqs)) # convert to Counter

    def save(self, path="word_pairs.json"):
        if os.path.exists(path):
            os.remove(path)
            
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.pair_freqs, f, indent=4, ensure_ascii=False)

    def process_files(self, file_paths):
        """Read all files and collect pairs from all."""
        text = ""
        for file in tqdm(file_paths, desc="Reading files"):
            with open(file, 'r', encoding='utf-8') as f:
                text += f.read()
        self.collect_pairs(text)
        self.save()

class WordPairTokenizer:
    def __init__(self, special_tokens=None, word_pairs_path="word_pairs.json"):
        self.special_tokens = special_tokens or []
        self.word_pairs_path = word_pairs_path
        self.vocab = set()
        self.freqs = {}
        self.stoi = {}
        self.itos = {}
        self.punctuations = set(r"""!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~…“”‘’""")

    def load_word_pairs(self):
        if not os.path.exists(self.word_pairs_path):
            raise FileNotFoundError(f"{self.word_pairs_path} not found")
        with open(self.word_pairs_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # data is expected to be {pair: freq, ...}
        self.freqs = {k: int(v) for k, v in data.items()}
        self.vocab.update(data.keys())

    def build_vocab_from_corpus(self, file_paths):
        pattern = re.compile(r"\p{L}+|\p{N}+|[^\s\p{L}\p{N}]+", re.UNICODE)
        for file in tqdm(file_paths, desc="Building vocab from corpus"):
            with open(file, "r", encoding="utf-8") as f:
                text = f.read().lower()
            words = set(re.findall(pattern, text))
            self.vocab.update(words)
        # add special tokens and punctuations
        self.vocab.update(self.special_tokens)
        self.vocab.update(self.punctuations)

    def build_mappings(self):
        # Sort by frequency descending, then alphabetically
        sorted_vocab = sorted(self.vocab, key=lambda x: (-self.freqs.get(x, 0), x))
        self.stoi = {token: i for i, token in enumerate(sorted_vocab, start=0)}
        self.itos = {i: token for token, i in self.stoi.items()}

    def tokenize(self, text):
        text = text.lower()
        tokens = []
        i = 0
        n = len(text)
        while i < n:
            match = None
            max_len = min(n - i, 50)  # max token length limit
            for l in range(max_len, 0, -1):
                candidate = text[i:i+l]
                if candidate in self.stoi:
                    match = candidate
                    break
            if match:
                tokens.append(match)
                i += len(match)
            else:
                tokens.append(text[i])
                i += 1
        return tokens

    def encode(self, text):
        tokens = self.tokenize(text)
        return [self.stoi[t] for t in tokens if t in self.stoi]

    def decode(self, token_ids):
        tokens = [self.itos[str(token_id)] for token_id in token_ids if str(token_id) in self.itos]
        text = "".join(tokens)
        text = re.sub(r"\s+([?.!,;:])", r"\1", text)
        text = re.sub(r"([?.!,;:])([^\s])", r"\1 \2", text)
        return text

    def save_mappings(self, stoi_path="stoi.json", itos_path="itos.json"):
        with open(stoi_path, "w", encoding="utf-8") as f:
            json.dump(self.stoi, f, ensure_ascii=False, indent=4)
        with open(itos_path, "w", encoding="utf-8") as f:
            json.dump(self.itos, f, ensure_ascii=False, indent=4)

    def load_mappings(self, stoi_path="stoi.json", itos_path="itos.json"):
        with open(stoi_path, "r", encoding="utf-8") as f:
            self.stoi = json.load(f)
        with open(itos_path, "r", encoding="utf-8") as f:
            self.itos = json.load(f)

# === Example usage ===
if __name__ == "__main__":
    all_files = [
        r"EVE\Data_storage\Synth_data.txt",
        r"moreShit.txt",
        r"DataSet\openwebtext.txt"
    ]

    collector = WordPairCollector()
    collector.process_files(all_files)

#_____________________________ End of WordPairCollector _____________________________#

    special_tokens = [' ', '<user>', '<eve>', '<start>', '<end>', '<name>', '<exc>']
    tokenizer = WordPairTokenizer(special_tokens=special_tokens)

    # Load word pairs
    tokenizer.load_word_pairs()

    # Build vocab from corpus
    tokenizer.build_vocab_from_corpus(all_files)

    # Build stoi/itos mappings
    tokenizer.build_mappings()
    tokenizer.save_mappings()

    # Interactive tokenization test
    while True:
        text = input("Enter text to tokenize (or 'exit' to quit): ")
        if text.lower() == "exit":
            break

        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        print(f"Tokens: {tokenizer.tokenize(text)}")
        print(f"Encoded: {encoded}")
        print(f"Decoded: {decoded}")
