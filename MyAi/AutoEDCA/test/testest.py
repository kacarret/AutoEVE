
import json
import os
import regex as re
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


# === Example usage ===
if __name__ == "__main__":
    all_files = [
        "EVE/Data_storage/Synth_data.txt",
        "moreShit.txt",
        "DataSet/openwebtext.txt"
    ]

    collector = WordPairCollector()
    collector.process_files(all_files)

    print("Pair collection complete.")
    print(f"Top pairs: {collector.pair_freqs.most_common(10)}")
