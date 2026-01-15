import regex as re
import os
import json
from collections import Counter, defaultdict

class BPETokenizer:
    def __init__(self, special_tokens=None, merge_threshold=0):
        #input("Warning: Ensure that the re.compile is correct with the list of special tokens. Hit enter to continue...")
        self.special_tokens = special_tokens or []
        self.vocab = {}
        self.bpe_merges = {}
        self.stoi = {}
        self.itos = {}
        self.merge_threshold = merge_threshold

    def preprocess(self, text):
        # Process the text by ensuring special tokens are included and words are split properly
        tokens = []

        #tokens.append(self.special_tokens)
        # create a pattern to match all tokens
        pattern = re.compile(r"""<\p{L}+>|(?:'s|'t|'re|'ve|'m|'ll|'d)|\p{L}+|\p{N}+|[^\s\p{L}\p{N}]+""", re.UNICODE)
        #for token in re.split(r"(\s+)", text):
        for token in re.findall(pattern, text):
            if token.strip() in self.special_tokens:
                tokens.append(token.strip())
            elif token.strip():
                chars = list(token.strip())
                tokens.append(" ".join(chars + ["</w>"]))
        return tokens

    def build_vocab(self, corpus):
        # Build the vocabulary with word frequencies
        vocab = Counter(corpus)
        self.vocab = dict(vocab)

    def get_pair_freq(self):
        # Initialize the pair_freq dictionary with default integer values
        pair_freq = defaultdict(int)

        # Iterate through the vocabulary and count bigrams
        for word, freq in self.vocab.items():
            symbols = word.split()  # Split the word into symbols (sub-words)
            
            for i in range(len(symbols) - 1):
                # Increment the frequency of the bigram
                pair_freq[(symbols[i], symbols[i + 1])] += freq

        pairs_above_threshold = {pair: freq for pair, freq in pair_freq.items() if freq >= self.merge_threshold}
        # Return the sorted pair frequencies
        return pairs_above_threshold


    def merge_vocab(self, pairs):
    # Merge only the most frequent pair in the vocabulary
        for pair, _ in pairs.items():
            bigram = " ".join(pair)
            replacement = "".join(pair)

            merged_vocab = {}

            # Create a new token and update the vocabulary
            for word, freq in self.vocab.items():
                merged_word = word.replace(bigram, replacement)
                merged_vocab[merged_word] = freq

            # Update the vocabulary with the newly merged words
            self.vocab = merged_vocab

    def train(self, text):
        self.clear_files()
        corpus = self.preprocess(text.lower())
        self.build_vocab(corpus)

        merge_count = 0
        trained = []
        
        while True:
            pair_freq = self.get_pair_freq()
            if not pair_freq:
                # No more pairs to merge, exit loop
                print("No more pairs to merge. Stopping training.")
                break

            # Get the most frequent pair
            most_frequent_pair = max(pair_freq, key=pair_freq.get)

            # Merge the most frequent pair
            self.merge_vocab({most_frequent_pair: pair_freq[most_frequent_pair]})
            
            # Save the merge to track the new token as an individual token
            new_token = "".join(most_frequent_pair)

            # check if the new token created is already in the stoi dict
            self.special_tokens.append(new_token)  # Add the new token to special tokens list
            
            # Update mappings
            #self.build_mappings()

            self.bpe_merges[merge_count] = {most_frequent_pair: pair_freq[most_frequent_pair]}
            trained.append({most_frequent_pair: pair_freq[most_frequent_pair]})
            merge_count += 1

            # Check if the merge threshold has been reached
            if ((pair_freq[most_frequent_pair]) <= self.merge_threshold):
                print("Pair limit reached. Stopping training.")
                break

            # Optional: Print merge info for debugging
            print(f"Merge {merge_count}: {most_frequent_pair} -> {pair_freq[most_frequent_pair]}")
        self.build_mappings()

    def build_mappings(self):
        # Start with special tokens
        all_tokens = set(self.special_tokens)

        # Include tokens from the final merged vocab
        for word in self.vocab:
            for token in word.split():
                all_tokens.add(token)

        # Clean whitespace just in case
        all_tokens = [token.replace(" ", "") for token in all_tokens]

        # Build mappings
        self.stoi = {token: i for i, token in enumerate(sorted(all_tokens), start=1)}
        self.itos = {i: token for token, i in self.stoi.items()}

        self.save_stoi()
        #self.save_merges()

    def t_prep(self, text):
        tokens = []
        pattern = re.compile(r"""<\p{L}+>|(?:'s|'t|'re|'ve|'m|'ll|'d)|\p{L}+|\p{N}+|[^\s\p{L}\p{N}]+""", re.UNICODE)

        for token in re.findall(pattern, text):
            token = token.strip()
            if not token:
                continue

            if token + "</w>" in self.stoi:
                #print("token case 1 found: ", token)
                tokens.append([token])
            elif token in self.stoi:
                #print("token case 2 found: ", token)
                tokens.append([token])
            else:
                # Turn word into list of characters with </w> at the end
                #print("token not found: ", token)
                char_tokens = list(token)
                char_tokens[-1] = char_tokens[-1]  # attach </w> to last char
                tokens.append(char_tokens)

        # Flatten the list: [['t', 'h', 'a', 't</w>'], ...] → ['t', 'h', 'a', 't</w>', ...]
        return tokens


    def tokenize(self, text):
        tokens = self.t_prep(text.lower())
        result = []
        words = tokens
        words = ["".join(word) for word in words]

        #print(f"Initial words: {words}")

        # Replay all BPE merges in order
        for word in words:
            if word + "</w>" in self.stoi:
                #print(f"case 1: {word}")
                result.append(word + "</w>")
            elif word in self.stoi:
                #print(f"case 2: {word}")
                result.append(word)
            else:
                print(f"case not found: {word}")
                word = list(word) + ["</w>"]
                for merge_step in self.bpe_merges.values():
                    (a, b), _ = list(merge_step.items())[0]
                    #print(f"Processing merge step: {a} {b}")
                    i = 0
                    new_word = []
                    while i < len(word):
                        if i < len(word) - 1 and word[i] == a and word[i+1] == b:
                            #print(f"Merging: {word[i]} {word[i+1]}")
                            new_word.append(a + b)
                            i += 2
                        else:
                            new_word.append(word[i])
                            i += 1
                    word = new_word
                    print(f"New word after merge: {word}")
                result.extend(word)

        #print(f"Final result: {result}")
        #token checker
        final_tokens = [t for t in result if t in self.stoi]
        #print(f"Final tokens: {final_tokens}")
        return final_tokens

                
    def encode(self, text):
        # Encode text into token IDs
        self.load_stoi()
        #tokens = self.tokenize(text.lower())
        
        # we may need to save the token pair frequencies and merge them to the inputed text then use the stoi dict and convert to token ids
        tokens = self.tokenize(text.lower())
        #print(f"Tokens: {tokens}")
        token_ids = [self.stoi[token] for token in tokens if token in self.stoi]
        #print(f"Token IDs: {token_ids}")
        return token_ids

    def decode(self, token_ids):
        # Decode token IDs back into text
        self.load_itos()
        tokens = [self.itos[token_id] for token_id in token_ids if token_id in self.itos]
        text = " ".join(tokens).replace("</w>", "").strip()
        text = text.replace(" .", ".").replace(" ?", "?").replace(" !", "!").replace(" ,", ",").replace(" '", "'").replace(" :", ":").replace(" ;", ";").replace(" / ", "/")
        return text

    def save_stoi(self, filename="AutoEDCA\\Data\\stoi.json"):
        # Save the string-to-index mapping
        with open(filename, "w") as f:
            json.dump(self.stoi, f, indent=4)

    def load_stoi(self, filename="AutoEDCA\\Data\\stoi.json"):
        # Load the string-to-index mapping
        if os.path.exists(filename):
            with open(filename, "r") as f:
                self.stoi = json.load(f)
                return self.stoi
        else:
            raise FileNotFoundError(f"File {filename} does not exist")

    def load_itos(self, filename="AutoEDCA\\Data\\stoi.json"):
        # Load the stoi dictionary
        self.load_stoi(filename)
        self.itos = {v: k for k, v in self.stoi.items()}
        return self.itos

    def save_merges(self, filepath="AutoEDCA\\Data\\bpe_merges.json"):
        with open(filepath, "w") as f:
            json.dump({k: list(v.items())[0] for k, v in self.bpe_merges.items()}, f)

    def load_merges(self, filepath="AutoEDCA\\Data\\bpe_merges.json"):
        with open(filepath, "r") as f:
            raw_data = json.load(f)
            self.bpe_merges = {
                k: {tuple(v[0]): v[1]} for k, v in raw_data.items()
            }
            #print(self.bpe_merges)

    def load_vocab_size(self, added_special_tokens):
        # Calculate the size of the vocabulary
        return len(self.stoi) + added_special_tokens + 1

    def clear_files(self):
        # Clear the saved files (use with caution)
        files = ["stoi.json"]
        for file in files:
            if os.path.exists(file):
                os.remove(file)

# # # Example usage
while __name__ == "__main__":
    tokenizer = BPETokenizer(special_tokens=["<start>", "<end>", "<user>", "<eve>", "<name>", "<exc>", "<time>"])
    text = ""
    all_files = ["AutoEDCA\\Data\\MoreDialogues.txt", 
                 "AutoEDCA\\Data\\Dialogues.txt"
                 ]
    for file in all_files:
        with open(file, 'r', encoding='utf-8') as f:
            text += f.read()

    tokenizer.train(text)

    tokenizer.save_stoi()
    tokenizer.save_merges()

    # Test tokenization after training
    is_exit = False
    while not is_exit:
        sample_text = input("Enter text to tokenize: ")
        #tokenized = tokenizer.tokenize(sample_text)
        #print(f"Tokenized '{sample_text}': {tokenized}")

    #Test encoding and decoding
        encoded = tokenizer.encode(sample_text)
        decoded = tokenizer.decode(encoded)
        print(f"Encoded: {encoded}\nDecoded: {decoded}")

    #Check if the user wants to exit
        if sample_text.lower() == "exit":
            is_exit = True
    # #with this we want a itterate again over the pairs to create more complex pairs still while removing anything that is "too small"