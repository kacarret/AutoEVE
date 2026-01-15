import json
import os

def parse_cmudict(file_path):
    word_to_phones = {}
    with open(file_path, 'r', encoding='latin-1') as f:
        for line in f:
            line = line.strip()
            if line.startswith(';') or line == '':
                continue
            parts = line.split('  ')
            if len(parts) != 2:
                continue
            word, phones_str = parts
            word = word.split('(')[0].lower()
            phones = phones_str.split()
            word_to_phones[word] = phones
    return word_to_phones

def index_vocab(words, phonemes):
    texts = words
    char_vocab = sorted(set("".join(texts)))
    char_to_idx = {c: i+1 for i, c in enumerate(char_vocab)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    print("Character Vocabulary:", char_to_idx)
    all_phonemes = sorted(set(p for seq in phonemes for p in seq))
    phoneme_to_idx = {p: i+1 for i, p in enumerate(all_phonemes)}
    idx_to_phoneme = {i: p for p, i in phoneme_to_idx.items()}
    print("Phoneme Vocabulary:", phoneme_to_idx)
    return char_to_idx, phoneme_to_idx

def tokenize_text(text, char_to_idx):
    return [char_to_idx.get(c, 0) for c in text]

def tokenize_phonemes(phoneme_seq, phoneme_to_idx):
    return [phoneme_to_idx.get(p, 0) for p in phoneme_seq]

def save_vocab(char_to_idx, phoneme_to_idx, cmudict, out_dir):
    with open(os.path.join(out_dir, "char_to_idx.json"), "w") as f:
        json.dump(char_to_idx, f)

    with open(os.path.join(out_dir, "phoneme_to_idx.json"), "w") as f:
        json.dump(phoneme_to_idx, f)

    with open(os.path.join(out_dir, "cmudict.json"), "w") as f:
        json.dump(cmudict, f)


if __name__ == "__main__":
    cmudict_path = r'AutoVoice\Data\cmudict-0.7b'
    word_to_phones = parse_cmudict(cmudict_path)

    char_to_idx, phoneme_to_idx = index_vocab(word_to_phones.keys(), word_to_phones.values())

    
