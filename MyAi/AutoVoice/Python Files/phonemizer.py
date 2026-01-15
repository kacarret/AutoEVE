import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm
import os
import numpy as np
import json
import string
import logging
import matplotlib.pyplot as plt
import soundfile as sf

files_to_check = 'AutoVoice/Logging/phonemizer.log', 'AutoVoice/Data/All_Saves/metadata.csv'
for file in files_to_check:
    if os.path.exists(file):
        os.remove(file)
logging.basicConfig(filename=files_to_check[0], level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

# define the mel-spectro and inverse of the transformations
mel_transform = T.MelSpectrogram(sample_rate=22050, n_fft=1024,hop_length=126, n_mels=80)
inv_mel_transform = T.InverseMelScale(n_stft=513, n_mels=80, sample_rate=22050)
griffin_lim = T.GriffinLim(n_fft=1024, hop_length=126)

class Parser:
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
        texts = [''.join(c for c in w if c in string.ascii_lowercase) for w in words]
        char_vocab = sorted(set("".join(texts)))
        char_to_idx = {c: i+1 for i, c in enumerate(char_vocab)}
        idx_to_char = {i: c for c, i in char_to_idx.items()}
        logging.info(f"Character Vocabulary: {char_to_idx}")
        all_phonemes = sorted(set(p for seq in phonemes for p in seq))
        phoneme_to_idx = {p: i+1 for i, p in enumerate(all_phonemes)}
        if "SPN" not in phoneme_to_idx:
            phoneme_to_idx["SPN"] = 0
        idx_to_phoneme = {i: p for p, i in phoneme_to_idx.items()}
        logging.info(f"Phoneme Vocabulary: {phoneme_to_idx}")
        return char_to_idx, phoneme_to_idx

    def tokenize_text(text, char_to_idx):
        text = text.translate(str.maketrans('', '', string.punctuation)).lower()
        return [char_to_idx.get(c, 0) for c in text]

    def tokenize_phonemes(phoneme_seq, phoneme_to_idx):
        return [phoneme_to_idx.get(p, 0) for p in phoneme_seq]

class TextToSpectrogramDataset(Dataset):
    def __init__(self, data_list, cmudict, char_to_idx, phoneme_to_idx, sampling_rate=22050):
        """
        data_list: list of (text, audio_path) tuples
        cmudict: dict word → [PHONES]
        *_to_idx: vocab dicts
        """
        self.data_list = data_list
        self.cmudict = cmudict
        self.phoneme_to_idx = phoneme_to_idx
        self.sampling_rate = sampling_rate
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sampling_rate,
            n_fft=1024,
            hop_length=126,
            n_mels=80
        )

    def __len__(self):
        return len(self.data_list)

    def phonemize(self, text):
        tokens = []
        for word in text.lower().split():
            phones = self.cmudict.get(word)
            if phones:
                tokens.extend(phones)
            else:
                # fallback for unknown word (you can also try letter fallback)
                tokens.extend(['SPN'])  # 'SPN' = spoken noise / unknown
        return [self.phoneme_to_idx.get(p, 0) for p in tokens]

    def __getitem__(self, idx):
        audio_path,text = self.data_list[idx]
        if not os.path.exists(audio_path):
            audio_path = 'AutoVoice/Recorder/Recordings/' + audio_path # may need to change the text and audio path things
        phoneme_ids = self.phonemize(text)

        # Load audio and convert to mel spectrogram
        waveform, sr = torchaudio.load(audio_path)
        if sr != self.sampling_rate:
            resample = T.Resample(sr, self.sampling_rate)
            waveform = resample(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        mel = self.mel_transform(waveform).squeeze(0).transpose(0, 1)  # (Time, Mel)

        return {
            "audio_path": audio_path,
            "phoneme_ids": torch.tensor(phoneme_ids, dtype=torch.long),
            "mel": mel
        }

class Utils:
    # Helper function to reconstruct the mel from the audio file and then evaluate laterdef reconstruct_and_compare(audio_path, original_mel):
    def save_to_wav(linear_spec, save_path):
        # for this to work:
        # 1. linear_spec must be [1, 80, T]
        # 2. linear_spec must be a waveform or (the byproduct of inverting a mel spectrogram)
        # 3. the ai must be able to produce these things for the write path to work
        reconstructed_waveform = griffin_lim(linear_spec)

        try:
            if reconstructed_waveform.dim() == 2 and reconstructed_waveform.shape[0] == 1:
                reconstructed_waveform = reconstructed_waveform.squeeze(0)

            # Convert to NumPy
            reconstructed_waveform = reconstructed_waveform.numpy()

            # Ensure correct data type
            reconstructed_waveform = reconstructed_waveform.astype(np.float32)

            # Normalize if necessary
            if np.max(np.abs(reconstructed_waveform)) > 1.0:
                reconstructed_waveform = reconstructed_waveform / np.max(np.abs(reconstructed_waveform))

            # Now save
            sf.write(save_path, reconstructed_waveform, 22050)
        except:
            pass

    def save_to_wav(device, spectrogram, save_path):
        # for this to work: this doesnt work fix it
        # 1. linear_spec must be [1, 80, T]
        # 2. linear_spec must be a waveform or (the byproduct of inverting a mel spectrogram)
        # 3. the ai must be able to produce these things for the write path to work
        if not isinstance(spectrogram, torch.Tensor):
            raise ValueError("spectrogram must be a torch tensor")

        # convert spectrogram to 1 ,80, T ( normally it is 1, T, 80)
        spectrogram = spectrogram.transpose(1, 2)

        spectrogram = torch.abs(spectrogram)

        #print("spectrogram.shape:", spectrogram.shape)

        global inv_mel_transform, griffin_lim
        inv_mel_transform=inv_mel_transform.to(device)
        griffin_lim=griffin_lim.to(device)

        linear_spec = inv_mel_transform(spectrogram)

        # check to ensure that the program is actually working and also make sure that the tensor is passed properly

        print("[save_to_wav] Current spectrogram shape:", linear_spec.shape)
        min_T = 1024
        current_T = linear_spec.shape[-1]
        if current_T < min_T:
            pad_amt = min_T - current_T
            linear_spec = F.pad(linear_spec, (0, pad_amt), mode='constant', value=0.0)
            print(f"[save_to_wav] Padded spectrogram from T={current_T} to T={min_T}")
            print("[save_to_wav] New spectrogram shape:", linear_spec.shape)

        #print("linear_spec.shape:", linear_spec.shape)

        linear_spec = linear_spec.squeeze(0)

        #linear_spec = torch.abs(linear_spec)

        reconstructed_waveform = griffin_lim(linear_spec)

        try:
            # Remove channel dim if needed
            if reconstructed_waveform.dim() == 2 and reconstructed_waveform.shape[0] == 1:
                reconstructed_waveform = reconstructed_waveform.squeeze(0)

            # Convert to NumPy
            waveform_np = reconstructed_waveform.cpu().numpy().astype(np.float32)

            # Normalize to -1 to 1
            max_val = np.max(np.abs(waveform_np))
            if max_val > 1.0:
                waveform_np = waveform_np / max_val

            # Save as WAV
            sf.write(save_path, waveform_np, 22050)
            print(f"[save_to_wav] Saved WAV to: {save_path}")

        except Exception as e:
            print(f"[save_to_wav] Error saving WAV: {e}")


    def reconstruct_and_compare(audio_path, original_mel, log_dir=None):
        waveform, sr = torchaudio.load(audio_path)
        if sr != 22050:
            resampler = T.Resample(orig_freq=sr, new_freq=22050)
            waveform = resampler(waveform)

        # Convert original waveform to Mel
        mel = mel_transform(waveform) # shape: [1, 80, T]'

        # Invert mel to waveform
        linear_spec = inv_mel_transform(mel)
        #self.save_to_wav(linear_spec, "AutoVoice/Logging/reconstructed.wav")
        reconstructed_waveform = griffin_lim(linear_spec)

        # Recreate Mel from reconstructed waveform
        recon_mel = mel_transform(reconstructed_waveform).transpose(1, 2)  # shape: [1, 80, T']

        # Make sure both spectrograms have same shape
        T_min = min(original_mel.shape[0], recon_mel.shape[0])
        mel = original_mel[:T_min]
        recon_mel = recon_mel[:T_min]

        if mel.shape != recon_mel.shape:
            raise ValueError(f"Shape mismatch: mel={mel.shape}, recon={recon_mel.shape}")

        mse = F.mse_loss(recon_mel, mel)

        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            Utils.plot_energy(
                audio_path,
                mel,
                recon_mel,
                title_prefix="Reconstruction",
                show=False,
                save_path=os.path.join(log_dir, os.path.basename(audio_path) + "_reconstruction.png")
            )

        return mse.item()


    def collate_fn(batch):
        phoneme_ids = [item['phoneme_ids'] for item in batch]
        mels = [item['mel'] for item in batch]
        return {
            'phoneme_ids': pad_sequence(phoneme_ids, batch_first=True, padding_value=0),
            'mel': pad_sequence(mels, batch_first=True, padding_value=0)
        }

    def compute_energy_from_mel(mel: torch.Tensor):
        """
        Computes per-frame energy from mel spectrogram of shape [1, 80, T]
        Returns: energy [T]
        """
        if mel.dim() != 3 or mel.shape[2] != 80:
            raise ValueError(f"Expected mel shape [1, 80, T], got {mel.shape}")

        energy = mel.pow(2).mean(dim=1).squeeze(0)  # → [T]
        return energy

    def plot_energy(audio_path, mel: torch.Tensor, recon_mel: torch.Tensor, sample_rate=22050, hop_length=126, title_prefix="", show=True, save_path=None):
        mel_energy = Utils.compute_energy_from_mel(mel)
        recon_energy = Utils.compute_energy_from_mel(recon_mel)

        T_len = mel_energy.shape[0]
        time_axis = torch.arange(T_len) * hop_length / sample_rate  # Convert frames to seconds

        plt.figure(figsize=(12, 4))
        plt.plot(time_axis, mel_energy.numpy(), label="Original Mel Energy")
        plt.plot(time_axis[:len(recon_energy)], recon_energy.numpy(), label="Reconstructed Mel Energy", linestyle='--')
        plt.title(f"{title_prefix} - Energy Over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Energy")
        plt.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        else:
            plt.close()


    def save_vocab(char_to_idx, phoneme_to_idx, cmudict, out_dir="AutoVoice/Data/All_Saves/"):
        with open(os.path.join(out_dir, "char_to_idx.json"), "w") as f:
            json.dump(char_to_idx, f)

        with open(os.path.join(out_dir, "phoneme_to_idx.json"), "w") as f:
            json.dump(phoneme_to_idx, f)

        with open(os.path.join(out_dir, "cmudict.json"), "w") as f:
            json.dump(cmudict, f)

    def load_json(path):
        with open(path, "r") as f:
            return json.load(f)

    def save_metadata(data_list, path="AutoVoice/Data/All_Saves/metadata.csv"):
        with open(path, "w") as f:
            for text, wav_path in data_list:
                f.write(f"{text}|{wav_path}\n")

    def load_metadata(path="metadata.csv"):
        data_list = []
        with open(path, "r") as f:
            for line in f:
                text, wav_path = line.strip().split("|")
                data_list.append((text, wav_path))
        return data_list

    def evaluate_dataset(dataset, log_dir="AutoVoice/Logging/energy_plots_phonemizer"):
        os.makedirs(log_dir, exist_ok=True)
        summary = []

        for sample in tqdm(dataset, desc="Evaluating phonemizer"):
            mel = sample['mel'].unsqueeze(0)
            audio_path = sample['audio_path']
            mse = Utils.reconstruct_and_compare(audio_path, mel, log_dir=log_dir)
            summary.append((audio_path, mse))

        summary.sort(key=lambda x: x[1], reverse=True)  # Sort by highest MSE

        with open(os.path.join(log_dir, "mse_summary.csv"), "w") as f:
            f.write("audio_path,mse\n")
            for audio_path, mse in summary:
                f.write(f"{audio_path},{mse:.6f}\n")

        print(f"Evaluation complete. Results saved to {log_dir}/mse_summary.csv")


if __name__ == "__main__":

    data_list = []
    with open("AutoVoice\Recorder\Logging\metadata.csv", "r") as f:
        for line in f:
            wav_path, text = line.strip().split("|")
            data_list.append((text, os.path.join("AutoVoice\\Recorder\\Recordings\\", wav_path)))

    # get the total time for all the recordings
    total_time = 0
    for text, wav_path in data_list:
        waveform, sr = torchaudio.load(wav_path)
        if sr != 22050:
            resample = T.Resample(orig_freq=sr, new_freq=22050)
            waveform = resample(waveform)
        total_time += len(waveform)

    total_time_h = int(total_time / 3600)
    total_time_m = int((total_time % 3600) / 60)
    total_time_s = int(total_time % 60)

    logging.info(f"Total time found: {total_time_h}h {total_time_m}m {total_time_s}s")
    
    # remove all data inside of the log directory
    if os.path.exists("AutoVoice/Logging/energy_plots_phonemizer"):
        for file in os.listdir("AutoVoice/Logging/energy_plots_phonemizer"):
            logging.info(f"Removing {os.path.join('AutoVoice/Logging/energy_plots_phonemizer', file)}")
            os.remove(os.path.join("AutoVoice/Logging/energy_plots_phonemizer", file))

    cmudict_path = r'AutoVoice\Data\cmudict-0.7b'
    word_to_phones = Parser.parse_cmudict(file_path=cmudict_path)

    char_to_idx, phoneme_to_idx = Parser.index_vocab(word_to_phones.keys(), word_to_phones.values())

    dataset = TextToSpectrogramDataset(
        data_list,
        cmudict=word_to_phones,
        char_to_idx=char_to_idx,
        phoneme_to_idx=phoneme_to_idx
    )
    '''
    sample = dataset[0]
    logging.info(f"Phoneme IDs: {sample['phoneme_ids']}")
    logging.info(f"Mel shape: {sample['mel'].shape}")  # e.g., [T, 80]
    '''
    #loader = DataLoader(dataset, batch_size=5, collate_fn=Utils.collate_fn)

    Utils.evaluate_dataset(dataset=dataset, log_dir="AutoVoice/Logging/energy_plots_phonemizer")
    Utils.save_metadata(data_list)
    Utils.save_vocab(word_to_phones, char_to_idx, phoneme_to_idx)

"""
how to use the loader:

char_to_idx = load_json("char_to_idx.json")
phoneme_to_idx = load_json("phoneme_to_idx.json")
cmudict = load_json("cmudict.json")

# load training list
data_list = []
with open("metadata.csv", "r") as f:
    for line in f:
        text, path = line.strip().split("|")
        data_list.append((text, path))

or


cmudict = load_json("cmudict.json")
char_to_idx = load_json("char_to_idx.json")
phoneme_to_idx = load_json("phoneme_to_idx.json")
data_list = load_metadata("metadata.csv")

dataset = TextToSpectrogramDataset(data_list, cmudict, char_to_idx, phoneme_to_idx)
"""