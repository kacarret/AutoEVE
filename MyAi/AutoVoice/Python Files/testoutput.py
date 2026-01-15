import torch
import json
import torchaudio
import soundfile as sf

#load the tensor from the output.txt file
with open('AutoVoice\Logging\output.json', 'r') as f:
    data = json.load(f)
    spectrogram = torch.tensor(data, dtype=torch.float32).to("cpu")

if spectrogram.shape[0] != 80:
    spectrogram = spectrogram.transpose(1, 2)

# Remove batch dimension
spectrogram = spectrogram.squeeze(0)

n_fft = 158
# Use Griffin-Lim to estimate phase and reconstruct waveform
griffin_lim = torchaudio.transforms.GriffinLim(n_fft=n_fft, hop_length=126)
waveform = griffin_lim(spectrogram)

# Save the waveform to a WAV file
sf.write("AutoVoice\Logging\output.wav", waveform.numpy(), samplerate=16000)

print("Audio saved as 'output.wav'")
