import os
import queue
import time
import csv
import uuid
import shutil
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import whisper

# === Config ===
SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_DURATION = 0.03  # seconds
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION)
ENERGY_THRESHOLD = 0.01  # tune this!
SILENCE_FRAMES = 15  # how many silent frames to wait before stopping
OUTPUT_DIR = "AutoVoice\Recorder\Recordings"
METADATA_FILE = "AutoVoice\Recorder\Logging\metadata.csv"

# remove all data inside of the log directory
direct = "AutoVoice/Recorder/Recordings"
if os.path.exists(direct):
    for file in os.listdir(direct):
        print(f"Removing {os.path.join(direct, file)}")
        os.remove(os.path.join(direct, file))

if os.path.exists(METADATA_FILE):
    print(f"Removing {METADATA_FILE}")
    os.remove(METADATA_FILE)

local_ffmpeg = os.path.join(os.path.dirname(__file__), "ffmpeg.exe")

if shutil.which("ffmpeg") is None:
    os.environ["PATH"] += os.pathsep + os.path.dirname(local_ffmpeg)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Initialize whisper model
model = whisper.load_model("small")

def transcribe_audio(file_path):
    result = model.transcribe(file_path)
    return result["text"]

def main():
    print("Starting recording... Press Ctrl+C to stop.")

    q = queue.Queue()
    recording = []
    silence_counter = 0
    i = 0
    recording_active = True

    def audio_callback(indata, frames, time_info, status):
        q.put(indata.copy())


    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,blocksize=FRAME_SIZE, callback=audio_callback):
        try:
            while True:
                frame = q.get()
                energy = np.sqrt(np.mean(frame**2))

                if energy > ENERGY_THRESHOLD:
                    if not recording_active:
                        print("Speech detected. Recording started...")
                        recording_active = True
                    recording.append(frame)
                    silence_counter = 0
                elif recording_active:
                    recording.append(frame)
                    silence_counter += 1
                    if silence_counter >= SILENCE_FRAMES:
                        print("Silence detected. Saving recording...")
                        audio_np = np.concatenate(recording, axis=0)
                        duration = len(audio_np) / SAMPLE_RATE

                        if duration < 0.75:
                            print(f"Recording too short ({duration:.2f} seconds). Skipping...")
                        else:
                            temp_filename = f"audio_{i}.wav"
                            temp_filepath = os.path.join(OUTPUT_DIR, temp_filename)
                            write(temp_filepath, SAMPLE_RATE, audio_np)
                            print(f"Saved recording to {temp_filepath}")

                            #add the transcription
                            with open(METADATA_FILE, "a", newline="", encoding="utf-8") as f:
                                writer = csv.writer(f, delimiter="|")
                                transcription = transcribe_audio(temp_filepath)
                                writer.writerow([temp_filename, transcription])
                                print(f"Transcription was: {transcription}")

                            i += 1

                        recording = []
                        silence_counter = 0
                        recording_active = False

        except KeyboardInterrupt:
            print("\nRecording stopped by user")

if __name__ == "__main__":
    main()
