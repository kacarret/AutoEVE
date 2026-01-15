import os
import re
import threading
import time
import wave
import pyaudio
from typing import Dict, List

class TextToSpeech:
    CHUNK = 1024

    def __init__(self, words_pron_dict: str = 'text-to-speech/english-to-base.txt'):
        self._pronunciation_dict: Dict[str, List[str]] = {}
        self._load_words(words_pron_dict)

    def _load_words(self, words_pron_dict: str) -> None:
        with open(words_pron_dict, 'r') as file:
            for line in file:
                if not line.startswith(';;;'):
                    key, val = line.split('  ', 2)
                    self._pronunciation_dict[key] = self._extract_pronunciation_codes(val)

    def _extract_pronunciation_codes(self, val: str) -> List[str]:
        return re.findall(r"[A-Z]+", val)

    def get_pronunciation(self, str_input: str) -> None:
        pronunciations = self._get_pronunciations(str_input)
        self._play_audio(pronunciations)

    def _get_pronunciations(self, str_input: str) -> List[str]:
        pronunciations = []
        for word in re.findall(r"[\w']+", str_input.upper()):
            if word in self._pronunciation_dict:
                pronunciations.extend(self._pronunciation_dict[word])
        print(pronunciations)
        return pronunciations

    def get_audio_duration(file_path):
        with wave.open(file_path, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = frames / float(rate)
            return duration

    def _play_audio(self, pronunciations: List[str]) -> None:
        delay = 0
        for pronunciation in pronunciations:
            self._play_audio_file(pronunciation, delay)
            """
            file_path = f"text-to-speech/sounds/{pronunciation}.wav"
            if pronunciation != [0]:
                with wave.open(file_path, 'rb') as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    duration = frames / float(rate)
                    print(f"Delay: {delay}, Duration: {duration}")
                    delay += duration
            time.sleep(delay)  # wait for the delay before playing the next file
            """
            
    def _play_audio_file(self, pronunciation: str, delay: float) -> None:
        try:
            time.sleep(delay)
            pronunciation = pronunciation.lower()
            with wave.open(f"text-to-speech/sounds/{pronunciation}.wav", 'rb') as wf:
                p = pyaudio.PyAudio()
                stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                                channels=wf.getnchannels(),
                                rate=wf.getframerate(),
                                output=True)
                data = wf.readframes(self.CHUNK)
                while data:
                    stream.write(data)
                    data = wf.readframes(self.CHUNK)
                stream.stop_stream()
                stream.close()
                p.terminate()
        except Exception as e:
            print(f"Error playing audio file: {e}")
            pass
    
if __name__ == '__main__':
    tts = TextToSpeech()
    tts.get_pronunciation('I am eve')