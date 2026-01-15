import pyttsx3
import speech_recognition as sr
import os
from ModelGeneration import chat

for index, name in enumerate(sr.Microphone.list_microphone_names()):
    print(f"{index}: {name}")

# Initialize the engine
engine = pyttsx3.init()
r = sr.Recognizer()
mic = sr.Microphone(device_index=1)

voices = engine.getProperty('voices')

# Set properties (optional)
engine.setProperty('voice', voices[1].id)
engine.setProperty('rate', 150)  # Speed of speech
engine.setProperty('volume', 0.75)  # Volume (0.0 to 1.0)

def speak(text):
    engine.say(text)
    engine.runAndWait()

while True:
    try:
        with mic as source:
            print("Listening...")
            audio = r.listen(source)
            r.energy_threshold = 150
            r.pause_threshold = 1
            print("Recognizing...")
            user_input = r.recognize_google(audio)
            print(f"You: {user_input}"+". ")
    except sr.UnknownValueError:
        print("Could not understand audio")
        continue
    if user_input.lower() == 'exit':
        exit()
    response_text = chat(None, minput=user_input, mem_enabled=True, max_tokens=100)
    print(f"Eve: {response_text}")
    speak(response_text)