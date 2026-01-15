import os
import discord
from dotenv import load_dotenv
from discord.ext import commands, tasks
import string as s 
import logging
import wave
import numpy as np
import speech_recognition as sr
recognizer = sr.Recognizer()


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
GUILD = os.getenv('DISCORD_GUILD')
intents = discord.Intents.default()
intents.voice_states = True  # Necessary for voice state updates
intents.members = True  # Necessary for member updates
intents.message_content = True

friends = [342067333084479489, 319613239891722240, 713821158880837667, 879744824591396965]  # List of friend IDs
owner_id = 518982204458795018  # ID of the owner
bot = commands.Bot(command_prefix="!", intents=intents)
recording_states = {}

class AudioSink(discord.VoiceClient):
    def __init__(self):
        super().__init__()
        self.frames = []

    def write(self, data):
        self.frames.append(data)

    def save(self, filename):
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(2)  # Stereo
            wf.setsampwidth(2)  # 16 bits
            wf.setframerate(48000)  # Sample rate
            wf.writeframes(b''.join(self.frames))

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user}')

@bot.command()
async def join(ctx):
    if ctx.author.voice:
        channel = ctx.author.voice.channel
        await channel.connect()
        await ctx.send(f'Joined {channel}')
    else:
        await ctx.send("You're not in a voice channel.")

@bot.command()
async def leave(ctx):
    if ctx.voice_client:
        await ctx.voice_client.disconnect()
        await ctx.send("Disconnected.")
    else:
        await ctx.send("I'm not in a voice channel.")

recording_states = {}

@bot.command()
async def record(ctx):
    if ctx.voice_client is None:
        await ctx.send("I need to be in a voice channel to record.")
        return

    if ctx.guild.id not in recording_states:
        recording_states[ctx.guild.id] = False

    if recording_states[ctx.guild.id]:
        await ctx.send("Already recording.")
        return

    sink = AudioSink()
    ctx.voice_client.start_recording(sink, lambda: stop_callback(sink, ctx.guild.id))
    recording_states[ctx.guild.id] = True

    await ctx.send("Recording started...")

@bot.command()
async def stop(ctx):
    if ctx.voice_client is None or ctx.guild.id not in recording_states or not recording_states[ctx.guild.id]:
        await ctx.send("I'm not recording.")
        return

    ctx.voice_client.stop_recording()
    recording_states[ctx.guild.id] = False
    await ctx.send("Recording stopped.")

def stop_callback(sink, guild_id):
    sink.save('recorded_audio.wav')

bot.run(TOKEN)