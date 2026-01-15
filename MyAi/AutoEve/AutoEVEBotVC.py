import os
import discord
from discord.ext import commands, tasks
from dotenv import load_dotenv
from ModelGeneration import chat as chats
import whisper
import asyncio
import discord.sinks
import shutil
from scipy.io import wavfile
import numpy as np

load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")

local_ffmpeg = os.path.join(os.path.dirname(__file__), "ffmpeg.exe")

if shutil.which("ffmpeg") is None:
    os.environ["PATH"] += os.pathsep + os.path.dirname(local_ffmpeg)

intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True

bot = commands.Bot(command_prefix='!', intents=intents)
model = whisper.load_model("small")

channel = None
Owner_ID = 518982204458795018
Is_bot_self = 911986419281592411
message_content = None

connections = {}

RECORD_SECONDS = 2
ENERGY_THRESHOLD = 100

command_list = ["!join", "!join_channel", "!leave", "!chat"]

def transcribe_audio(file_path):
    result = model.transcribe(file_path)
    return result["text"]

@bot.event
async def on_ready():
    await bot.change_presence(activity=discord.Game("AutoEVE is online!"))
    print(f"✅ Logged in as {bot.user}")

# join a voice channel that the user is in
@bot.command()
async def join(ctx):
    if not ctx.author.voice:
        return await ctx.send("❌ You're not in a voice channel.")

    try:
        vc = await ctx.author.voice.channel.connect()
        await ctx.send("🔊 Joined and listening...")
    except Exception as e:
        await ctx.send(f"❌ Error joining channel... Activating auto reconnect...")
        await ctx.voice_client.disconnect()
        vc = await ctx.author.voice.channel.connect()
        await ctx.send("🔊 Joined and listening...")

    async def monitor_once():
        sink = discord.sinks.WaveSink()

        async def on_finish(sink, channel):
            for user_id, audio in sink.audio_data.items():
                filename = f"{user_id}.wav"
                with open(filename, 'wb') as f:
                    f.write(audio.file.read())
                    audio.file.seek(0)

                # === ENERGY CHECK ===
                energy = calculate_energy(filename)
                print(f"[DEBUG] Energy for {user_id}: {energy}")
                if energy < ENERGY_THRESHOLD:
                    print("[INFO] Too quiet — skipping.")
                    os.remove(filename)
                    return

                # === PROCESS AUDIO ===
                async def handle_audio():
                    transcript = transcribe_audio(filename)
                    # keep adding the transcrpt to a list and once the transcrprt stops increasing after t seconds then send it
                    message += transcript
                    print(f"Transcript for <@{user_id}>:\n{message}")

                    # === TIMER ===
                    timer = asyncio.create_task(
                        asyncio.sleep(2),  # 2 seconds
                        name="reset_timer"
                    )
                    last_message = message

                    def check(msg):
                        nonlocal last_message
                        if msg != last_message:
                            print(f"[DEBUG] Updating last_message from {last_message} to {msg}")
                            last_message = msg
                            timer.cancel()
                            return True
                        print("[DEBUG] No change in message")
                        return False

                    while True:
                        await asyncio.sleep(0.1)
                        if check(message):
                            continue
                        if timer.done():
                            break

                    print(f"Final transcript for <@{user_id}>:\n{message}")
                    os.remove(filename)

                asyncio.create_task(handle_audio())

        vc.start_recording(sink, on_finish, ctx.channel)
        await asyncio.sleep(RECORD_SECONDS)
        if vc.recording:
            vc.stop_recording()

    # Start a repeating loop
    async def loop_monitor():
        while True:
            await monitor_once()
            await asyncio.sleep(0.5)  # small delay between loops

    bot.loop.create_task(loop_monitor())

# to tell eve where to joion and not just the user
@bot.command()
async def join_channel(ctx, channel_id: int):
    try:
        vc = await ctx.guild.get_channel(channel_id).connect()
        await ctx.send("🔊 Joined and listening...")
    except Exception as e:
        await ctx.send(f"❌ Error joining channel... Activating auto reconnect...")
        await ctx.voice_client.disconnect()
        vc = await ctx.guild.get_channel(channel_id).connect()
        await ctx.send("🔊 Joined and listening...")

    async def monitor_once():
        sink = discord.sinks.WaveSink()

        async def on_finish(sink, channel):
            for user_id, audio in sink.audio_data.items():
                filename = f"{user_id}.wav"
                with open(filename, 'wb') as f:
                    f.write(audio.file.read())
                    audio.file.seek(0)

                # === ENERGY CHECK ===
                energy = calculate_energy(filename)
                print(f"[DEBUG] Energy for {user_id}: {energy}")
                if energy < ENERGY_THRESHOLD:
                    print("[INFO] Too quiet — skipping.")
                    os.remove(filename)
                    return

                # === PROCESS AUDIO ===
                async def handle_audio():
                    transcript = transcribe_audio(filename)
                    await ctx.send(f"Transcript for <@{user_id}>:\n{transcript}") 
                    print(f"Transcript for <@{user_id}>:\n{transcript}")
                    os.remove(filename)

                asyncio.create_task(handle_audio())

        vc.start_recording(sink, on_finish, ctx.channel)
        await asyncio.sleep(RECORD_SECONDS)
        if vc.recording:
            vc.stop_recording()

    # Start a repeating loop
    async def loop_monitor():
        while True:
            await monitor_once()
            await asyncio.sleep(0.5)  # small delay between loops

    bot.loop.create_task(loop_monitor())

# calculate the energy of an audio file
def calculate_energy(file_path):
    """Calculates RMS energy from a WAV file safely."""
    if ((not os.path.exists(file_path)) or (os.path.getsize(file_path) == 0)):
        print(f"[WARN] File {file_path} is missing or empty.")
        return 0

    try:
        sample_rate, samples = wavfile.read(file_path)
        print(f"[DEBUG] Sample rate for {file_path}: {sample_rate}")
        print(f"[DEBUG] Samples for {file_path}: {samples}")
        if samples.ndim > 1:
            audio_data = samples[:, 0]
        else:
            audio_data = samples

        energy = np.sum(audio_data**2) / sample_rate
        
        print(f"[DEBUG] Energy for {file_path}: {energy}")
        return energy
    except Exception as e:
        print(f"[ERROR] Failed to process {file_path}: {e}")
        return 0

# leave the voice channel
@bot.command()
async def leave(ctx):
    if ctx.voice_client:
        await ctx.voice_client.disconnect()
        await ctx.send("👋 Left the voice channel.")
    else:
        await ctx.send("❌ I'm not connected to a voice channel.")

# moves the global channel to chat in the text channel
@bot.command()
async def chat(ctx):
    global channel
    channel = ctx.channel.id
    await ctx.send("💬 Chat channel set. I will now respond to messages here.")

# listens for messages in a channel
@bot.event
async def on_message(message):
    global channel
    global message_content

    if message.author.bot:
        return  # Ignore messages from bots

    message_content = message.content
    message_author = message.author

    if channel is not None and message.channel.id == channel:
        #chanenls = await bot.fetch_channel(channel)

        print(f'New message -> {message_author} said: {message_content}')

        if message_author.id != Is_bot_self:
            user_input = message_content

            if message_author.id == Owner_ID and user_input.lower() == 'help':
                await message.channel.send("🛠 Help: Type 'exit' to end the conversation.")

            if not any(cmd in message_content for cmd in command_list):
                try:
                    response_text = chats(str(message_author), user_input, mem_enabled=True, max_tokens=75)
                    if response_text:
                        await message.channel.send(response_text)
                except Exception as e:
                    await message.channel.send("⚠️ I encountered an error, please tell Kacarret.")
                    print(f"[ERROR] {e}")
            else:
                pass

    await bot.process_commands(message)


bot.run(TOKEN)