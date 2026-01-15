import os
import discord
from dotenv import load_dotenv
from discord.ext import commands
from ModelGeneration import chat
import string as s 

# https://discord.com/oauth2/authorize?client_id=911986419281592411 
# the eve chat in the server: 1280243673761845310
# personal server: 1353393960441352254
# ^ allows eve to be added to your server

updates = "Currently updates: Pre-Beta \n\nFuture Updates: \n1. Allow Eve to veiw personal data, allow Eve more conversational options and more advanced features. \n2. Data collection will be enabled in full beta, By talking to Eve you are authorizing the collection of data\n\nAny questions?\n Contact: @kacarret for more information."
version = "Eve: Version Pre-Beta"
on_start = chat(name=None, minput="How do you feel?", mem_enabled=False, max_tokens=20)

load_dotenv()
description = '''Hello my name is Eve! I was created by Kacarret and I can't wait to meet you!'''
TOKEN = os.getenv('DISCORD_TOKEN')
GUILD = os.getenv('DISCORD_GUILD')
intents = discord.Intents.default()
intents.members = True
intents.message_content = True
bot = commands.Bot(command_prefix='!', description=description, intents=intents)

channel = 1370816733933338638 #private eve chat
#channel = 1370816496770486332 #public eve chat
Owner_ID = 518982204458795018
Is_bot_self = 911986419281592411
message_content = None

@bot.event
async def on_ready():
    try:
        vc = await ctx.channel.connect()
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
        
bot.run(TOKEN)