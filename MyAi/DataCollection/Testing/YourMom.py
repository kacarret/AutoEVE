import os
import discord
from dotenv import load_dotenv
from discord.ext import commands, tasks
import string as s 
import logging
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

class voiceClient(discord.VoiceClient):
    async def on_voice_receive(self, data):
        audio_data = data.data
        self.prossess_audio(audio_data)

    def prossess_audio(self, audio_data):
        # Process the audio data here
        logging.info("audio_data")
        with sr.AudioFile(audio_data) as source:
            try:
                audio = recognizer.record(source)
                text = recognizer.recognize_google(audio)
                logging.info(f"You said: {text}")
            except sr.UnknownValueError:
                logging.info("Could not understand audio")
            except sr.RequestError as e:
                logging.error(f"Could not request results; {e}")

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user}!')
    check_voice_channel.start()

@tasks.loop(seconds=5)  # Check every 5 seconds
async def check_voice_channel():
    for guild in bot.guilds:
        owner = guild.get_member(owner_id)
        
        if owner and owner.voice:
            if not guild.voice_client:
                try:
                    voice_client = await owner.voice.channel.connect(cls=voiceClient)
                    await owner.voice.channel.connect()
                except Exception as e:
                    logging.error(f"Error connecting to voice channel: {e}")
            elif guild.voice_client.channel != owner.voice.channel:
                try:
                    voice_client = await owner.voice.channel.connect(cls=voiceClient)
                    await guild.voice_client.move_to(owner.voice.channel)
                    logging.info(f"Moved to voice channel: {owner.voice.channel.name}")
                except Exception as e:
                    logging.error(f"Error moving to voice channel: {e}")
        else:
            for member in guild.members:
                if member.id in friends and member.voice:
                    if not guild.voice_client:
                        try:
                            voice_client = await member.voice.channel.connect(cls=voiceClient)
                            await member.voice.channel.connect()
                        except Exception as e:
                            logging.error(f"Error connecting to friend's voice channel: {e}")

@bot.event
async def on_voice_state_update(member, before, after):
    owner = member.guild.get_member(owner_id)
    
    if (member.id == owner_id or member.id in friends) and after.channel is not None:
        try:
            voice_client = await owner.voice.channel.connect(cls=voiceClient)
            if voice_client:
                await voice_client.process_audio(voice_client.recv())
        except Exception as e:
            logging.error(f"Error connecting to voice channel: {e} Disconnected from voice channel.")
            if member.guild.voice_client:  # Check if the bot is connected
                await member.guild.voice_client.disconnect()
    
    elif member.id == owner_id and before.channel is not None and after.channel is None:
        if member.guild.voice_client:  # Check if the bot is connected
            await member.guild.voice_client.disconnect()
            logging.info(f"Owner left, disconnected from voice channel.")

    # Check if all friends have left the voice channel and owner has left the voice channel
    elif (member.id in friends and before.channel is not None and after.channel is None):
        remaining_friends = [m for m in member.guild.members if m.id in friends and m.voice]
        if not remaining_friends and not owner.voice:
            if member.guild.voice_client:  # Check if the bot is connected
                await member.guild.voice_client.disconnect()
                logging.info(f"All friends left, disconnected from voice channel.")

bot.run(TOKEN)