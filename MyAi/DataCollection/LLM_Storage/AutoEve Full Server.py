import os
import discord
from dotenv import load_dotenv
from discord.ext import commands, tasks
import string as s 
import logging
from Cmd_chat import chat

def punctuation_check(text):
    return text.endswith(tuple(s.punctuation))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
GUILD = os.getenv('DISCORD_GUILD')
intents = discord.Intents.default()
intents.guilds = True  # Necessary for guild updates
intents.members = True  # Necessary for member updates
intents.message_content = True

friends = [518982204458795018, 342067333084479489, 319613239891722240, 713821158880837667, 879744824591396965]  # List of friend IDs
bot = commands.Bot(command_prefix="!", intents=intents)


@bot.event
async def on_ready():
    logging.info(f'Logged in as {bot.user.name} (ID: {bot.user.id})')

@bot.event
async def on_message(message):
    # Ignore messages from the bot itself
    if message.author == bot.user:
        return

    # Check if the bot is mentioned
    if bot.user in message.mentions:
        # Send a welcome DM to the user
        try:
            await message.author.send("Hello! You mentioned me! How can I help you today?")
        except Exception as e:
            await message.channel.send(f'Failed to send DM: {e}')

    # If the user is chatting after mentioning the bot
    if message.author.dm_channel and message.author.dm_channel is not None:
        response = await handle_user_input(message.content)
        await message.author.send(response)

    # Process commands if you have any
    await bot.process_commands(message)

async def handle_user_input(user_input):
    # Simple response logic; you can expand this as needed
    if not punctuation_check(user_input):
        response_text = chat(user_input+". ") # to ensure that eve dosnt try to continue of the promt she will assume its a sentence, this may lead to issues later when eve cant assume context on the sentence. 
    else:
        response_text = chat(user_input)
        return(response_text)
bot.run(TOKEN)