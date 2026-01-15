import os
import json
from datetime import datetime, timedelta
from collections import defaultdict, deque

from tqdm import tqdm
import discord
from discord.ext import commands
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")

intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.members = True

bot = commands.Bot(command_prefix='!', intents=intents)

Owner_ID = 518982204458795018
Is_bot_self = 1403451385433034844

conversations = []
conversation_gap = timedelta(minutes=30)
active_conversations = []
recent_messages = deque(maxlen=1000)

CHANNEL_FILTER = {
    911723835655606272,
    1399572075856789524
}

SAVE_JSON_PATH = "AutoEDCA\\Data\\real_conversations.json"

@bot.event
async def on_ready():
    await bot.change_presence(activity=discord.Game("Studying conversations..."))
    print(f"✅ Logged in as {bot.user}")

    message_lookup = {}

    for guild in bot.guilds:
        for channel in tqdm(guild.text_channels, desc=f"Processing guild {guild.name}...", leave=False):
            if CHANNEL_FILTER and channel.id not in CHANNEL_FILTER:
                #print(f"Skipping channel {channel}")
                continue  # Skip channels not in filter

            try:
                messages = []
                async for message in channel.history(limit=None, oldest_first=True):
                    if not message.author.bot:
                        messages.append(message)
                        message_lookup[message.id] = message  # Store for reply reference

                # Group messages into conversations
                current_convo = []
                last_time = None

                for msg in tqdm(messages, desc=f"Processing channel {channel.name}", leave=False):
                    in_reply_chain = False

                    if msg.reference and msg.reference.message_id in message_lookup:
                        # If message is a reply to another known message, force inclusion
                        ref_msg = message_lookup[msg.reference.message_id]
                        for convo in conversations:
                            if ref_msg in convo:
                                convo.append(msg)
                                in_reply_chain = True
                                break

                    if in_reply_chain:
                        continue  # Already grouped, skip to next

                    if last_time and (msg.created_at - last_time) > conversation_gap:
                        if current_convo:
                            conversations.append(current_convo)
                        current_convo = []

                    current_convo.append(msg)
                    last_time = msg.created_at

                if current_convo:
                    conversations.append(current_convo)

            except Exception as e:
                print(f"[ERROR] {e} in channel {channel.name}")

    print(f"Found {len(conversations)} conversations.")

    # Save conversations to JSON
    json_data = []
    for convo in conversations:
        convo_data = []
        for msg in convo:
            timestamp = msg.created_at.strftime('%Y-%m-%d %H:%M:%S')
            convo_data.append(f"{msg.author.name}: {msg.content}")
        json_data.append(convo_data)

    with open(SAVE_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"💾 Conversations saved to {SAVE_JSON_PATH}")

    # Optional: Print out each conversation to console
    for idx, convo in tqdm(enumerate(conversations), desc="Printing conversations", leave=False):
        print(f"\nConversation {idx + 1} ({len(convo)} messages):")
        for msg in convo:
            timestamp = msg.created_at.strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{timestamp}] {msg.author.name}: {msg.content}")

# listens for messages in a channel
@bot.event
async def on_message(message: discord.Message):
    # Ignore bots
    if message.author.bot:
        return

    # Optional: filter by channel name
    if CHANNEL_FILTER and message.channel.name not in CHANNEL_FILTER:
        return

    now = message.created_at
    added_to_existing = False

    # Store this message in recent_messages for reply lookup
    recent_messages.append(message)

    # Handle reply-based conversation continuation
    if message.reference and message.reference.message_id:
        for convo in active_conversations:
            for msg in convo:
                if msg.id == message.reference.message_id:
                    convo.append(message)
                    added_to_existing = True
                    break
            if added_to_existing:
                break

    # If not a reply match, check by time gap
    if not added_to_existing:
        if active_conversations:
            last_convo = active_conversations[-1]
            if last_convo:
                last_msg_time = last_convo[-1].created_at
                if (now - last_msg_time) <= conversation_gap:
                    last_convo.append(message)
                    added_to_existing = True

    # If still not added, start a new conversation
    if not added_to_existing:
        active_conversations.append([message])

    # Optional: print it live
    timestamp = now.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message.author.name} in #{message.channel.name}: {message.content}")

    await bot.process_commands(message)

bot.run(TOKEN)