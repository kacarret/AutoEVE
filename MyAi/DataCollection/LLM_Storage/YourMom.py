import os
import discord
from dotenv import load_dotenv
from discord.ext import commands
#from Cmd_chat import chat
from Beta_Cmd_chat import chat
import string as s 

updates = "Currently updates: 2.3 Alpha This update will be the backbone for personality modifiers and user focused responses (assuming eve's test score is above 30%) post training will be tested peirodically as smaller updates until the 3.0 alpha, if needed there will be additional 2.X updates. \n\nFuture Updates: \n1. Allow Eve to veiw personal data, allow Eve more conversational options and more advanced features. \n2. Data collection will be enabled in full beta, Authorized users for collecting data:\n\nZarker14\nDany\nZrackz\nKacarret\nAnimeWatchList78\n\nTo become an authorized user contact @kacarret for more information."
#version = "Eve: Version 2.3 Alpha"
version = "Eve: TEST BETA BRANCH"
legal_warn = "Welcome to Eve, brought to you by Kacarret!\n\nBy continuing to interact with Eve, you acknowledge and consent to the collection of data, including your responses to EVE-generated prompts. This data is stored securely to help improve your experience and ensure accurate, up-to-date responses.\n\n**Here’s what you should know:**\n\n**Data Collection:** We collect your responses to enhance Eve's performance.\n\n**Data Storage:** Your data is stored securely in a private, offline space.\n\n**Your Rights:** You can request access to and deletion of your data at any time. Data cannot be corrected, as it may affect Eve’s responses.\n\n**GDPR Compliance:** Your privacy is important, and we comply with GDPR to protect your data.\n\nIf you have any questions or wish to opt out, please contact @kacarret."
on_start = chat("How do you feel?")

def punctuation_check(text):
    return text.endswith(tuple(s.punctuation))

load_dotenv()
description = '''Quite frankly Mr.Afton those arnt the types of modifications we wanted to talk about...'''
TOKEN = os.getenv('DISCORD_TOKEN')
GUILD = os.getenv('DISCORD_GUILD')
intents = discord.Intents.default()
intents.members = True
intents.message_content = True
bot = commands.Bot(command_prefix='!', description=description, intents=intents)

channel = 1280243673761845310
Owner_Channel = 925970564047306793
Owner_ID = 518982204458795018
Is_bot_self = 911986419281592411
message_content = None

chat_type = input("Select a chat type: 1 for owner only, 2 for server members only\nWhat is your chat type: ")

def Owner_only(chat_type):
    if chat_type== "1":
        @bot.event
        async def on_ready():
            print("Startup is Complete!")
            channels = await bot.fetch_user(Owner_ID)
            await bot.change_presence(activity=discord.Game(on_start))
            await channels.send(f"{version} Ready for usage!")
            await channels.send(legal_warn)

        @bot.event
        async def on_message(message):
            message_content = message.content
            message_author = message.author
            channels = await bot.fetch_user(Owner_ID)
            #this we want the bot to only see if the message is coming from the owners chat not any message the owner sends
            if message.channel.id == Owner_Channel:
                print(f'New message -> {message_author} said: {message_content}')
                if message_author.id == Is_bot_self:
                    #print("self") #debugging
                    pass
                else:
                    user_input = message_content
                    if message_author.id == Owner_ID:
                        if user_input.lower() == 'exit':
                            exit()
                        elif user_input.lower() == 'help':
                            await channels.send("Help Program: \nType 'exit' to end the conversation. \nType 'server mode' to enter developer mode. \nType 'updates' to see what is new and future updates.")
                        elif user_input.lower() == 'updates':
                            await channels.send(updates)
                        elif user_input.lower() == 'server mode':
                            await channels.send("Entering server mode.")
                            chat_type = "2"
                            server_members_only(chat_type)
                        else:
                            if not punctuation_check(user_input):
                                response_text = chat(user_input+". ") # to ensure that eve dosnt try to continue of the promt she will assume its a sentence, this may lead to issues later when eve cant assume context on the sentence. 
                            else:
                                response_text = chat(user_input)
                            await channels.send(response_text)
                    else:
                        if not punctuation_check(user_input):
                            response_text = chat(user_input+". ") # to ensure that eve dosnt try to continue of the promt she will assume its a sentence, this may lead to issues later when eve cant assume context on the sentence. 
                        else:
                            response_text = chat(user_input)
                        await channels.send(response_text)
            else:
                #print("Message not sent from channel")
                pass
    elif chat_type == "2":
        server_members_only(chat_type)
    else:
        print("Invalid chat type")

def server_members_only(chat_type):
    if chat_type == "2":
        @bot.event
        async def on_ready():
            print("Startup is Complete!")
            channels = await bot.fetch_channel(channel)
            await bot.change_presence(activity=discord.Game(on_start))
            await channels.send(f"{version} Ready for usage!")
            await channels.send(legal_warn)

        @bot.event
        async def on_message(message):
            message_content = message.content
            message_author = message.author
            channels = await bot.fetch_channel(channel)
            if message.channel.id == channel:
                print(f'New message -> {message_author} said: {message_content}')
                if message_author.id == Is_bot_self:
                    #print("Self") #debugging
                    pass
                else:
                    user_input = message_content
                    if message_author.id == Owner_ID:
                        if user_input.lower() == 'exit':
                            exit()
                        elif user_input.lower() == 'help':
                            await channels.send("Help Program: \nType 'exit' to end the conversation. \nType 'server mode' to enter developer mode. \nType 'updates' to see what is new and future updates.")
                        elif user_input.lower() == 'updates':
                            await channels.send(updates)
                        elif user_input.lower() == 'dev mode':
                            await channels.send("Entering developer mode.")
                            chat_type = "1"
                            Owner_only(chat_type)
                        else:
                            if not punctuation_check(user_input):
                                response_text = chat(user_input+". ") # to ensure that eve dosnt try to continue of the promt she will assume its a sentence, this may lead to issues later when eve cant assume context on the sentence. 
                            else:
                                response_text = chat(user_input)
                            await channels.send(response_text)
                    else:
                        if not punctuation_check(user_input):
                            response_text = chat(user_input+". ") # to ensure that eve dosnt try to continue of the promt she will assume its a sentence, this may lead to issues later when eve cant assume context on the sentence. 
                        else:
                            response_text = chat(user_input)
                        await channels.send(response_text)
            else:
                #print("Message not sent from channel")
                pass
    elif chat_type == "1":
        Owner_only(chat_type)
    else:
        print("Invalid chat type")

if chat_type == "1":
    Owner_only(chat_type)
elif chat_type == "2":
    server_members_only(chat_type)
else:
    print("Invalid input. Please enter 1 or 2.")

bot.run(TOKEN)