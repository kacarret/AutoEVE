import re
#read a file
with open('EVE\other convos\dialogues_train.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Original dialogue (you can replace this with your input text)
original_dialogue = text

def clean_punctuation(text):
    # Replace single quotes with apostrophes
    text = text.replace("’", "'")

    # Remove spaces before punctuation (e.g., " , " becomes ",")
    text = re.sub(r'\s([.,!?()%"])', r'\1', text)

    text = re.sub(r"\s(')&-\s", r'\1', text)
    
    # Ensure there is no space after punctuation (e.g., "hello ." becomes "hello.")
    text = re.sub(r'([.,!?()])\s', r'\1 ', text)

    # Ensure there is no space after characters (e.g., "$ 100" becomes "$100")
    text = re.sub(r'\$ (\d+)', r'$\1', text)
    
    return text

# Function to format each conversation
def reformat_dialogue(dialogue):
    # Clean the punctuation first
    cleaned_dialogue = clean_punctuation(dialogue)
    
    # Split by the end of utterance token (__eou__) and by newlines
    conversations = cleaned_dialogue.strip().split('\n')
    
    # Initialize a list to hold all formatted conversation blocks
    formatted_conversations = []

    for conversation in conversations:
        # Split each conversation by '__eou__' to get the individual turns
        turns = conversation.split(' __eou__ ')
        
        # Strip whitespace and clean up the turns
        turns = [turn.strip() for turn in turns if turn.strip()]
        
        # Initialize formatted conversation with the start tag
        formatted_conversation = "[start]\n"
        
        # Assign speakers in alternating order: [user] and [eve]
        for i, turn in enumerate(turns):
            speaker = "[user]" if i % 2 == 0 else "[eve]"
            formatted_conversation += f"{speaker} {turn}\n"
        
        # Add the end tag at the end of the conversation
        formatted_conversation += "[end]"

        # Add the formatted conversation to the list
        formatted_conversations.append(formatted_conversation)

        # remove any remaining __eou__ tokens
        formatted_conversations = [conversation.replace('__eou__', '') for conversation in formatted_conversations]
    
    return "\n\n".join(formatted_conversations)

# Call the function to reformat the dialogue
formatted_dialogue = reformat_dialogue(original_dialogue)

# Print the result
print(formatted_dialogue)

with open('MoreDialogues.txt', 'a', encoding='utf-8') as f:
        f.write(formatted_dialogue)
        f.close()