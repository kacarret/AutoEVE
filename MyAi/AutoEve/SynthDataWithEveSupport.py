import os
import tkinter as tk
from tkinter import messagebox
import regex as re
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import trange

from model import Transformer 
from model import DataLoader
from model import Hyperparameters

# initalize the full dataset as this requires eve's full model
DataLoader.load('Full')

# Load the generation tool from the model import
model = Transformer(use_conversational_head=False).to(Hyperparameters.device)
model.load_state_dict(torch.load('Redone Eve Model\\model.pth'), strict=False)
model.eval()

START_TOKEN = '<start>'
END_TOKEN = '<end>'
INPUT_FILE = "MoreDialogues.txt"
OUTPUT_FILE = "MoreDialoguesReformatted.txt"

# make a method to read data from the file and also works an editor
class FileEditor:
    def __init__(self, stoken, etoken, infile, outfile):
        self.stoken = stoken
        self.etoken = etoken
        self.infile = infile
        self.outfile = outfile

    def extract_chucks(self, file_content):
        chunks = []
        pos = 0
        while True:
            start_index = file_content.find(START_TOKEN, pos)
            if start_index == -1:
                break
            end_index = file_content.find(END_TOKEN, start_index)
            if end_index == -1:
                break

            chunk_text = file_content[start_index + len(START_TOKEN):end_index].strip()
            chunks.append((start_index, end_index + len(END_TOKEN), chunk_text))
            pos = end_index + len(END_TOKEN)
        return chunks

    def update_chunk(self, original, new_text, start_index, end_index):
        return original[:start_index] + START_TOKEN + '\n' + new_text + '\n' + END_TOKEN + original[end_index:]
    
    def find_next_unedited_chunk(self, chunks):
        for i, (_, _, chunk_text) in enumerate(chunks):
            if '[[edited]]' not in chunk_text:
                return i
        return None
    
    # allow for eve to work with us
    def eve_model(self, name=None, minput=None, mem_enabled=False, max_tokens=Hyperparameters.block_size):
        if name != None:
            print(f"A name was provided: '{name}' This should not be passed!")
            print(f"Setting {name} to: 'None'")
            name = None

        if mem_enabled == True:
            print(f"Memory was incorrectly passed: '{mem_enabled}' This should not be passed!")
            print(f"Setting mem_enabled to: 'False'")
            mem_enabled = False

        if minput == None:
            print(f"ERROR: No input was provided!")
            return

        minput = minput+"\n<eve> "
        minput = DataLoader.tokenizer.encode(minput)
        context = (torch.tensor(minput, dtype=torch.long).unsqueeze(0).to(Hyperparameters.device))
        response = DataLoader.tokenizer.decode(model.generate(idx=context, max_new_tokens=max_tokens, mem_enabled=mem_enabled)[0].tolist())
        return response

    def mark_chunk_as_edited(self, chunk_text):
        if '[[edited]]' not in chunk_text:
            return f'[[edited]]\n{chunk_text}'
        return chunk_text
    
    def edit_chunk_gui(self):
        if os.path.exists(OUTPUT_FILE):
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                content = f.read()
        elif os.path.exists(INPUT_FILE):
            with open(INPUT_FILE, 'r', encoding='utf-8') as f:
                content = f.read()
        else:
            print("No input or output file found.")
            return

        chunks = self.extract_chucks(content)
        chunk_index = self.find_next_unedited_chunk(chunks)
        if chunk_index is None:
            print("All chunks have been edited.")
            return

        content_holder = [content]  # Mutable holder for the content string

        def show_chunk():
            nonlocal chunk_index
            if chunk_index is None or chunk_index >= len(chunks):
                messagebox.showinfo("Done", "All chunks processed.")
                root.quit()
                return

            start, end, chunk_text = chunks[chunk_index]
            clean_chunk = chunk_text.replace('[[edited]]', '').strip()
            text_box.delete(1.0, tk.END)
            text_box.insert(tk.END, clean_chunk)
            label.config(text=f"Editing chunk {chunk_index + 1} of {len(chunks)}")

        def previous_chunk():
            nonlocal chunk_index
            chunk_index -= 1
            show_chunk()

        def save_and_next():
            nonlocal chunk_index
            edited_text = text_box.get(1.0, tk.END).strip() 
            start, end, _ = chunks[chunk_index]
            edited_text = self.mark_chunk_as_edited(edited_text)
            content_holder[0] = self.update_chunk(content_holder[0], edited_text, start, end)

            # Save to disk immediately
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
                out_f.write(content_holder[0])

            # Re-extract chunks from the saved content and move to next
            chunks[:] = self.extract_chucks(content_holder[0])
            chunk_index = self.find_next_unedited_chunk(chunks)
            show_chunk()

        def auto_generate():
            # print all chunk data
            edited_text = text_box.get(1.0, tk.END).strip() 
            
            #create a user warning popup
            response = messagebox.askyesno("Auto Generate", "Please ensure that the <eve> token is present with no text next to it. (Will consider all text above it...)\n\nAre you sure you want to auto generate?")
            
            if response:
                # Split the text into lines for easier processing
                lines = edited_text.splitlines()
                
                try:
                    # Find the index of the line that contains only '<eve>'
                    eve_index = next(i for i, line in enumerate(lines) if line.strip() == "<eve>")
                except StopIteration:
                    messagebox.showwarning("Missing <eve> token", "No valid <eve> token found on a line by itself.")
                    return
                
                # Extract the text before the <eve> token
                before_eve = '\n'.join(lines[:eve_index])

                # prep the response
                response = '<eve> '
                
                # pass everything for eve to process
                response += self.eve_model(minput=before_eve)
                
                # add the response to the lines
                lines[eve_index] = response
                
                # Join the lines back into a single string
                edited_text = '\n'.join(lines)
                
                # Update the text box with the generated text
                text_box.delete(1.0, tk.END)
                text_box.insert(tk.END, edited_text)

            if not response:
                messagebox.showinfo("Canceled", "Auto generate canceled.")
                return

        root = tk.Tk()
        root.title("Eve Model File Editor")

        label = tk.Label(root, text=f"Editing chunk {chunk_index + 1} of {len(chunks)}")
        label.pack()

        text_box = tk.Text(root, width=80, height=20)
        text_box.pack()

        button_frame = tk.Frame(root)
        button_frame.pack()

        previous_button = tk.Button(button_frame, text="Previous", command=previous_chunk)
        previous_button.pack(side=tk.LEFT, padx=5)

        save_button = tk.Button(button_frame, text="Save and Next", command=save_and_next)
        save_button.pack(side=tk.LEFT, padx=5)

        auto_generate_button = tk.Button(button_frame, text="Auto Generate", command=auto_generate)
        auto_generate_button.pack(side=tk.LEFT, padx=5)

        exit_button = tk.Button(button_frame, text="Exit", command=root.quit)
        exit_button.pack(side=tk.LEFT, padx=5)

        show_chunk()
        root.mainloop()

if __name__ == '__main__':
    editor = FileEditor(START_TOKEN, END_TOKEN, INPUT_FILE, OUTPUT_FILE)
    editor.edit_chunk_gui()