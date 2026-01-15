import tkinter as tk
from tkinter import messagebox
import os

START_TOKEN = '<start>'
END_TOKEN = '<end>'
INPUT_FILE = "MoreDialogues.txt"
OUTPUT_FILE = "MoreDialoguesReformatted.txt"

def extract_chunks(file_content):
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


def update_chunk(original, new_text, start_index, end_index):
    return original[:start_index] + START_TOKEN + '\n' + new_text + '\n' + END_TOKEN + original[end_index:]


def find_next_unedited_chunk(chunks):
    for i, (_, _, chunk_text) in enumerate(chunks):
        if '[[edited]]' not in chunk_text:
            return i
    return None


def mark_chunk_as_edited(chunk_text):
    if '[[edited]]' not in chunk_text:
        return f'[[edited]]\n{chunk_text}'
    return chunk_text


def edit_chunks_gui():
    # Step 1: Determine whether to load from output or input
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
    elif os.path.exists(INPUT_FILE):
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
    else:
        print("No input or output file found.")
        return

    chunks = extract_chunks(content)
    chunk_index = find_next_unedited_chunk(chunks)
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

    def save_and_next():
        nonlocal chunk_index
        edited_text = text_box.get(1.0, tk.END).strip() 
        start, end, _ = chunks[chunk_index]
        edited_text = mark_chunk_as_edited(edited_text)
        content_holder[0] = update_chunk(content_holder[0], edited_text, start, end)

        # Save to disk immediately
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
            out_f.write(content_holder[0])

        # Re-extract chunks from the saved content and move to next
        chunks[:] = extract_chunks(content_holder[0])
        chunk_index = find_next_unedited_chunk(chunks)
        show_chunk()

    def previous_chunk():
        nonlocal chunk_index
        chunk_index -= 1
        show_chunk()

    # GUI setup
    root = tk.Tk()
    root.title("Persistent Chunk Editor")

    label = tk.Label(root, text="")
    label.pack()

    text_box = tk.Text(root, width=80, height=20)
    text_box.pack()

    button_frame = tk.Frame(root)
    button_frame.pack()

    previous_button = tk.Button(button_frame, text="Previous", command=previous_chunk)
    previous_button.pack(side=tk.LEFT, padx=10)

    save_button = tk.Button(button_frame, text="Save and Next", command=save_and_next)
    save_button.pack(side=tk.LEFT, padx=0)

    exit_button = tk.Button(button_frame, text="Exit", command=root.quit)
    exit_button.pack(side=tk.RIGHT, padx=10)

    show_chunk()
    root.mainloop()


if __name__ == '__main__':
    edit_chunks_gui()
