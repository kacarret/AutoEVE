import os
import sys
import json
import logging
import threading
import socket
import tkinter as tk
from tkinter import messagebox

HOST = '192.168.1.175'  # Replace with the server's local IP address
PORT = 65432

#use pyinstaller to create an exe: python -m PyInstaller -noconsole temp.py

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

def save_config(username):
    config = {
        'name': username,
    }
    with open('personal_config.json', 'w') as f:
        json.dump(config, f)

def send_message(username_entry, message_entry):
    username = username_entry.get()
    message = message_entry.get()
    if not username or not message or username == 'Enter a username' or message == 'Enter your message here':
        messagebox.showwarning("Warning", "Please enter a username and message.")
        return
    else:
        save_config(username)
        message = username + ": " + message
        s.sendall(message.encode())
        message_entry.delete(0, tk.END)

def receive_message():
    while True:
        data = s.recv(1024)
        decoded = data.decode()
        if decoded == "<message_token>":
            pass
        else:
            # Schedule the update of the output window in the main thread
            root.after(0, update_output_window, decoded)

# Function to update the output window with new messages
def update_output_window(message):
    output_window.config(state='normal')
    output_window.insert(tk.END, f"{message}\n")
    output_window.config(state='disabled')
    output_window.see(tk.END)

def on_entry_click(event, tempentry):
    # this will be called when the entry widget is clicked
    if tempentry.get() != '':
        tempentry.delete(0, tk.END)
        tempentry.config(fg='black')

def on_focusout(event, tempentry, text):
    if tempentry.get() == '':
        tempentry.insert(0, text)
        tempentry.config(fg='grey')

try:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        root = tk.Tk()
        root.title("OpenChatter By: Kacarret")
        root.configure(bg="black")
        root.geometry("350x350")
        root.resizable(False, False)

        label_title = tk.Label(root, text="OpenChatter By: Kacarret", bg="black", fg="white", font=("Arial", 20))
        label_title.grid(row=0, column=0)

        username_entry = tk.Entry(root, fg="grey", width=50)
        username_entry.insert(0, 'Enter a username')
        username_entry.bind("<FocusIn>", lambda event: on_entry_click(event, username_entry))
        username_entry.bind("<FocusOut>", lambda event: on_focusout(event, username_entry, 'Enter a username'))
        if os.path.exists('personal_config.json'):
            with open('personal_config.json', 'r') as f:
                config = json.load(f)
                username_entry.delete(0, tk.END)
                username_entry.config(fg='black')
                username_entry.insert(0, config['name'])
        username_entry.grid(row=1, column=0)

        # add the receive_message function to the main loop use treading
        output_window = tk.Text(root, bg="black", fg="white", width=30, height=10, state="disabled")
        output_window.grid(row=2, column=0)
        output_window.see(tk.END)

        receive_thread = threading.Thread(target=receive_message)
        receive_thread.start()

        message_entry = tk.Entry(root, fg="grey", width=50)
        message_entry.insert(0, 'Enter your message here')
        message_entry.bind("<FocusIn>", lambda event: on_entry_click(event, message_entry))
        message_entry.bind("<FocusOut>", lambda event: on_focusout(event, message_entry, 'Enter your message here'))
        message_entry.grid(row=3, column=0)

        # add the username_entry to message_entry to add the username to the message

        button = tk.Button(root, text="Send", command=lambda: send_message(username_entry,message_entry))
        button.grid()

        root.mainloop()

except Exception as e:
    logging.error(f"An error occurred: {e}")