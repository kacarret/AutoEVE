import json
import os
import subprocess
import re
import tkinter as tk
from tkinter import *

class Gui():
    def __init__(self, root):
        font = ("Arial", 10)

        root.title("Eve Trainer")
        root.geometry("700x650")
        root.resizable(False, False)

        Label(root, text="Batch Size", font=(font)).grid()
        Label(root, text="Block Size", font=(font)).grid()
        Label(root, text="Max Iters", font=(font)).grid()
        Label(root, text="Eval Interval", font=(font)).grid()
        Label(root, text="Learning Rate", font=(font)).grid()
        Label(root, text="Eval Iters", font=(font)).grid()
        Label(root, text="Number of Embeddings", font=(font)).grid()
        Label(root, text="Number of Heads", font=(font)).grid()
        Label(root, text="Number of Layers", font=(font)).grid()
        Label(root, text="Dropout", font=(font)).grid()
        Label(root, text="Weight Decay", font=(font)).grid()
        Label(root, text="Number of Splits", font=(font)).grid()
        Label(root, text="Fine Tune: (0 or 1)", font=(font)).grid()
        Label(root, font=(font)).grid()

        self.batch = Entry(root)
        self.batch.grid(row=0, column=1)
        self.block = Entry(root)
        self.block.grid(row=1, column=1)
        self.iters = Entry(root)
        self.iters.grid(row=2, column=1)
        self.interval = Entry(root)
        self.interval.grid(row=3, column=1)
        self.lr = Entry(root)
        self.lr.grid(row=4, column=1)
        self.eval = Entry(root)
        self.eval.grid(row=5, column=1)
        self.emb = Entry(root)
        self.emb.grid(row=6, column=1)
        self.head = Entry(root)
        self.head.grid(row=7, column=1)
        self.layer = Entry(root)
        self.layer.grid(row=8, column=1)
        self.dropout = Entry(root)
        self.dropout.grid(row=9, column=1)
        self.decay = Entry(root)
        self.decay.grid(row=10, column=1)
        self.splits = Entry(root)
        self.splits.grid(row=11, column=1)
        self.fine = Entry(root)
        self.fine.grid(row=12, column=1)
        self.file = Entry(root)
        self.file.grid(row=1, column=3)

        #make 5 sets of buttons that change the hyperparameters to sets of values
        past = tk.Button(root, text="Json",width=20, command=self.json_button)
        past.grid(row=0, column=3)
        small = tk.Button(root, text="Small",width=20, command=self.small_button)
        small.grid(row=2, column=3)
        medium = tk.Button(root, text="Medium",width=20, command=self.medium_button)
        medium.grid(row=3, column=3)
        large = tk.Button(root, text="Large",width=20, command=self.large_button)
        large.grid(row=4, column=3)
        clear = tk.Button(root, text="Clear",width=20, command=self.clear_button)
        clear.grid(row=5, column=3)

        #start and generate from EveTransformer
        start = tk.Button(root, text="Start",width=20, command=self.start_button)
        start.grid(row=13, column=1)

        #show the terminal output as a sceen
        self.output_window = tk.Text(root, height=20, width=50)
        self.output_window.grid(row=14, column=1)

    def json_button(self):
        json_path = self.file.get()
        if not json_path:
            self.terminal_output("No file selected")
            return

        if json_path:
            self.terminal_output(f"Loading {json_path}...")
            if not os.path.exists(json_path):
                self.terminal_output(f"File {json_path} does not exist")
                return
            if os.path.isfile(json_path):
                if not json_path.endswith(".json"):
                    self.terminal_output(f"File {json_path} is not a json file")
                    return
                elif json_path.endswith(".json"):
                    with open(json_path, "r") as f:
                        data = json.load(f)
                        self.batch.insert(0, data["batch_size"])
                        self.block.insert(0, data["block_size"])
                        self.iters.insert(0, data["max_iters"])
                        self.interval.insert(0, data["eval_interval"])
                        self.lr.insert(0, data["learning_rate"])
                        self.eval.insert(0, data["eval_iters"])
                        self.emb.insert(0, data["n_embd"])
                        self.head.insert(0, data["n_head"])
                        self.layer.insert(0, data["n_layer"])
                        self.dropout.insert(0, data["dropout"])
                        self.decay.insert(0, data["weight_decay"])
                        self.splits.insert(0, data["n_splits"])
                        self.fine.insert(0, data["fine_tune"])
                        self.terminal_output(f"Loaded {json_path} successfully")
            else:
                self.terminal_output(f"{json_path} is not a file")

        
    def small_button(self):
        self.terminal_output("Loading Small LLM Parameters...")
        self.batch.insert(0, 32)
        self.block.insert(0, 128)
        self.iters.insert(0, 5000)
        self.interval.insert(0, 1000)
        self.lr.insert(0, 0.001)
        self.eval.insert(0, 200)
        self.emb.insert(0, 384)
        self.head.insert(0, 6)
        self.layer.insert(0, 6)
        self.dropout.insert(0, 0.2)
        self.decay.insert(0, 0.005)
        self.splits.insert(0, 2)

    def medium_button(self):
        self.terminal_output("Loading Medium LLM Parameters...")
        self.batch.insert(0, 64)
        self.block.insert(0, 256)
        self.iters.insert(0, 10000)
        self.interval.insert(0, 1000)
        self.lr.insert(0, 0.001)
        self.eval.insert(0, 200)
        self.emb.insert(0, 512)
        self.head.insert(0, 8)
        self.layer.insert(0, 8)
        self.dropout.insert(0, 0.2)
        self.decay.insert(0, 0.005)
        self.splits.insert(0, 3)

    def large_button(self):
        self.terminal_output("Loading Large LLM Parameters...")
        self.batch.insert(0, 128)
        self.block.insert(0, 512)
        self.iters.insert(0, 10000)
        self.interval.insert(0, 1000)
        self.lr.insert(0, 0.001)
        self.eval.insert(0, 200)
        self.emb.insert(0, 768)
        self.head.insert(0, 8)
        self.layer.insert(0, 8)
        self.dropout.insert(0, 0.2)
        self.decay.insert(0, 0.005)
        self.splits.insert(0, 5)

    def clear_button(self):
        self.output_window.delete(1.0, tk.END)
        for widget in root.winfo_children():
            if isinstance(widget, tk.Entry):
                widget.delete(0, tk.END)

    def validate_num(self, num):
        return re.match(r'^-?\d+(?:\.\d+)?$', num)

    def save_params(self, params, filename):
        with open(filename, 'w') as f:
            json.dump(params, f, indent=4)

    def terminal_output(self, to_output):
        self.output_window.insert(tk.END, to_output+"\n")
        self.output_window.see(tk.END)

    def start_button(self):
        #check all inputs if they are a number or if they are a decimal
        if self.validate_num(self.batch.get()) is None:
            self.terminal_output("Batch Size must be a number")
            return
        if self.validate_num(self.block.get()) is None:
            self.terminal_output("Block Size must be a number")
            return
        if self.validate_num(self.iters.get()) is None:
            self.terminal_output("Max Iters must be a number")
            return
        if self.validate_num(self.interval.get()) is None:
            self.terminal_output("Eval Interval must be a number")
            return
        if self.validate_num(self.lr.get()) is None:
            self.terminal_output("Learning Rate must be a number")
            return
        if self.validate_num(self.eval.get()) is None:
            self.terminal_output("Eval Iters must be a number")
            return
        if self.validate_num(self.emb.get()) is None:
            self.terminal_output("Number of Embeddings must be a number")
            return
        if self.validate_num(self.head.get()) is None:
            self.terminal_output("Number of Heads must be a number")
            return
        if self.validate_num(self.layer.get()) is None:
            self.terminal_output("Number of Layers must be a number")
            return
        if self.validate_num(self.dropout.get()) is None:
            self.terminal_output("Dropout must be a number")
            return
        if self.validate_num(self.decay.get()) is None:
            self.terminal_output("Weight Decay must be a number")
            return
        if self.validate_num(self.splits.get()) is None:
            self.terminal_output("Number of Splits must be a number")
            return
        if self.validate_num(self.fine.get()) is None:
            self.terminal_output("Fine Tune must be a value")
            return

        self.terminal_output("All inputs are valid... starting training...\nThis window will close...")

        #start training
        batch_size = int(self.batch.get())
        block_size = int(self.block.get())
        max_iters = int(self.iters.get())
        eval_interval = int(self.interval.get())
        learning_rate = float(self.lr.get())
        eval_iters = int(self.eval.get())
        n_embd = int(self.emb.get())
        n_head = int(self.head.get())
        n_layer = int(self.layer.get())
        dropout = float(self.dropout.get())
        weight_decay = float(self.decay.get())
        n_splits = int(self.splits.get())
        fine_tune = bool(self.fine.get())

        params = {
            'batch_size': batch_size,
            'block_size': block_size,
            'max_iters': max_iters,
            'eval_interval': eval_interval,
            'learning_rate': learning_rate,
            'eval_iters': eval_iters,
            'n_embd': n_embd,
            'n_head': n_head,
            'n_layer': n_layer,
            'dropout': dropout,
            'weight_decay': weight_decay,
            'n_splits': n_splits,
            'fine_tune': fine_tune
        }

        self.save_params(params, 'AutoEve\setup.json')
        #run and kill the program
        root.destroy()
        self.run_eve()
        
    def run_eve(self):
        subprocess.run(['python', 'AutoEve\model.py'])

if __name__ == "__main__":
    root = tk.Tk()
    Gui(root)
    root.mainloop()