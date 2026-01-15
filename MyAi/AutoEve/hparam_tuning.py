import json
import random
import os
import subprocess
import time
import re
import logging

if os.path.exists('AutoEve\hparam_tuning.log'):
    os.remove('AutoEve\hparam_tuning.log')

logging.basicConfig(filename='AutoEve\hparam_tuning.log', level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

class SmartHparamTuner:
    def __init__(self, search_space, top_k=5, explore_prob=0.2):
        self.search_space = search_space
        self.keys = list(search_space.keys())
        self.history = []  # (params_dict, val_loss)
        self.best_loss = float('inf')
        self.best_params = None
        self.top_k = top_k
        self.explore_prob = explore_prob

    def load_base_params(self):
        with open('AutoEve\setup.json', 'r') as f:
            return json.load(f)

    def sample_random_params(self):
        return {k: random.choice(self.search_space[k]) for k in self.keys}

    def mutate_params(self, base_params):
        new_params = base_params.copy()
        mutable_keys = [k for k in self.keys if k != "block_size"]

        # Mutate 1 to 3 parameters
        keys_to_mutate = random.sample(mutable_keys, k=random.randint(1, min(3, len(mutable_keys))))

        for key in keys_to_mutate:
            values = self.search_space[key]
            current = base_params[key]

            # Integer-valued mutation (e.g., n_layer, n_embd, n_head)
            if isinstance(values[0], int):
                idx = values.index(current)
                shift = random.choice([-1, 1])
                new_idx = max(0, min(len(values) - 1, idx + shift))
                new_params[key] = values[new_idx]

            # Float-valued mutation (e.g., dropout, weight_decay, learning_rate)
            elif isinstance(values[0], float):
                factor = random.choice([0.5, 0.8, 1.25, 2.0])
                mutated = current * factor
                closest = min(values, key=lambda x: abs(x - mutated))
                new_params[key] = closest

            # Fallback for categorical values (e.g., batch_size)
            else:
                new_params[key] = random.choice([v for v in values if v != current])

        # --- Constraint Check: n_embd must be divisible by n_head ---
        if new_params["n_embd"] % new_params["n_head"] != 0:
            # Adjust n_head to a compatible value (favor smaller)
            compatible_heads = [h for h in self.search_space["n_head"] if new_params["n_embd"] % h == 0]
            if compatible_heads:
                new_params["n_head"] = random.choice(compatible_heads)
            else:
                # Fallback: adjust n_embd instead
                compatible_embds = [e for e in self.search_space["n_embd"] if e % new_params["n_head"] == 0]
                if compatible_embds:
                    new_params["n_embd"] = random.choice(compatible_embds)

        return new_params


    def suggest_params(self):
        if len(self.history) == 0 or random.random() < self.explore_prob:
            return self.sample_random_params()
        
        # Exploit: mutate one of the top_k
        top_configs = sorted(self.history, key=lambda x: x[1])[:self.top_k]
        base = random.choice(top_configs)[0]
        return self.mutate_params(base)

    def merge_with_base(self, trial_params, base_params):
        trial_params.update({
            "block_size": base_params["block_size"],
            "max_iters": base_params["max_iters"],
            "eval_interval": base_params["eval_interval"],
            "eval_iters": base_params["eval_iters"],
            "n_splits": base_params["n_splits"],
            "fine_tune": base_params["fine_tune"]
        })
        return trial_params

    def run_trial(self, params):
        with open('AutoEve/setup.json', 'w') as f:
            json.dump(params, f, indent=4)

        log_file = 'AutoEve/model.log'
        if os.path.exists(log_file):
            os.remove(log_file)

        print(f"Running trial with: {params}")
        process = subprocess.Popen(['python', 'AutoEve/model.py'])

        val_loss = float('inf')

        try:
            while True:
                try:
                    val_loss = self.parse_val_loss(log_file)
                    if val_loss != float('inf'):
                        break
                except:
                    pass
                time.sleep(0.5)
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
        except Exception as e:
            print(f"Error during training: {e}")
            process.kill()

        return val_loss

    def parse_val_loss(self, log_file):
        if not os.path.exists(log_file):
            return float('inf')

        with open(log_file, 'r') as f:
            lines = f.readlines()

        losses = []
        for line in lines:
            if "val loss" in line:
                match = re.search(r"val loss ([\d.]+)", line)
                if match:
                    losses.append(float(match.group(1)))
        return losses[-1] if losses else float('inf')

    def tune(self, n_trials=50):
        base = self.load_base_params()
        for i in range(n_trials):
            os.system("cls" if os.name == "nt" else "clear")
            trial_params = self.suggest_params()
            full_params = self.merge_with_base(trial_params, base)
            val_loss = self.run_trial(full_params)

            self.history.append((trial_params, val_loss))

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_params = full_params
                logging.info(f"New best loss: {val_loss:.4f} | Params: {trial_params}")
                self.save_params(full_params, 'AutoEve/best_setup.json')

        print(f"Best loss: {self.best_loss}")
        print(f"Best params: {self.best_params}")
        self.save_params(self.best_params)

    def save_params(self, params, filename='AutoEve/setup.json'):
        with open(filename, 'w') as f:
            json.dump(params, f, indent=4)

if __name__ == "__main__":
    search_space = {
        "batch_size": [2, 4, 8, 16, 32],
        "learning_rate": [1e-5, 5e-5, 1e-4, 3e-4, 2e-4],
        "n_embd": [128, 256, 384, 512, 640],
        "n_head": [2, 4, 6, 8, 10],
        "n_layer": [2, 4, 6, 8, 10],
        "dropout": [0.1, 0.2, 0.3, 0.4, 0.5],
        "weight_decay": [0.01, 0.02, 0.03, 0.04, 0.05]
    }

    tuner = SmartHparamTuner(search_space)
    tuner.tune(n_trials=100)
