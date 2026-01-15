from datasets import load_dataset
import re
import os

dataset = load_dataset("openwebtext", split="train", streaming=True, trust_remote_code=True)

os.makedirs("AutoEDCA\\test\\subsets", exist_ok=True)

output_file = "AutoEDCA\\test\\subsets\\tester.txt"
if os.path.exists(output_file):
    os.remove(output_file)
    print(f"Removed {output_file}")

# inside the datra corpus there is 8013769 examples total
# the full 8M is 40gb
# we want something manageable of 500 MB
# so we want around 1% of the data
# so around 80,000 examples should be 400mb of data
# also assume that the each example is around 5KB in size(average)
max_examples = 80000

print(f"Writing {max_examples} examples to {output_file}")
print(f"Max examples: {max_examples}")
print(f"Max examples in MB: {(max_examples * 5) / 1024}")

delimiter = "\n\n<|endoftext|>\n\n"  # common delimiter used by OpenAI and others

with open(output_file, "w", encoding="utf-8") as f:
    for i, example in enumerate(dataset):
        text = example["text"].strip()

        # Replace multiple newlines (2 or more) with a single space
        text = re.sub(r'\n{2,}', ' ', text)

        # Optional: also replace single newlines with spaces to flatten completely
        text = text.replace('\n', ' ')

        f.write(text + delimiter)

        if i % 100 == 0:
            print(f"Wrote {i} examples...")
        if i >= max_examples - 1:
            break

print("Done!")