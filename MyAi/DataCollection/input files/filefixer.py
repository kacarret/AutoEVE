import os
import re
from tqdm import tqdm

def clean_line(text):
    quote = '"'
    # Define a list of allowed characters (letters and spaces)
    allowed_chars = set(" abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,?!':;()~`&$0123456789"+quote)
    # Filter out characters not in the allowed set
    clean_text = ''.join(c for c in text if c in allowed_chars)
    return clean_text

def process_files(input_directory, output_file):
    files = [f for f in os.listdir(input_directory) if f.endswith('.txt')]
    print(f"Files found: {files}")  # Debug statement
    
    total_lines = 0
    file_line_counts = {}

    for file in files:
        file_path = os.path.join(input_directory, file)
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as infile:
                line_count = sum(1 for _ in infile)
                file_line_counts[file] = line_count
                total_lines += line_count
                print(f"File: {file_path}, Lines: {line_count}")  # Debug statement
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")

    print(f"Total lines to process: {total_lines}")

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for file in tqdm(files, desc="Processing files"):
            file_path = os.path.join(input_directory, file)
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as infile:
                    with tqdm(total=file_line_counts[file], desc=f"Processing lines in {file}", leave=False) as file_progress:
                        for line in infile:
                            cleaned_line = clean_line(line)
                            if cleaned_line:
                                outfile.write(' '.join(cleaned_line.split()) + ' ')
                            file_progress.update(1)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
    
    print(f"Processing complete. Output written to {output_file}")


# Configuration
input_directory = 'DataCollection\input files'  # Replace with your input directory
output_file = 'input.txt'  # Replace with your desired output file name

process_files(input_directory, output_file)
