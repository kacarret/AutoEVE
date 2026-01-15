import re

def find_bytes_and_words(file_path, byte_sequence):
    # Convert the byte sequence from a list of hex values to bytes
    byte_sequence = bytes(byte_sequence)
    
    words_with_special_char = set()
    
    try:
        # Open the file in binary mode
        with open(file_path, 'rb') as file:
            content = file.read()
            
            # Find all occurrences of the byte sequence
            start = 0
            while True:
                start = content.find(byte_sequence, start)
                if start == -1:
                    break
                end = start + len(byte_sequence)
                
                # Find the start and end of the surrounding word
                before = content.rfind(b' ', 0, start) + 1
                after = content.find(b' ', end)
                if after == -1:
                    after = len(content)
                
                # Extract the word containing the byte sequence
                word = content[before:after].decode('utf-8', errors='replace').strip()
                words_with_special_char.add(word)
                
                start = end  # Move past this occurrence
        
        # Print results
        if words_with_special_char:
            print(f"Words containing the byte sequence {byte_sequence}:")
            for word in words_with_special_char:
                print(word)
        else:
            print(f"No words containing the byte sequence {byte_sequence} were found.")
    
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
file_path = 'new AI created dataset\Synth_data.txt'  # Replace with the path to your text file
byte_sequence = [0xef, 0xbf, 0xbd]  # Byte sequence to find
find_bytes_and_words(file_path, byte_sequence)
