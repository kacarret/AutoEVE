# open and read a file and all contents line by line
with open('test.txt', 'r', encoding='utf-8') as f:
    for line in f:
        print(line)