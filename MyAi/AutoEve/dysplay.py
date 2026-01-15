import threading
import pygame
import time
from ModelGeneration import chat

pygame.init()

pygame.display.set_caption("Evedysp")

black = (0, 0, 0)
white = (255, 255, 255)

dem = 1920, 1080
screen = pygame.display.set_mode((dem))  # Set the screen dimensions to 640x480
running = True

class InputReader(threading.Thread):
    def __init__(self):
        super().__init__()
        self.input_text = ""
        self.timer = None  # Timer to clear input after 10 seconds

    def clear_input(self):
        self.input_text = ''

    def run(self):
        while True:
            self.input_text = chat(input("You: "))
            # Cancel any previous timer if user inputs text again
            if self.timer:
                self.timer.cancel()
            
            # Set a new timer to clear input after 10 seconds of inactivity
            self.timer = threading.Timer(10.0, self.clear_input)
            self.timer.start()

            lines = []
            words = self.input_text.split()
            current_line = ''
            
            for word in words:
                if font.size(current_line + ' ' + word)[0] > screen.get_width():
                    lines.append(current_line)
                    current_line = word
                else:
                    if current_line:
                        current_line += ' '
                    current_line += word

            lines.append(current_line)
            self.input_text = '\n'.join(lines)

input_reader = InputReader()
input_reader.start()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((black))
    font = pygame.font.Font(None, 46)

    lines = input_reader.input_text.split('\n')
    for i, line in enumerate(lines):
        text = font.render(line, True, white, black)
        # get the lower half of the screen
        text_rect = text.get_rect(center=(screen.get_width()// 2, screen.get_height() // 2 + i * text.get_height() + dem[1] * 0.25))
        screen.blit(text, text_rect)

    pygame.display.flip()  # update the display