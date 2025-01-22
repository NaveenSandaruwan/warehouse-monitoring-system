import pygame
import sys
from login import LoginPage
from menu import MainMenu

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800

# Create the main window
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Admin Dashboard")

def main():
    login_page = LoginPage(screen)
    if login_page.run():
        app = MainMenu(screen)
        app.run()

if __name__ == '__main__':
    main()
    pygame.quit()
    sys.exit()