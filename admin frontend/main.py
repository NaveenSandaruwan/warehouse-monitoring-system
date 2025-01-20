import pygame
import sys
import os
import json
from login import LoginPage
from menu import MainMenu
from report import ReportGenerator
from camera import CameraSystem

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 1500
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