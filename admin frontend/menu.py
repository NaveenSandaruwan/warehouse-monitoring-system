import pygame
import cv2
import numpy as np
from report import ReportGenerator
from camera import CameraSystemInvoker

class MainMenu:
    def __init__(self, screen):
        self.screen = screen
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.options = ["Generate Report", "Start Camera System", "Exit"]
        self.selected_option = 0

        # Load and blur background image using OpenCV
        background_image = cv2.imread("admin frontend/warehouse1.jpg")
        blurred_image = cv2.GaussianBlur(background_image, (21, 21), 0)

        # Convert the blurred image to a format Pygame can use
        self.background = pygame.image.frombuffer(blurred_image.tobytes(), blurred_image.shape[1::-1], "BGR")

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self.selected_option = (self.selected_option - 1) % len(self.options)
                    elif event.key == pygame.K_DOWN:
                        self.selected_option = (self.selected_option + 1) % len(self.options)
                    elif event.key == pygame.K_RETURN:
                        if self.selected_option == 0:
                            ReportGenerator(self.screen).generate()
                        elif self.selected_option == 1:
                            CameraSystemInvoker().run()
                        
                        elif self.selected_option == 2:
                            running = False
                    elif event.key == pygame.K_ESCAPE:
                        running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left mouse button
                        for i, option in enumerate(self.options):
                            text_surface = self.font.render(option, True, (255, 0, 0))
                            text_rect = text_surface.get_rect(center=(self.screen.get_width() // 2, self.screen.get_height() // 2 - len(self.options) * 20 + i * 40))
                            if text_rect.collidepoint(event.pos):
                                self.selected_option = i
                                if self.selected_option == 0:
                                    ReportGenerator(self.screen).generate()
                                elif self.selected_option == 1:
                                    CameraSystemInvoker().run()
                                
                                elif self.selected_option == 2:
                                    running = False

            self.screen.blit(self.background, (0, 0))  # Draw the blurred background image
            for i, option in enumerate(self.options):
                text_surface = self.font.render(option, True, (255, 165, 0))
                if i == self.selected_option:
                    text_surface = pygame.transform.scale(text_surface, (int(text_surface.get_width() * 1.5), int(text_surface.get_height() * 1.5)))
                if i == self.selected_option:
                    text_surface = pygame.transform.scale(text_surface, (text_surface.get_width() * 1.2, text_surface.get_height() * 1.2))
                text_rect = text_surface.get_rect(center=(self.screen.get_width() // 2, self.screen.get_height() // 2 - len(self.options) * 20 + i * 40))
                self.screen.blit(text_surface, text_rect)

            pygame.display.flip()
            self.clock.tick(30)