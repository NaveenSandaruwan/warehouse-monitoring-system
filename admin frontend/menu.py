import pygame
from report import ReportGenerator
from camera import CameraSystem


class MainMenu:
    def __init__(self, screen):
        self.screen = screen
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.options = ["Generate Report", "Start Camera System"]
        self.selected_option = 0

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
                            CameraSystem().run()
                    elif event.key == pygame.K_ESCAPE:
                        running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left mouse button
                        for i, option in enumerate(self.options):
                            text_surface = self.font.render(option, True, (255, 255, 255))
                            text_rect = text_surface.get_rect(topleft=(100, 100 + i * 40))
                            if text_rect.collidepoint(event.pos):
                                self.selected_option = i
                                if self.selected_option == 0:
                                    ReportGenerator(self.screen).generate()
                                elif self.selected_option == 1:
                                    CameraSystem().run()

            self.screen.fill((0, 0, 0))
            for i, option in enumerate(self.options):
                color = (255, 255, 255) if i == self.selected_option else (100, 100, 100)
                text_surface = self.font.render(option, True, color)
                self.screen.blit(text_surface, (100, 100 + i * 40))

            pygame.display.flip()
            self.clock.tick(30)