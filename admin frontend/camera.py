import subprocess
import os
import pygame

class CameraSystem:
    def __init__(self):
        self.process = None

    def start_camera_system(self):
        if self.process is None:
            camprocess_main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../warehouse-path-optimizing/camprocess/main.py'))
            if os.path.exists(camprocess_main_path):
                self.process = subprocess.Popen(["python", camprocess_main_path], shell=True)
                print("Started Camera System")
            else:
                print(f"Error: {camprocess_main_path} does not exist")

    def stop_camera_system(self):
        if self.process is not None:
            self.process.terminate()
            self.process.wait()
            self.process = None
            print("Stopped Camera System")

    def run(self):
        pygame.init()
        screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Camera Control")

        font = pygame.font.Font(None, 36)
        buttons = [
            {"label": "Start Camera ", "rect": pygame.Rect(100, 100, 200, 50), "action": "start"},
            {"label": "Stop Camera ", "rect": pygame.Rect(350, 100, 200, 50), "action": "stop"},
        ]

        running = True
        show_message = True
        while running:
         for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                # Instead of quitting, you can set a flag or call a method to go back to the previous page
                running = False  # Set running to False to exit the current loop
                # Add your code here to go back to the previous page
            elif event.type == pygame.MOUSEBUTTONDOWN:
                for button in buttons:
                    if button["rect"].collidepoint(event.pos):
                        if button["action"] == "start":
                            self.start_camera_system()
                        elif button["action"] == "stop":
                            self.stop_camera_system()

            screen.fill((30, 30, 30))
            if show_message:
                message_surface = font.render("Press ESC to go back", True, (255, 255, 255))
                screen.blit(message_surface, (250, 50))

            for button in buttons:
                pygame.draw.rect(screen, (0, 128, 255), button["rect"])
                text_surface = font.render(button["label"], True, (255, 255, 255))
                screen.blit(text_surface, (button["rect"].x + 10, button["rect"].y + 10))

            pygame.display.flip()


        self.stop_camera_system()
        pygame.quit()

if __name__ == "__main__":
    camera_system = CameraSystem()
    camera_system.run()