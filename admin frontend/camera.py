import multiprocessing
import sys
import os
import pygame

# Add the path to multiplecam_process.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../camprocess')))

from multiplecam_process import start_camera

class CameraSystem:
    def __init__(self):
        self.processes = {}

    def start_camera(self, camera_id):
        if camera_id not in self.processes:
            p = multiprocessing.Process(target=start_camera, args=(camera_id,))
            p.start()
            self.processes[camera_id] = p
            print(f"Started Camera {camera_id + 1}")

    def stop_camera(self, camera_id):
        if camera_id in self.processes:
            self.processes[camera_id].terminate()
            self.processes[camera_id].join()
            del self.processes[camera_id]
            print(f"Stopped Camera {camera_id + 1}")

    def stop_all_cameras(self):
        for camera_id in list(self.processes.keys()):
            self.stop_camera(camera_id)

    def run(self):
        pygame.init()
        screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Camera Control")

        font = pygame.font.Font(None, 36)
        buttons = [
            {"label": "Start Camera 1", "rect": pygame.Rect(100, 100, 200, 50), "action": "start", "camera_id": 0},
            {"label": "Stop Camera 1", "rect": pygame.Rect(350, 100, 200, 50), "action": "stop", "camera_id": 0},
            {"label": "Start Camera 2", "rect": pygame.Rect(100, 200, 200, 50), "action": "start", "camera_id": 1},
            {"label": "Stop Camera 2", "rect": pygame.Rect(350, 200, 200, 50), "action": "stop", "camera_id": 1},
            {"label": "Start Camera 3", "rect": pygame.Rect(100, 300, 200, 50), "action": "start", "camera_id": 2},
            {"label": "Stop Camera 3", "rect": pygame.Rect(350, 300, 200, 50), "action": "stop", "camera_id": 2},
        ]

        running = True
        while running:
            for event in pygame.event.get():
             if event.type == pygame.QUIT:
                running = False
             elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
             elif event.type == pygame.MOUSEBUTTONDOWN:
                for button in buttons:
                    if button["rect"].collidepoint(event.pos):
                        if button["action"] == "start":
                            self.start_camera(button["camera_id"])
                        elif button["action"] == "stop":
                            self.stop_camera(button["camera_id"])

            screen.fill((30, 30, 30))
            for button in buttons:
                pygame.draw.rect(screen, (0, 128, 255), button["rect"])
                text_surface = font.render(button["label"], True, (255, 255, 255))
                screen.blit(text_surface, (button["rect"].x + 10, button["rect"].y + 10))

            # Add the topic at the beginning of the screen
            topic_surface = font.render("Press ESC to go back", True, (255, 255, 255))
            screen.blit(topic_surface, (10, 10))

            pygame.display.flip()

        self.stop_all_cameras()
        pygame.quit()

if __name__ == "__main__":
    camera_system = CameraSystem()
    camera_system.run()