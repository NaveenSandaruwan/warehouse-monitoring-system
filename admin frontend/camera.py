import sys
import os
import pygame

class CameraSystemInvoker:
    def __init__(self):
        self.setup_paths()
        self.camera_system = None  # Placeholder for the CameraSystem object
        self.is_running = False
        from camprocess.main import startCameraSystem, stopCameraSystem
        from idle_detection.test import idle_detection_start
        self.startCameraSystem = startCameraSystem
        self.stopCameraSystem = stopCameraSystem
        self.startIdleDetector = idle_detection_start

    def setup_paths(self):
        # Add the parent directory to the Python path
        parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        if parent_path not in sys.path:
            sys.path.append(parent_path)

        # Add the camprocess directory to the Python path
        camprocess_path = os.path.abspath(os.path.join(parent_path, 'camprocess'))
        if camprocess_path not in sys.path:
            sys.path.append(camprocess_path)
        idle_detector_path = os.path.abspath(os.path.join(parent_path, 'idle_detection'))
        if idle_detector_path not in sys.path:
            sys.path.append(idle_detector_path)

    def startIdelDetector(self):
        self.startIdleDetector()

    def start_camera_system(self):
        if not self.is_running:
            self.camera_system = self.startCameraSystem()  # Start the CameraSystem
            # self.startIdleDetector()
            self.is_running = True
            print("Started Camera System")
        else:
            print("Camera System is already running")

    def stop_camera_system(self):
        if self.is_running and self.camera_system:
            self.stopCameraSystem(self.camera_system)  # Stop the CameraSystem
            self.camera_system = None
            self.is_running = False
            print("Stopped Camera System")
        else:
            print("Camera System is not running")

    def run(self):
        pygame.init()
        screen = pygame.display.set_mode((1200, 800))
        pygame.display.set_caption("Camera Control")

        # Load background image
        background = pygame.image.load("admin frontend/warehouse1.jpg").convert()

        font = pygame.font.Font(None, 36)
        button_width = 400
        button_height = 70
        button_spacing = 20  # Space between buttons

        # Calculate the starting y position to center the buttons vertically
        total_height = (button_height + button_spacing) * 4 - button_spacing
        start_y = (screen.get_height() - total_height) // 2

        buttons = [
            {"label": "Start Block Detection Camera", "rect": pygame.Rect((screen.get_width() - button_width) // 2, start_y + i * (button_height + button_spacing), button_width, button_height), "action": "start"} for i in range(4)
        ]

        # Update button labels and actions
        buttons[0]["label"] = "Start Block Detection Camera"
        buttons[0]["action"] = "start"
        buttons[1]["label"] = "Stop Block Detection Camera"
        buttons[1]["action"] = "stop"
        buttons[2]["label"] = "Start Idle Detection Camera"
        buttons[2]["action"] = "start_idle"
        buttons[3]["label"] = "Stop Idle Detection Camera"
        buttons[3]["action"] = "stop_idle"

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
                                self.start_camera_system()
                            elif button["action"] == "stop":
                                self.stop_camera_system()
                            elif button["action"] == "start_idle":
                                self.startIdelDetector()

                            

            screen.blit(background, (0, 0))  # Draw the background image
            for button in buttons:
                pygame.draw.rect(screen, (0, 128, 255), button["rect"])
                text_surface = font.render(button["label"], True, (255, 255, 255))
                text_rect = text_surface.get_rect(center=button["rect"].center)
                screen.blit(text_surface, text_rect)

            pygame.display.flip()

        self.stop_camera_system()
        pygame.quit()

if __name__ == "__main__":
    camera_system = CameraSystemInvoker()
    camera_system.run()