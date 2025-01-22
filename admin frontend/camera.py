import sys
import os
import pygame



class CameraSystemInvoker:
    def __init__(self):
        self.setup_paths()
        self.camera_system = None  # Placeholder for the CameraSystem object
        self.is_running = False
        from camprocess.multiplecam_process import startCameraSystem, stopCameraSystem
        from idle_detection.test2 import idle_detection_start
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


    def start_camera_system(self):
        if not self.is_running:
            self.camera_system = self.startCameraSystem()  # Start the CameraSystem
            self.startIdleDetector()
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
        screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Camera Control")

        font = pygame.font.Font(None, 36)
        buttons = [
            {"label": "Start Camera", "rect": pygame.Rect(100, 100, 200, 50), "action": "start"},
            {"label": "Stop Camera", "rect": pygame.Rect(350, 100, 200, 50), "action": "stop"},
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
                                self.start_camera_system()
                            elif button["action"] == "stop":
                                self.stop_camera_system()

            screen.fill((30, 30, 30))
            for button in buttons:
                pygame.draw.rect(screen, (0, 128, 255), button["rect"])
                text_surface = font.render(button["label"], True, (255, 255, 255))
                screen.blit(text_surface, (button["rect"].x + 10, button["rect"].y + 10))

            pygame.display.flip()

        self.stop_camera_system()
        pygame.quit()


if __name__ == "__main__":
    camera_system = CameraSystemInvoker()
    camera_system.run()
