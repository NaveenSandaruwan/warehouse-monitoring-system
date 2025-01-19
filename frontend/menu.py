import pygame
import requests

class MainMenu:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption('Main Menu')
        
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.LIGHT_BLUE = (173, 216, 230)
        
        self.font = pygame.font.Font(None, 36)
        
        self.tasks_button = pygame.Rect(300, 200, 200, 50)
        self.quit_button = pygame.Rect(300, 300, 200, 50)
        
        self.running = True

    def draw_menu(self):
        self.screen.fill(self.WHITE)
        pygame.draw.rect(self.screen, self.BLACK, self.tasks_button)
        pygame.draw.rect(self.screen, self.BLACK, self.quit_button)
        
        tasks_text = self.font.render('Tasks', True, self.WHITE)
        quit_text = self.font.render('Quit', True, self.WHITE)
        
        self.screen.blit(tasks_text, (self.tasks_button.x + 50, self.tasks_button.y + 10))
        self.screen.blit(quit_text, (self.quit_button.x + 50, self.quit_button.y + 10))
        
        pygame.display.flip()

    def fetch_tasks(self):
        try:
            response = requests.get("http://127.0.0.1:5000/tasks")  # Flask API
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error fetching tasks: {response.status_code}")
                return []
        except Exception as e:
            print(f"Error: {e}")
            return []

    def tasks_window(self):
        running = True
        tasks = self.fetch_tasks()
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
            
            self.screen.fill(self.WHITE)
            tasks_text = self.font.render('Tasks Window - Press ESC to return', True, self.BLACK)
            self.screen.blit(tasks_text, (100, 250))
            
            y = 300
            for task in tasks:
                task_rect = pygame.Rect(90, y, 620, 40)
                pygame.draw.rect(self.screen, self.LIGHT_BLUE, task_rect)
                task_text = self.font.render(f"- {task['task_description']}", True, self.BLACK)
                self.screen.blit(task_text, (100, y + 5))
                y += 50

            pygame.display.flip()

        self.draw_menu()

    def run(self):
        while self.running:
            self.draw_menu()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if self.tasks_button.collidepoint(event.pos):
                        self.tasks_window()
                    elif self.quit_button.collidepoint(event.pos):
                        self.running = False

        pygame.quit()

if __name__ == "__main__":
    main_menu = MainMenu()
    main_menu.run()
