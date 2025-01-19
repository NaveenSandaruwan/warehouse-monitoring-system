import pygame
import sys
import requests
from test import SimulationRunner

class MainMenu:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Main Menu")

        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.BLUE = (0, 0, 255)
        self.LIGHT_BLUE = (173, 216, 230)

        self.font = pygame.font.Font(None, 36)

        self.tasks_button = pygame.Rect(300, 200, 200, 50)
        self.quit_button = pygame.Rect(300, 300, 200, 50)

        self.running = True

    def draw_menu(self):
        self.screen.fill(self.WHITE)
        mouse_pos = pygame.mouse.get_pos()

        tasks_button_color = self.LIGHT_BLUE if self.tasks_button.collidepoint(mouse_pos) else self.BLACK
        quit_button_color = self.LIGHT_BLUE if self.quit_button.collidepoint(mouse_pos) else self.BLACK

        pygame.draw.rect(self.screen, tasks_button_color, self.tasks_button)
        pygame.draw.rect(self.screen, quit_button_color, self.quit_button)

        tasks_text = self.font.render("Tasks", True, self.WHITE)
        quit_text = self.font.render("Quit", True, self.WHITE)

        # Center the text inside the buttons
        self.screen.blit(
            tasks_text,
            (
                self.tasks_button.centerx - tasks_text.get_width() // 2,
                self.tasks_button.centery - tasks_text.get_height() // 2,
            ),
        )
        self.screen.blit(
            quit_text,
            (
                self.quit_button.centerx - quit_text.get_width() // 2,
                self.quit_button.centery - quit_text.get_height() // 2,
            ),
        )

        pygame.display.flip()

    def fetch_unoccupied_tasks(self):
        try:
            # Add a query parameter or modify the API URL to fetch only unoccupied tasks
            response = requests.get("http://localhost:5000/tasks/unoccupied")  # Adjust the URL if needed
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching unoccupied tasks: {e}")
            return []

    def tasks_window(self):
        running = True
        tasks = self.fetch_unoccupied_tasks()

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = event.pos
                    y_offset = 100
                    for task in tasks:
                        task_rect = pygame.Rect(100, y_offset, 500, 30)
                        if task_rect.collidepoint(mouse_pos):
                            print(f"Selected task: {task['task_description']}")
                            # Add your logic here for when a task is selected
                            # For example:
                            # self.handle_task_selection(task)
                            show_simulation = SimulationRunner()
                            start = (2, 1)  # Initial start position (row, col)
                            goal = (13, 1)  # Goal position (row, col)
                            camcoordinates = [(1, 0), (13, 0)]
                            show_simulation.run(start, goal, camcoordinates)
                            running = False  # Exit the tasks window after selecting a task
                        y_offset += 40

            self.screen.fill(self.WHITE)
            title_text = self.font.render(
                "Unoccupied Tasks - Press ESC to return", True, self.BLACK
            )
            self.screen.blit(title_text, (100, 50))

            if not tasks:
                no_tasks_text = self.font.render("No unoccupied tasks available.", True, self.BLACK)
                self.screen.blit(no_tasks_text, (100, 100))
            else:
                y_offset = 100
                for task in tasks:
                    task_text = self.font.render(
                        f"Task: {task['task_description']}", True, self.BLACK
                    )
                    self.screen.blit(task_text, (100, y_offset))
                    y_offset += 40

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
        sys.exit()


if __name__ == "__main__":
    main_menu = MainMenu()
    main_menu.run()
