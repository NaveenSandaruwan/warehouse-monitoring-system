import os
import pygame
import sys
import requests
import json
from test import SimulationRunner

class MainMenu:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Main Menu")

        # Load the background images
        self.background_image = pygame.image.load("iron-steel-material-storage.jpg")
        self.background_image = pygame.transform.scale(self.background_image, (800, 600))  # Scale to fit window
        self.tasks_background_image = pygame.image.load("warehouse_05.jpg")  # Load the new image
        self.tasks_background_image = pygame.transform.scale(self.tasks_background_image, (800, 600))  # Scale to fit window

        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.ORANGE = (255, 165, 0)
        self.LIGHT_ORANGE = (255, 200, 100)
        self.GRAY = (169, 169, 169)  # Color for walls

        self.font = pygame.font.Font(None, 36)
        self.type = None

        self.tasks_button = pygame.Rect(300, 200, 200, 50)
        self.map_button = pygame.Rect(300, 300, 200, 50)
        self.quit_button = pygame.Rect(300, 400, 200, 50)

        self.running = True
        self.wid = None

    def draw_menu(self):
        self.screen.blit(self.background_image, (0, 0))  # Draw background image
        mouse_pos = pygame.mouse.get_pos()

        if self.type == "forklift":
            tasks_button_color = self.LIGHT_ORANGE if self.tasks_button.collidepoint(mouse_pos) else self.ORANGE
            pygame.draw.rect(self.screen, tasks_button_color, self.tasks_button)
            tasks_text = self.font.render("Tasks", True, self.WHITE)
            self.screen.blit(
                tasks_text,
                (
                    self.tasks_button.centerx - tasks_text.get_width() // 2,
                    self.tasks_button.centery - tasks_text.get_height() // 2,
                ),
            )

        map_button_color = self.LIGHT_ORANGE if self.map_button.collidepoint(mouse_pos) else self.ORANGE
        quit_button_color = self.LIGHT_ORANGE if self.quit_button.collidepoint(mouse_pos) else self.ORANGE

        pygame.draw.rect(self.screen, map_button_color, self.map_button)
        pygame.draw.rect(self.screen, quit_button_color, self.quit_button)

        map_text = self.font.render("Map", True, self.WHITE)
        quit_text = self.font.render("Quit", True, self.WHITE)

        self.screen.blit(
            map_text,
            (
                self.map_button.centerx - map_text.get_width() // 2,
                self.map_button.centery - map_text.get_height() // 2,
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
            response = requests.get("http://localhost:5000/tasks/unoccupied")
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
                            show_simulation = SimulationRunner()
                            start = tuple(map(int, task["start"].strip("()").split(',')))
                            goal = tuple(map(int, task["end"].strip("()").split(',')))
                            itemsize = task["itemsize"]
                            iterations = int(task["iterations"])
                            camcoordinates = [(1, 0), (13, 0)]
                            isdone = show_simulation.run(start, goal, camcoordinates, itemsize, self.wid, iterations)
                            if isdone:
                                print(task["_id"])
                                self.delete_task(task["_id"])
                            running = False
                        y_offset += 40

            # Blur the new background image
            blurred_background = pygame.transform.smoothscale(self.tasks_background_image, (80, 60))
            blurred_background = pygame.transform.smoothscale(blurred_background, (800, 600))
            self.screen.blit(blurred_background, (0, 0))  # Draw the blurred background image

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

    def delete_task(self, task_id):
        url = f"http://localhost:5000/tasks/{task_id}"
        headers = {
            "Accept": "application/json"
        }
        response = requests.delete(url, headers=headers)
        return response.json()

    def map_window(self):
        sections = fetch_sections_and_locations()
        running = True
        grid_size = len(grid)
        cell_size = 20
        start_pos = None
        goal_pos = None

        def find_nearest_empty_cell(x, y):
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < grid_size and 0 <= ny < grid_size and grid[ny][nx] == 0:
                    return (ny, nx)
            return None

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = event.pos
                    grid_x = mouse_pos[0] // cell_size
                    grid_y = mouse_pos[1] // cell_size
                    if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                        if grid[grid_y][grid_x] == 1:
                            nearest_empty = find_nearest_empty_cell(grid_x, grid_y)
                            if nearest_empty:
                                grid_y, grid_x = nearest_empty
                        if start_pos is None:
                            start_pos = (grid_y, grid_x)  # Reverse x and y
                        elif goal_pos is None:
                            goal_pos = (grid_y, grid_x)  # Reverse x and y
                            show_simulation = SimulationRunner()
                            camcoordinates = [(1, 0), (13, 0)]
                            isdone = show_simulation.run(start_pos, goal_pos, camcoordinates, "small", self.wid, 1)
                            running = False

            self.screen.fill(self.WHITE)  # Clear screen with white background
            pygame.display.set_caption("Select Start and Goal - Press ESC to return")

            for x in range(grid_size):
                for y in range(grid_size):
                    rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
                    if grid[y][x] == 1:
                        pygame.draw.rect(self.screen, self.GRAY, rect)
                    pygame.draw.rect(self.screen, self.BLACK, rect, 1)

            for section in sections:
                section_name = section['name']
                coord = tuple(map(int, section['coordinates'].strip("()").split(',')))
                pygame.draw.rect(self.screen, self.BLUE, (coord[1] * cell_size, coord[0] * cell_size, cell_size, cell_size))
                text = self.font.render(section_name, True, self.BLACK)
                self.screen.blit(text, (coord[1] * cell_size, coord[0] * cell_size))

            if start_pos:
                pygame.draw.rect(self.screen, self.BLUE, (start_pos[1] * cell_size, start_pos[0] * cell_size, cell_size, cell_size))  # Reverse x and y
            if goal_pos:
                pygame.draw.rect(self.screen, self.LIGHT_BLUE, (goal_pos[1] * cell_size, goal_pos[0] * cell_size, cell_size, cell_size))  # Reverse x and y

            pygame.display.flip()

        self.draw_menu()

    def run(self, user):
        self.wid = user[0]
        self.type = user[1]["type"]
        while self.running:
            self.draw_menu()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if self.tasks_button.collidepoint(event.pos):
                        self.tasks_window()
                    elif self.map_button.collidepoint(event.pos):
                        self.map_window()
                    elif self.quit_button.collidepoint(event.pos):
                        self.running = False

        pygame.quit()
        sys.exit()

def read_grid_layout(file_path):
    # Ensure the file path is correct
    file_path = os.path.join(os.path.dirname(__file__), file_path)
    with open(file_path, 'r') as file:
        grid = json.load(file)
    return grid

def fetch_sections_and_locations():
    try:
        response = requests.get("http://localhost:5000/locations")
        response.raise_for_status()  # Check if the request was successful
        return response.json()  # Return the fetched locations as a JSON object
    except requests.exceptions.RequestException as e:
        print(f"Error fetching sections and locations: {e}")
        return []  # Return an empty list in case of an error

grid = read_grid_layout('../simulation/grid_layout.txt')

if __name__ == "__main__":
    main_menu = MainMenu()
    main_menu.run()
