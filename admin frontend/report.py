import pygame
import requests

class ReportGenerator:
    def __init__(self, screen):
        self.screen = screen
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)  # Smaller font for data
        self.options = ["Worker Report", "Task Assigned Report", "Daily Report"]
        self.selected_option = 0

        # Load background image
        self.background = pygame.image.load("admin frontend/warehouse1.jpg").convert()

    def generate(self):
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
                            self.generate_worker_report()
                        elif self.selected_option == 1:
                            self.generate_task_assigned_report()
                        elif self.selected_option == 2:
                            self.generate_daily_report()
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
                                    self.generate_worker_report()
                                elif self.selected_option == 1:
                                    self.generate_task_assigned_report()
                                elif self.selected_option == 2:
                                    self.generate_daily_report()

            self.screen.blit(self.background, (0, 0))  # Draw the background image
            for i, option in enumerate(self.options):
                color = (255, 255, 255) if i == self.selected_option else (100, 100, 100)
                text_surface = self.font.render(option, True, color)
                self.screen.blit(text_surface, (100, 100 + i * 40))

            pygame.display.flip()
            self.clock.tick(30)

    def generate_worker_report(self):
        # Fetch data from the API
        users_response = requests.get("http://localhost:5000/users")
        works_response = requests.get("http://localhost:5000/works")

        if users_response.status_code == 200 and works_response.status_code == 200:
            users = users_response.json()
            works = works_response.json()

            # Calculate total work done for each user
            work_done = {}
            for work in works:
                wid = work["wid"]
                for work_entry in work["work"]:
                    work_done[wid] = work_done.get(wid, 0) + work_entry["work_done"]

            running = True
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False

                self.screen.blit(self.background, (0, 0))  # Draw the background image

                # Display the grid headers
                headers = ["Worker Name", "Type", "Current Location", "Status", "Total Work Done"]
                for i, header in enumerate(headers):
                    header_text = self.font.render(header, True, (255, 255, 255))
                    self.screen.blit(header_text, (50 + i * 200, 50))

                # Display the user data
                for j, user in enumerate(users):
                    wid = user["wid"]
                    total_work_done = work_done.get(wid, 0)
                    user_data = [
                        user["name"],
                        user["type"],
                        user["current_location"],
                        str(user["status"]),
                        str(total_work_done)
                    ]
                    for i, data in enumerate(user_data):
                        data_text = self.small_font.render(data, True, (255, 255, 255))
                        self.screen.blit(data_text, (50 + i * 200, 100 + j * 40))

                pygame.display.flip()
                self.clock.tick(30)
        else:
            print("Error fetching data from API")

    def generate_task_assigned_report(self):
        # Fetch data from the API
        tasks_response = requests.get("http://localhost:5000/tasks")

        if tasks_response.status_code == 200:
            tasks = tasks_response.json()

            # Separate occupied and unoccupied tasks
            occupied_tasks = [task for task in tasks if task["occupied"]]
            unoccupied_tasks = [task for task in tasks if not task["occupied"]]

            running = True
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False

                self.screen.blit(self.background, (0, 0))  # Draw the background image

                # Display the grid headers for occupied tasks
                headers = ["Task Description", "Status", "Created At", "Start Location", "End Location", "Updated At", "Item Size", "Iterations"]
                for i, header in enumerate(headers):
                    header_text = self.font.render(header, True, (255, 255, 255))
                    self.screen.blit(header_text, (50 + i * 150, 50))

                # Display the occupied tasks data
                for j, task in enumerate(occupied_tasks):
                    task_data = [
                        task["task_description"],
                        task["status"],
                        task["created_at"],
                        task["start"],
                        task["end"],
                        task["updated_at"],
                        task["itemsize"],
                        str(task["iterations"])
                    ]
                    for i, data in enumerate(task_data):
                        data_text = self.small_font.render(data, True, (255, 255, 255))
                        self.screen.blit(data_text, (50 + i * 150, 100 + j * 40))

                # Display the grid headers for unoccupied tasks
                for i, header in enumerate(headers):
                    header_text = self.font.render(header, True, (255, 255, 255))
                    self.screen.blit(header_text, (50 + i * 150, 400))

                # Display the unoccupied tasks data
                for j, task in enumerate(unoccupied_tasks):
                    task_data = [
                        task["task_description"],
                        task["status"],
                        task["created_at"],
                        task["start"],
                        task["end"],
                        task["updated_at"],
                        task["itemsize"],
                        str(task["iterations"])
                    ]
                    for i, data in enumerate(task_data):
                        data_text = self.small_font.render(data, True, (255, 255, 255))
                        self.screen.blit(data_text, (50 + i * 150, 450 + j * 40))

                pygame.display.flip()
                self.clock.tick(30)
        else:
            print("Error fetching data from API")

    def generate_daily_report(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

            self.screen.blit(self.background, (0, 0))  # Draw the background image
            report_text = self.font.render("Daily Report", True, (255, 255, 255))
            self.screen.blit(report_text, (100, 100))

            pygame.display.flip()
            self.clock.tick(30)