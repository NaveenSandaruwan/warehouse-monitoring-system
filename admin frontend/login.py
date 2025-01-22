import pygame

class LoginPage:
    def __init__(self, screen):
        self.screen = screen
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 72)  # Larger font size
        self.input_box = pygame.Rect(0, 0, 400, 50)  # Larger input box
        self.login_button = pygame.Rect(0, 0, 200, 75)  # Larger login button
        self.color_inactive = pygame.Color('lightskyblue3')
        self.color_active = pygame.Color('dodgerblue2')
        self.color = self.color_inactive
        self.active = False
        self.text = ''
        self.done = False

        # Load background image
        self.background = pygame.image.load("admin frontend/warehouse1.jpg").convert()
        # Blur the background image
        self.background = pygame.transform.smoothscale(self.background, (self.background.get_width() // 10, self.background.get_height() // 10))
        self.background = pygame.transform.smoothscale(self.background, (self.background.get_width() * 10, self.background.get_height() * 10))

    def run(self):
        while not self.done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.done = True
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if self.input_box.collidepoint(event.pos):
                        self.active = not self.active
                    else:
                        self.active = False
                    self.color = self.color_active if self.active else self.color_inactive
                    if self.login_button.collidepoint(event.pos):
                        print(self.text)
                        self.done = True
                if event.type == pygame.KEYDOWN:
                    if self.active:
                        if event.key == pygame.K_RETURN:
                            print(self.text)
                            self.done = True
                        elif event.key == pygame.K_BACKSPACE:
                            self.text = self.text[:-1]
                        else:
                            self.text += event.unicode

            self.screen.blit(self.background, (0, 0))  # Draw the background image

            # Center positions
            screen_center_x = self.screen.get_width() // 2
            screen_center_y = self.screen.get_height() // 2

            # Render the username label
            username_label = self.font.render("Username:", True, (255, 0, 0))
            username_label_rect = username_label.get_rect(center=(screen_center_x, screen_center_y - 100))
            self.screen.blit(username_label, username_label_rect)

            # Render the input box
            self.input_box.center = (screen_center_x, screen_center_y)
            txt_surface = self.font.render(self.text, True, self.color)
            width = max(400, txt_surface.get_width() + 10)
            self.input_box.w = width
            self.screen.blit(txt_surface, (self.input_box.x + 5, self.input_box.y + 5))
            pygame.draw.rect(self.screen, self.color, self.input_box, 2)

            # Render the login button
            self.login_button.center = (screen_center_x, screen_center_y + 100)
            pygame.draw.rect(self.screen, (0, 255, 0), self.login_button)
            login_text = self.font.render("Login", True, (0, 0, 0))
            login_text_rect = login_text.get_rect(center=self.login_button.center)
            self.screen.blit(login_text, login_text_rect)

            pygame.display.flip()
            self.clock.tick(30)
        return True