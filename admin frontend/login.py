import pygame

class LoginPage:
    def __init__(self, screen):
        self.screen = screen
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.input_box = pygame.Rect(100, 150, 200, 32)
        self.login_button = pygame.Rect(100, 200, 100, 50)
        self.color_inactive = pygame.Color('lightskyblue3')
        self.color_active = pygame.Color('dodgerblue2')
        self.color = self.color_inactive
        self.active = False
        self.text = ''
        self.done = False

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

            self.screen.fill((30, 30, 30))
            # Render the username label
            username_label = self.font.render("Username:", True, (255, 255, 255))
            self.screen.blit(username_label, (100, 100))
            # Render the input box
            txt_surface = self.font.render(self.text, True, self.color)
            width = max(200, txt_surface.get_width() + 10)
            self.input_box.w = width
            self.screen.blit(txt_surface, (self.input_box.x + 5, self.input_box.y + 5))
            pygame.draw.rect(self.screen, self.color, self.input_box, 2)
            # Render the login button
            pygame.draw.rect(self.screen, (0, 255, 0), self.login_button)
            login_text = self.font.render("Login", True, (0, 0, 0))
            self.screen.blit(login_text, (self.login_button.x + 10, self.login_button.y + 10))

            pygame.display.flip()
            self.clock.tick(30)
        return True