import pygame
import sys
import requests

users = ['1', '2', '3']

class LoginPage:
    def __init__(self, background_image_path):
        # Initialize Pygame
        pygame.init()

        # Screen dimensions
        self.screen_width = 800
        self.screen_height = 600

        # Colors
        self.white = (255, 255, 255)
        self.black = (0, 0, 0)
        self.gray = (200, 200, 200)
        self.blue = (0, 0, 255)
        self.red = (255, 0, 0)

        # Fonts
        self.font = pygame.font.Font(None, 36)

        # Screen setup
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Login Page")

        # Load background image
        self.background_image = pygame.image.load(background_image_path)
        self.background_image = pygame.transform.scale(self.background_image, (self.screen_width, self.screen_height))

        # Input box
        self.input_box = pygame.Rect(300, 250, 200, 50)
        self.color_inactive = self.gray
        self.color_active = self.black
        self.color = self.color_inactive
        self.active = False
        self.text = ''
        self.wid = 0

        # Login button
        self.button = pygame.Rect(350, 320, 100, 50)

        # Error message
        self.error_message = ''

    def draw_text(self, text, font, color, surface, x, y):
        textobj = font.render(text, True, color)
        textrect = textobj.get_rect()
        textrect.topleft = (x, y)
        surface.blit(textobj, textrect)

    def check_wid_exists(self, wid):
        url = f"http://localhost:5000/users/wid/{wid}/type"
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                user_type = response.json()
                print('User Type:', user_type)
                return user_type
            elif response.status_code == 404:
                print('Error: User not found')
                return False
            else:
                error = response.json()
                print('Error:', error)
                return False
        except requests.exceptions.RequestException as error:
            print('Network error:', error)
            return False

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if self.input_box.collidepoint(event.pos):
                        self.active = not self.active
                    else:
                        self.active = False
                    self.color = self.color_active if self.active else self.color_inactive

                    if self.button.collidepoint(event.pos):
                        if self.text.isdigit():
                            user_type = self.check_wid_exists(int(self.text))
                            if user_type:
                                self.wid = self.text
                                print(f"User ID entered: {self.text}")
                                self.error_message = ''
                                return [int(self.wid), user_type]
                                running = False  # Exit the run function
                            else:
                                self.error_message = 'User ID not found'
                        else:
                            self.error_message = 'Invalid User ID. Only integers are allowed.'
                        self.text = ''  # Clear the input box

                if event.type == pygame.KEYDOWN:
                    if self.active:
                        if event.key == pygame.K_RETURN:
                            if self.text.isdigit():
                                user_type = self.check_wid_exists(int(self.text))
                                if user_type:
                                    self.wid = self.text
                                    print(f"User ID entered: {self.text}")
                                    self.error_message = ''
                                    return int(self.wid), user_type
                                    running = False  # Exit the run function
                                else:
                                    self.error_message = 'User ID not found'
                            else:
                                self.error_message = 'Invalid User ID. Only integers are allowed.'
                            self.text = ''  # Clear the input box
                        elif event.key == pygame.K_BACKSPACE:
                            self.text = self.text[:-1]
                        else:
                            self.text += event.unicode

            self.screen.blit(self.background_image, (0, 0))  # Draw the background image
            self.draw_text('User ID:', self.font, self.black, self.screen, 200, 220)  # Adjusted y position
            txt_surface = self.font.render(self.text, True, self.color)
            width = max(200, txt_surface.get_width() + 10)
            self.input_box.w = width
            self.screen.blit(txt_surface, (self.input_box.x + 5, self.input_box.y + 5))
            pygame.draw.rect(self.screen, self.color, self.input_box, 2)

            # Draw login button
            pygame.draw.rect(self.screen, self.blue, self.button)
            self.draw_text('Login', self.font, self.white, self.screen, self.button.x + 20, self.button.y + 10)

            # Draw error message
            if self.error_message:
                self.draw_text(self.error_message, self.font, self.red, self.screen, 300, 400)

            pygame.display.flip()
            pygame.time.Clock().tick(30)

if __name__ == "__main__":
    background_image_path = '20150402_004.png'  # Replace with your image path
    login_page = LoginPage(background_image_path)
    print(login_page.run())
