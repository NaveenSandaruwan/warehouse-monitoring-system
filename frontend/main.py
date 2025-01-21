from login import LoginPage
from menu import MainMenu

if __name__ == '__main__':
    login_page = LoginPage()
    user=login_page.run()
    app = MainMenu()
    app.run(user)
