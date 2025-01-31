from login import LoginPage
from menu import MainMenu

if __name__ == '__main__':
    login_page = LoginPage('20150402_004.jpg')
    user=login_page.run()
    app = MainMenu()
    app.run(user)

    