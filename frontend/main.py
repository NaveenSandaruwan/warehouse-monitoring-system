import pygame
import sys
import os
import json
from login import LoginPage
from grid_generate import WarehouseGrid
from menu import MainMenu

if __name__ == '__main__':
    login_page = LoginPage()
    wid=login_page.run()
    app = MainMenu()
    app.run(wid)
