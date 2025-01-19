import pygame
import sys
import os
import json
from login import LoginPage
from grid_generate import WarehouseGrid

if __name__ == '__main__':
    login_page = LoginPage()
    login_page.run()
