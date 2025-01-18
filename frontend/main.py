import pygame
import sys
import os
import json
from login import LoginPage
from grid_generate import WarehouseGrid


def read_grid_layout(file_path):
    # Ensure the file path is correct
    file_path = os.path.join(os.path.dirname(__file__), file_path)
    with open(file_path, 'r') as file:
        grid = json.load(file)
    return grid

if __name__ == '__main__':
    login_page = LoginPage()
    login_page.run()
    grid = read_grid_layout('grid_layout.txt')
    warehouse_grid = WarehouseGrid(grid)
    warehouse_grid.run()