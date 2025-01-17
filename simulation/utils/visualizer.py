import cv2
import numpy as np

def visualize_warehouse(grid, path, screen_width, screen_height):
    """
    Visualize the warehouse layout and the path.

    Args:
        grid (list): 2D list representing the warehouse layout.
        path (list): List of positions representing the path.
        screen_width (int): Width of the display window.
        screen_height (int): Height of the display window.

    Returns:
        np.ndarray: Scaled image of the warehouse layout with the path.
    """
    cell_size = 50  # Default size for each cell
    rows, cols = len(grid), len(grid[0])
    warehouse_image = np.zeros((rows * cell_size, cols * cell_size, 3), dtype=np.uint8)

    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            color = (0, 0, 0) if cell == 1 else (255, 255, 255)
            cv2.rectangle(
                warehouse_image,
                (j * cell_size, i * cell_size),
                ((j + 1) * cell_size, (i + 1) * cell_size),
                color,
                -1
            )

    # Highlight the path
    if path:
        for r, c in path:
            cv2.rectangle(
                warehouse_image,
                (c * cell_size, r * cell_size),
                ((c + 1) * cell_size, (r + 1) * cell_size),
                (0, 255, 0),
                -1
            )

    # Scale the image to fit the screen
    warehouse_image_scaled = cv2.resize(
        warehouse_image, 
        (screen_width, screen_height), 
        interpolation=cv2.INTER_AREA
    )
    
    return warehouse_image_scaled
