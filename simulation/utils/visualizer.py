import cv2
import numpy as np
from utils.getOthersPositions import othersPositions
def visualize_warehouse(grid, path, screen_width, screen_height,camcoordinates):
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
    cv2.circle(warehouse_image, (path[0][1]* cell_size+25, path[0][0]* cell_size+25), 20, (0, 0, 255), -1)
     # get others positions and draw them as blue circles
    others = othersPositions()
    for other in others:
        other_x = other[1] * cell_size + 25  # X coordinate (column)
        other_y = other[0] * cell_size + 25  # Y coordinate (row)
        cv2.circle(warehouse_image, (other_x, other_y), 20, (255, 0, 0), -1)

    # Draw the camera positions as red circles
    for cam in camcoordinates:
        cam_x = cam[1] * cell_size + 25
        cam_y = cam[0] * cell_size + 25
        cv2.circle(warehouse_image, (cam_x, cam_y), 20, (0, 255, 255), -1)    

    # Scale the image to fit the screen
    warehouse_image_scaled = cv2.resize(
        warehouse_image, 
        (screen_width, screen_height), 
        interpolation=cv2.INTER_AREA
    )
    
    return warehouse_image_scaled
