import cv2
import numpy as np
from utils.getOthersPositions import othersPositions
def drawdirection(direction,warehouse_image,cell_size,path):

          # Draw the worker's position as a red circle
    cv2.circle(warehouse_image, (path[0][1] * cell_size + 25, path[0][0] * cell_size + 25), 20, (0, 0, 255), -1)
    
    # Determine the direction and draw an arrow
     # Replace this with the actual direction value
    start_point = (path[0][1] * cell_size + 25, path[0][0] * cell_size + 25)
    end_point = start_point
    
    if direction == 'up':
        end_point = (start_point[0], start_point[1] - 50)
    elif direction == 'down':
        end_point = (start_point[0], start_point[1] + 50)
    elif direction == 'left':
        end_point = (start_point[0] - 50, start_point[1])
    elif direction == 'right':
        end_point = (start_point[0] + 50, start_point[1])
    
    if direction != 'stay':
        cv2.arrowedLine(warehouse_image, start_point, end_point, (255, 0, 0), 5)
    

def getdirection(start, end):
    """
    Get the direction of the movement based on the start and end positions.

    Args:
        start (tuple): Starting position as a (row, col) tuple.
        end (tuple): Ending position as a (row, col) tuple.

    Returns:
        str: Direction of movement ('up', 'down', 'left', 'right').
    """
    row_diff = end[0] - start[0]
    col_diff = end[1] - start[1]

    if row_diff == 1:
        return 'down'
    elif row_diff == -1:
        return 'up'
    elif col_diff == 1:
        return 'right'
    elif col_diff == -1:
        return 'left'
    else:
        return 'stay'

def visualize_warehouse(grid, path, screen_width, screen_height,camcoordinates,coordinates_to_block):
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

    # get the direction of the movement
    if len(path) > 1:
        direction = getdirection(path[0], path[1])
    else:
        direction = 'stay'    
    print("Direction:", direction)

    # Draw the grid
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
    # Highlight the blocked coordinates
    for coord in coordinates_to_block:
        cv2.rectangle(
            warehouse_image,
            (coord[1] * cell_size, coord[0] * cell_size),
            ((coord[1] + 1) * cell_size, (coord[0] + 1) * cell_size),
            (0, 165, 255),
            -1
        )

    
    drawdirection(direction,warehouse_image,cell_size,path)  

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
