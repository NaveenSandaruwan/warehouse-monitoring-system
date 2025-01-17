import cv2
import time
from utils.grid_loader import load_grid
from utils.astar import astar
from utils.visualizer import visualize_warehouse
from utils.worker import get_worker_position

from utils.blockRoute import block_route_in_grid

# File path to the grid layout
file_path = 'simulation\grid_layout.txt'

# Load the grid layout
warehouse_layout = load_grid(file_path)
# print("Warehouse Layout:" , warehouse_layout)

# Define start and end points
start = (2, 1)  # Initial start position (row, col)
goal = (8, 5)   # Goal position (row, col)

while True:
    coordinates_to_block = [(0, 0), (3, 1)]  # Coordinates to block (set to 1)

    warehouse_layout_updated = block_route_in_grid(warehouse_layout, coordinates_to_block)
    # print("warehouse_layout:", warehouse_layout_updated)
    # Find the path using A* algorithm
    path = astar(warehouse_layout_updated, start, goal)
    print("Path:", path)

    # Visualize the warehouse and path
    screen_width = 800
    screen_height = 800
    warehouse_image_scaled = visualize_warehouse(
        warehouse_layout, path, screen_width, screen_height
    )

   
   

    # Display the updated image
    cv2.imshow("Warehouse Pathfinding", warehouse_image_scaled)

    # Wait for 100 ms and check for 'q' key press to exit
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

    # Update the worker's start position to simulate movementq
    worker_position = get_worker_position(path)
    start = worker_position

    # Simulate delay for worker movement
    time.sleep(1)
# Close all OpenCV windows
cv2.destroyAllWindows()
