import cv2
import time
from utils.grid_loader import load_grid
from utils.astar import astar
from utils.visualizer import visualize_warehouse
from utils.worker import get_worker_position

# File path to the grid layout
file_path = 'simulation\grid_layout.txt'

# Load the grid layout
warehouse_layout = load_grid(file_path)

# Define start and end points
start = (1, 1)  # Initial start position (row, col)
goal = (3, 5)   # Goal position (row, col)

while True:
    # Find the path using A* algorithm
    path = astar(warehouse_layout, start, goal)
    print("Path:", path)

    # Visualize the warehouse and path
    screen_width = 800
    screen_height = 600
    warehouse_image_scaled = visualize_warehouse(
        warehouse_layout, path, screen_width, screen_height
    )

    # Draw the worker's position as a red circle
    worker_position = start
    worker_x = worker_position[1] * 25+ 25  # X coordinate (column)
    worker_y = worker_position[0] * 25 + 0  # Y coordinate (row)
    cv2.circle(warehouse_image_scaled, (worker_x, worker_y), 10, (0, 0, 255), -1)

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
