import heapq
import numpy as np
import cv2
import json

# Read the grid layout from a file
def load_grid(file_path):
    with open(file_path, 'r') as file:
        grid = eval(file.read())  # Alternatively, use json.loads() if the file contains valid JSON
    return grid

# Load the grid
file_path = 'testing\simulation\grid_layout.txt'
warehouse_layout = load_grid(file_path)

# Define start and end points
start = (1, 1)  # (row, col)
goal = (3, 5)

# A* Algorithm
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan distance

def astar(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    
    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
        
        neighbors = [
            (current[0] + 1, current[1]), (current[0] - 1, current[1]),
            (current[0], current[1] + 1), (current[0], current[1] - 1)
        ]
        
        for neighbor in neighbors:
            r, c = neighbor
            if 0 <= r < rows and 0 <= c < cols and grid[r][c] == 0:
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    return None  # No path found

# Find the path
path = astar(warehouse_layout, start, goal)
print("Path:", path)

# Visualize the result
cell_size = 50  # Default size for each cell
rows, cols = len(warehouse_layout), len(warehouse_layout[0])
warehouse_image = np.zeros((rows * cell_size, cols * cell_size, 3), dtype=np.uint8)

for i, row in enumerate(warehouse_layout):
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
screen_width = 800  # Adjust as per your screen resolution
screen_height = 600  # Adjust as per your screen resolution
warehouse_image_scaled = cv2.resize(
    warehouse_image, 
    (screen_width, screen_height), 
    interpolation=cv2.INTER_AREA
)

# Display the warehouse with the path
cv2.imshow("Warehouse Pathfinding", warehouse_image_scaled)
cv2.waitKey(0)
cv2.destroyAllWindows()
