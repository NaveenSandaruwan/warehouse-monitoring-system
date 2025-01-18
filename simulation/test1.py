import cv2
import time
from utils.grid_loader import load_grid
from utils.astar import astar
from utils.visualizer import visualize_warehouse
from utils.worker import get_worker_position
from utils.getblockpositions import getblockposi
from utils.blockRoute import block_route_in_grid
from utils.getbocktime import gettime
from utils.checknearestblocks import check_nearest_blocks

def run_simulation(start, goal, camcoordinates):
    # File path to the grid layout
    file_path = 'simulation/grid_layout.txt'

    # Load the grid layout
    warehouse_layout = load_grid(file_path)

    finalpath=[]
    finalpath.append(start)

    type="small"
    mass =0
    if type == "small":
        mass = 1
    elif type == "medium":
        mass = 2
    elif type == "large":
        mass = 3

    # # Define start and end points
    # start = (2, 1)  # Initial start position (row, col)
    # goal = (13, 1)  # Goal position (row, col)
    # camcoordinates = [(1, 0), (13, 0)]

    last_length = None
    length_diff = 0

    while True:
        coordinates_to_block = getblockposi()  # Coordinates to block (set to 1)
        # list = check_nearest_blocks( start,coordinates_to_block)
        # removednearest = list[0]
        # coordinates_to_block_updated = list[1]
        warehouse_layout = load_grid(file_path)
        warehouse_layout_updated = block_route_in_grid(warehouse_layout, coordinates_to_block)
        # warehouse_layout_removenearest = block_route_in_grid(warehouse_layout, removednearest)
        # Find the path using A* algorithm
        path = astar(warehouse_layout_updated, start, goal)
        path_length = len(path)

        # pathwithoutblock = astar(warehouse_layout_removenearest, start, goal)
        # path_length_without_block = len(pathwithoutblock)

        # t=gettime(coordinates_to_block_updated,start)
        
        # if path_length - path_length_without_block > t:
        #    path = pathwithoutblock

        # if last_length is not None:
        #     length_diff = path_length - last_length + 1
        #     if length_diff <= 0:
        #         print(f"Path is shorter by {abs(length_diff)}")
        #     else:
        #         print(f"Path is longer by {length_diff}")

        # Visualize the warehouse and path
        screen_width = 800
        screen_height = 800
        warehouse_image_scaled = visualize_warehouse(
            warehouse_layout_updated, path, screen_width, screen_height, camcoordinates, coordinates_to_block
        )

        # Display the updated image
        cv2.imshow("Warehouse Pathfinding", warehouse_image_scaled)

        # Wait for 100 ms and check for 'q' key press to exit
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

        # Update the worker's start position to simulate movement
        worker_position = get_worker_position(path)
        

        if start == goal:
            print(f"work :{len(finalpath)*mass}")
            break
        else:
            finalpath.append(start)

        start = worker_position
        last_length = path_length

        # Simulate delay for worker movement
        time.sleep(1)

    # Close all OpenCV windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start = (2, 1)  # Initial start position (row, col)
    goal = (13, 1)  # Goal position (row, col)
    camcoordinates = [(1, 0), (13, 0)]
    run_simulation(start, goal, camcoordinates)
