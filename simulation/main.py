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
from utils.storework import send_work_data
from utils.updateposition import update_user_location
from utils.getworkerstartposition import get_user_location
from datetime import datetime

def runmap(start, goal, camcoordinates,type="small",wid=111):
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
    elif type == "xlarge":
        mass = 4    

    # # Define start and end points
    # start = (2, 1)  # Initial start position (row, col)
    # goal = (13, 1)  # Goal position (row, col)
    # camcoordinates = [(1, 0), (13, 0)]

    last_length = None
    length_diff = 0

    while True:
        # update_user_location(wid, start[0], start[1])

        coordinates_to_block = getblockposi()  # Coordinates to block (set to 1)
        copy = coordinates_to_block.copy()
        # print(f"coordinates_to_block 1   {coordinates_to_block}")
        list = check_nearest_blocks( start,coordinates_to_block)
        removednearest = list[1]
        coordinates_to_block_updated = list[0]
        # print(f"coordinates_to_block_up  {coordinates_to_block_updated}")
        # print(f"coordinates_to_block     {copy}")


        warehouse_layout = load_grid(file_path)
        warehouse_layout_updated = block_route_in_grid(warehouse_layout, copy)
        warehouse_layout = load_grid(file_path)
        warehouse_layout_removenearest = block_route_in_grid(warehouse_layout, coordinates_to_block_updated)
        # Find the path using A* algorithm
        path = astar(warehouse_layout_updated, start, goal)
        path_length = len(path)

        pathwithoutblock = astar(warehouse_layout_removenearest, start, goal)
        path_length_without_block = len(pathwithoutblock)

        t=gettime(removednearest,start)
        # print(f"t: {t}")
        if path_length - path_length_without_block > t:
           path = pathwithoutblock

        # print(f"path diff: {path_length - path_length_without_block}")
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
            warehouse_layout_updated, path, screen_width, screen_height, camcoordinates, copy
        )

        # Display the updated image
        cv2.imshow("Warehouse Pathfinding", warehouse_image_scaled)

        # Wait for 100 ms and check for 'q' key press to exit
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

        # Update the worker's start position to simulate movement
        worker_position = get_worker_position(path)
        

        if start == goal:
            current_date = datetime.now().strftime("%Y-%m-%d")
            work_done = len(finalpath)*mass
            response = send_work_data(wid, current_date, work_done)
            update_user_location(wid, start[0], start[1])
            print(response)
            print(f"work :{work_done}")
            break
        else:
            finalpath.append(start)

        start = worker_position
        last_length = path_length

        # Simulate delay for worker movement
        time.sleep(0.5)

    # Close all OpenCV windows
    cv2.destroyAllWindows()

def run_simulation(start, goal, camcoordinates,type="small",wid=111,n=1):
    for i in range(n):
        w_start= get_user_location(wid)
        runmap(w_start, start, camcoordinates,"small",wid)
        runmap(start, goal, camcoordinates,type,wid)




if __name__ == "__main__":
    start = (2, 1)  # Initial start position (row, col)
    goal = (13, 1)  # Goal position (row, col)
    camcoordinates = [(1, 0), (13, 0)]
    run_simulation(start, goal, camcoordinates)
