

# import cv2
# import time
# from utils.grid_loader import load_grid
# from utils.astar import astar
# from utils.visualizer import visualize_warehouse
# from utils.worker import get_worker_position
# from utils.getblockpositions import getblockposi
# from utils.blockRoute import block_route_in_grid



# # File path to the grid layout
# file_path = 'simulation\grid_layout.txt'

# # Load the grid layout
# warehouse_layout = load_grid(file_path)
# # print("Warehouse Layout:" , warehouse_layout)

# # Define start and end points
# start = (2, 1)  # Initial start position (row, col)
# goal = (13, 1)   # Goal position (row, col)
# camcoordinates = [(1, 0), (13, 0)]

# lastlenght = None
# length_diff = 0
# while True:
#     coordinates_to_block = getblockposi()  # Coordinates to block (set to 1)
#     warehouse_layout = load_grid(file_path)
#     warehouse_layout_updated = block_route_in_grid(warehouse_layout, coordinates_to_block)
#     # print("warehouse_layout:", warehouse_layout_updated)
#     # Find the path using A* algorithm
#     path = astar(warehouse_layout_updated, start, goal)
#     # print("Path:", path)
#     pathlength = len(path)

#     if lastlenght:
#         length_diff = pathlength - lastlenght+1
#         if length_diff <= 0:
#             print(f"Path is shorter  {length_diff}")
#         else:
#             print(f"Path is longer  {length_diff}")

#     # Visualize the warehouse and path
#     screen_width = 800
#     screen_height = 800
#     warehouse_image_scaled = visualize_warehouse(
#         warehouse_layout_updated, path, screen_width, screen_height,camcoordinates,coordinates_to_block
#     )

   
   

#     # Display the updated image
#     cv2.imshow("Warehouse Pathfinding", warehouse_image_scaled)

#     # Wait for 100 ms and check for 'q' key press to exit
#     if cv2.waitKey(100) & 0xFF == ord('q'):
#         break

#     # Update the worker's start position to simulate movementq
#     worker_position = get_worker_position(path)
#     start = worker_position


#     lastlenght=pathlength
#     # Simulate delay for worker movement
#     time.sleep(1)
# # Close all OpenCV windows
# cv2.destroyAllWindows()
