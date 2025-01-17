import ast

def block_route_in_grid(grid, coordinates):
    """
    Blocks the route at the given coordinates by setting the grid value to 1 in the provided grid string.
    
    Args:
        grid_str (str): The string representation of the 2D grid.
        coordinates (list of tuples): A list of coordinates (x, y) to block (set to 1).
        
    Returns:
        str: The updated grid as a string.
    """
    try:
       
        # Convert the string representation of the grid into a 2D list
        # grid = ast.literal_eval(grid_str)
        
        # Loop through each coordinate and set the corresponding position in the grid to 1
        for x, y in coordinates:
            if 0 <= x < len(grid) and 0 <= y < len(grid[0]): 
                # print("grid[x][y]:", grid[x][y])
                 # Check if coordinates are within bounds
                grid[x][y] = 1  # Block the route (set the position to 1)
                # print("grid[x][y]:", grid[x][y])
        # Convert the updated grid back to a string
        updated_grid_str = grid
        # print("updated_grid_str:", updated_grid_str)
        
        return updated_grid_str
    
    except Exception as e:
        return f"An error occurred: {e}"

# # Example usage:
# grid_str = """
# [
#     [0, 1, 0, 0],
#     [0, 0, 0, 1],
#     [1, 0, 1, 0],
#     [0, 0, 0, 0]
# ]
# """  # Your grid as a string

# coordinates_to_block = [(0, 2), (3, 1)]  # Coordinates to block (set to 1)

# updated_grid = block_route_in_grid(grid_str, coordinates_to_block)

# # Output the updated grid
# print("Updated Grid:")
# print(updated_grid)
