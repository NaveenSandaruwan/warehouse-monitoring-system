def load_grid(file_path):
    """
    Load the grid layout from a file.
    The file should contain a 2D list representing the grid.

    Args:
        file_path (str): Path to the file containing the grid layout.

    Returns:
        list: 2D list representing the grid layout.
    """
    with open(file_path, 'r') as file:
        grid = eval(file.read())  # Alternatively, use json.loads() if the file contains valid JSON
    return grid
