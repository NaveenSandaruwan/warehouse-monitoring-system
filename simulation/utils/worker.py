# utils/worker.py

def update_worker_position(path, worker_position_index):
    """
    Update the worker's position along the path.
    
    Args:
        path (list): List of coordinates representing the path.
        worker_position_index (int): The current index of the worker's position on the path.
    
    Returns:
        tuple: The updated position of the worker.
    """
    if worker_position_index < len(path):
        return path[worker_position_index]
    return None

def get_worker_position(path):
    """
    Return the worker's current position from the path.
    
    Args:
        worker_position_index (int): The current index of the worker's position on the path.
    
    Returns:
        tuple: The worker's current position (row, col).

    """
    if len(path) > 1:
      return path[1]
    return path[0]
