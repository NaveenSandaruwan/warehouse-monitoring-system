def check_nearest_blocks(coord, coord_list):
    """
    Given a coordinate, find the nearest 4 coordinates (up, down, left, right),
    check if they are in the given coordinate list, and remove them if they are present.

    Args:
        coord (tuple): The coordinate to check around (row, col).
        coord_list (list): The list of coordinates to check against.

    Returns:
        list: The updated list of coordinates.
    """
    row = coord[0]
    col = coord[1]
    removedlist=[]
    nearest_coords = [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]

    for nc in nearest_coords:
        if nc in coord_list:
            coord_list.remove(nc)
            removedlist.append(nc)

    return [coord_list,removedlist]

# Example usage
if __name__ == "__main__":
    coord = (5, 3)
    coord_list = [(5, 5), (6, 5), (5, 4), (5, 6), (7, 7)]
    updated_list = check_nearest_blocks(coord, coord_list)
    print(updated_list[1])  # Output should be [(7, 7)]