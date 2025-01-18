# def write_log(log_file_path, frame_number, row_counts,camcoordinates,rawcoordinates):
#     """
#     Writes the row-wise person count for each frame to a log file, maintaining only 4 lines.

#     Args:
#         log_file_path (str): Path to the log file.
#         frame_number (int): Current frame number.
#         row_counts (list): List of person counts for each row.
#     """
#     with open(log_file_path, 'w') as file:
#         file.write(f"Frame: {frame_number} \n")
#         file.write(f"Camera Coordinates: {camcoordinates} \n")
#         for i, count in enumerate(row_counts):
#             file.write(f"  Row {i + 1}: {count} persons\n")

def determine_blocked_coordinates(row_counts, raw_coordinates):
    """
    Determines which coordinates to block based on row counts.

    Args:
        row_counts (list): List of person counts for each row.
        raw_coordinates (list): List of raw coordinates corresponding to rows.

    Returns:
        list: Coordinates to block.
    """
    coordinates_to_block = []
    for i, count in enumerate(row_counts):
        if count >= 3:  # Block if row count is 3 or more
            coordinates_to_block.append(raw_coordinates[i])
    return coordinates_to_block


def write_log(log_file_path, frame_number, row_counts, cam_coordinates, raw_coordinates, shared_data, cam_id):
    """
    Writes the blocked coordinates for each frame to a log file.

    Args:
        log_file_path (str): Path to the log file.
        frame_number (int): Current frame number.
        row_counts (list): List of person counts for each row.
        cam_coordinates (list): Camera-specific coordinates (optional, for reference).
        raw_coordinates (list): List of raw coordinates corresponding to rows.
    """
    # Determine the blocked coordinates
    blocked_coordinates = determine_blocked_coordinates(row_counts, raw_coordinates)
    shared_data[cam_id] = {"frame": frame_number, "blocked_coordinates": blocked_coordinates}
    # Write only the blocked coordinates to the file
    with open(log_file_path, 'w') as file:
        # file.write(f"Frame: {frame_number}\n")
        file.write(str(blocked_coordinates))
        # file.write(f"Camera Coordinates: {cam_coordinates}\n")