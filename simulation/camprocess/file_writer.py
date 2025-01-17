def write_log(log_file_path, frame_number, row_counts):
    """
    Writes the row-wise person count for each frame to a log file, maintaining only 4 lines.

    Args:
        log_file_path (str): Path to the log file.
        frame_number (int): Current frame number.
        row_counts (list): List of person counts for each row.
    """
    with open(log_file_path, 'w') as file:
        file.write(f"Frame: {frame_number}\n")
        for i, count in enumerate(row_counts):
            file.write(f"  Row {i + 1}: {count} persons\n")