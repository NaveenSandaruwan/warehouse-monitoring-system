from video_processor import process_video

def camprocess():
    # Input video path
    input_video_path = "simulation\hallway.mp4"

    # Output video path
    output_video_path = "testing/output_video_with_rows.mp4"

    # Log file path
    log_file_path = "simulation\camprocess\counts.txt"

    # Call the video processing function
    process_video(input_video_path, output_video_path, log_file_path)

    print(f"Processing completed. Output video saved at: {output_video_path}")
    print(f"Log file saved at: {log_file_path}")

if __name__ == "__main__":
    camprocess()