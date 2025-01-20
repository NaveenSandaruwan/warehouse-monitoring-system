# from video_processor import process_video

# def camprocess():
#     # Input video path
#     input_video_path = "simulation\hallway.mp4"

#     # Output video path
#     output_video_path = "testing/output_video_with_rows.mp4"

#     # Log file path
#     log_file_path = "simulation\camprocess\counts.txt"

#     # Call the video processing function
#     process_video(input_video_path, output_video_path, log_file_path)

#     print(f"Processing completed. Output video saved at: {output_video_path}")
#     print(f"Log file saved at: {log_file_path}")

# if __name__ == "__main__":
#     camprocess()

from multiprocessing import Process
from video_processor import process_video
from multiprocessing import Process, Manager
import time

def real_time_file_writer(shared_data):
    """
    Continuously writes real-time updates from the shared data to a file.
    """
    output_file = "camprocess/real_time_updates.txt"
   

    while True:
        if shared_data:
            with open(output_file, 'w') as file:
                for cam_id, data in shared_data.items():
                    file.write(
                        
                        f"{data['blocked_coordinates']}\n"
                    )
        time.sleep(0.5)  # Adjust the interval as needed to control file writes

def camprocess(input_video_path, output_video_path, log_file_path, camcoordinates,rawcoordinates,shared_data,cam_id):
    # Call the video processing function
    process_video(input_video_path, output_video_path, log_file_path, camcoordinates,rawcoordinates,    shared_data,cam_id)
    print(f"Processing completed. Output video saved at: {output_video_path}")
    print(f"Log file saved at: {log_file_path}")

def start_blockcheking_camsystem():
    manager = Manager()
    shared_data = manager.dict()
    # Video 1 details
    video1 = {
        "input": "simulation\hallway.mp4",
        "output": "testing/output_video1.mp4",
        "log": "camprocess\log1.txt",
        "camcoordinates": [(13, 0)],
        "rawcoordinates": [(13, 6), (13, 4), (13, 2)],
        "cam_id": "cam1"
    }

    # Video 2 details
    video2 = {
        "input": "simulation/hallway.mp4",
        "output": "testing/output_video2.mp4",
        "log": "camprocess/log2.txt",
        "camcoordinates": [(1, 0)],
        "rawcoordinates": [(1, 6), (1, 4), (1, 2)],
        "cam_id": "cam2"
    }

    # Create processes for each video
    process1 = Process(target=camprocess, args=(video1["input"], video1["output"], video1["log"],video1["camcoordinates"],video1["rawcoordinates"],shared_data,video1["cam_id"]))
    process2 = Process(target=camprocess, args=(video2["input"], video2["output"], video2["log"],video2["camcoordinates"],video2["rawcoordinates"],shared_data,video2["cam_id"]))
     # Start the real-time file writer process
    printer_process = Process(target=real_time_file_writer, args=(shared_data,))  # Note the comma for a single-element tuple
    printer_process.start()

    # Start processes
    process1.start()
    process2.start()

    # Wait for processes to complete
    process1.join()
    process2.join()
    printer_process.terminate()

    print("Both videos processed successfully.")

if __name__ == "__main__":
    start_blockcheking_camsystem()    
