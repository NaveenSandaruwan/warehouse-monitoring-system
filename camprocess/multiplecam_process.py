from multiprocessing import Process, Manager
import time
from test_video import process_video


class CameraSystem:
    def __init__(self):
        """
        Initialize the CameraSystem with three processes.
        """
        self.manager = Manager()
        self.shared_data = self.manager.dict()
        
        # Video details
        self.video1 = {
            "input": "simulation/hallway.mp4",
            "output": "testing/output_video1.mp4",
            "log": "camprocess/log1.txt",
            "camcoordinates": [(13, 0)],
            "rawcoordinates": [(13, 6), (13, 4), (13, 2)],
            "cam_id": "cam1",
            "use_camera": False
        }
        self.video2 = {
            "input": "",
            "output": "testing/output_video2.mp4",
            "log": "camprocess/log2.txt",
            "camcoordinates": [(1, 0)],
            "rawcoordinates": [(1, 6), (1, 4), (1, 2)],
            "cam_id": "cam2",
            "use_camera": True
        }
        
        # Processes
        self.process1 = Process(target=self.camprocess, args=(
            self.video1["input"], self.video1["output"], self.video1["log"],
            self.video1["camcoordinates"], self.video1["rawcoordinates"],
            self.shared_data, self.video1["cam_id"], self.video1["use_camera"]
        ))
        self.process2 = Process(target=self.camprocess, args=(
            self.video2["input"], self.video2["output"], self.video2["log"],
            self.video2["camcoordinates"], self.video2["rawcoordinates"],
            self.shared_data, self.video2["cam_id"], self.video2["use_camera"]
        ))
        self.printer_process = Process(target=self.real_time_file_writer, args=(self.shared_data,))

    @staticmethod
    def real_time_file_writer(shared_data):
        """
        Continuously writes real-time updates from the shared data to a file.
        """
        print("Real-time file writer started.")
        output_file = "camprocess/real_time_updates.txt"
        while True:
            if shared_data:
                with open(output_file, 'w') as file:
                    for cam_id, data in shared_data.items():
                        file.write(f"{data['blocked_coordinates']}\n")
            time.sleep(0.5)  # Adjust the interval as needed to control file writes

    @staticmethod
    def camprocess(input_video_path, output_video_path, log_file_path, camcoordinates, rawcoordinates, shared_data, cam_id, use_camera):
        """
        Process a video and save the output.
        """
        process_video(input_video_path, output_video_path, log_file_path, camcoordinates, rawcoordinates, shared_data, cam_id, use_camera)
        print(f"Processing completed. Output video saved at: {output_video_path}")
        print(f"Log file saved at: {log_file_path}")

    def start_processes(self):
        """
        Start all processes.
        """
        print("Starting processes...")
        self.process1.start()
        self.process2.start()
        self.printer_process.start()
        print("Processes started.")

    def terminate_processes(self):
        """
        Terminate all processes.
        """
        print("Terminating processes...")
        for process in [self.process1, self.process2, self.printer_process]:
            if process.is_alive():
                process.terminate()
                process.join()
        print("All processes terminated.")


def startCameraSystem():
    camera_system = CameraSystem()
    camera_system.start_processes()
    return camera_system

def stopCameraSystem(camera_system):
    camera_system.terminate_processes()

# Example Usage
if __name__ == "__main__":
    
    try:
        c = startCameraSystem()
        # Simulate runtime
        time.sleep(20)  # Adjust as needed
    finally:
        stopCameraSystem(c)