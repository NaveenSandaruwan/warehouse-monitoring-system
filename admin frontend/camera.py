import multiprocessing
import time

class CameraSystem:
    def start(self):
        print("Starting camera system...")
        processes = []
        for i in range(4):  # Example: 4 camera streams
            p = multiprocessing.Process(target=self.camera_process, args=(i,))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    def camera_process(self, camera_id):
        while True:
            print(f"Camera {camera_id} is running...")
            time.sleep(1)  # Simulate camera processing