import cv2
import numpy as np
import time
from threading import Thread, Lock

# Load pre-trained model and configuration file for person detection
net = cv2.dnn.readNetFromCaffe('E:/VS CODE/warehousing/idle-detection-2/idle-detection-2/deploy.prototxt', 
                               'E:/VS CODE/warehousing/idle-detection-2/idle-detection-2/mobilenet_iter_73000.caffemodel')

class PersonTracker:
    def __init__(self, video_path):
        self.video_path = video_path
        self.person_tracks = {}
        self.lock = Lock()
        self.net = cv2.dnn.readNetFromCaffe(
            'E:/VS CODE/warehousing/idle-detection-2/idle-detection-2/deploy.prototxt',
            'E:/VS CODE/warehousing/idle-detection-2/idle-detection-2/mobilenet_iter_73000.caffemodel'
        )
        self.matrix =[[(1,1), (1,2), (1,4)]]

    def update_db(self):
        while True:
            with self.lock:
                print("Updating database with person tracks...")
                # Add code to update the database with the person tracks
                print("Database updated.")
                for pid, track in self.person_tracks.items():
                    posi = track['positions']
                    idle_time = time.time() - track['idle_start_time'] if track['idle_start_time'] else 0
                    print(f"Person ID {pid}: Position {posi}, Idle Time {idle_time}")

                print(self.person_tracks)
            time.sleep(5)  # Update the database every 5 seconds

    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        person_id = 0

        # Desired dimensions
        desired_width = 640
        desired_height = 480

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize the frame to the desired dimensions
            frame = cv2.resize(frame, (desired_width, desired_height))
            h, w = frame.shape[:2]

            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
            self.net.setInput(blob)
            detections = self.net.forward()

            current_positions = []

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.7:  # Lowered confidence threshold
                    idx = int(detections[0, 0, i, 1])
                    if idx == 15:  # Class ID for person in MobileNet-SSD
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        current_positions.append((startX, startY, endX, endY))
                        # print(f"Detected person with confidence {confidence}: {startX, startY, endX, endY}")

            # print(f"Current positions: {current_positions}")
            # print(f"Person tracks before update: {self.person_tracks.items()}")

            # Update tracks
            with self.lock:
                for pos in current_positions:
                    matched = False
                    for pid, track in self.person_tracks.items():
                        if np.linalg.norm(np.array(pos) - np.array(track['position'])) < 100:  # Threshold for matching
                            track['position'] = pos
                            track['frames'] += 1
                            track['positions'].append(pos)
                            if len(track['positions']) > 10:  # Keep the last 10 positions
                                track['positions'].pop(0)
                            if np.linalg.norm(np.array(pos) - np.array(track['positions'][0])) < 20: # Threshold for idle
                                if track['idle_frames'] == 0:
                                    track['idle_start_time'] = time.time()  # Record the start time of idle
                                track['idle_frames'] += 1
                            else:
                                track["total_idle_time"] += track['idle_time']
                                track['idle_frames'] = 0
                                track['idle_start_time'] = None  # Reset idle start time
                            track['prev_position'] = pos
                            matched = True
                            break
                    if not matched:
                        self.person_tracks[person_id] = {
                            'position': pos,
                            'prev_position': pos,
                            'frames': 1,
                            'idle_frames': 0,
                            'idle_start_time': None,  # Initialize idle start time
                            'missed_frames': 0,  # Add a counter for missed frames
                            'positions': [pos],  # Store the last 10 positions
                            'section': None,  # Initialize section
                            'created_at': time.time(),
                            'idle_time': 0,
                            'total_idle_time': 0
                        }
                        person_id += 1

                # Increment missed frames for all tracks
                for pid, track in self.person_tracks.items():
                    if track['position'] not in current_positions:
                        track['missed_frames'] += 1
                    else:
                        track['missed_frames'] = 0

                # print(f"Person tracks after update: {self.person_tracks.items()}")

                # Remove old tracks that have not been updated for a certain number of frames
                self.person_tracks = {pid: track for pid, track in self.person_tracks.items() if track['missed_frames'] < 20}

            # Draw bounding boxes and labels
            for pid, track in self.person_tracks.items():
                print(len(self.person_tracks))
                (startX, startY, endX, endY) = track['position']
                # Ensure coordinates are within bounds
                startX = max(0, min(startX, w - 1))
                startY = max(0, min(startY, h - 1))
                endX = max(0, min(endX, w - 1))
                endY = max(0, min(endY, h - 1))

                if track['idle_start_time']:
                    idle_time = time.time() - track['idle_start_time']
                    track['idle_time'] = idle_time
                else:
                    idle_time = 0
                    track['idle_time'] = idle_time

                label = f"ID {pid}: {'Idle' if track['idle_frames'] > 10 else 'Not Idle'} ({idle_time:.2f}s)"
                # print(f"Drawing box for ID {pid}: {startX, startY, endX, endY} with label {label}")
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw horizontal and vertical lines to divide the screen into sections based on the matrix
            rows = len(self.matrix)
            cols = len(self.matrix[0])
            for i in range(1, rows):
                cv2.line(frame, (0, i * h // rows), (w, i * h // rows), (255, 0, 0), 2)
            for j in range(1, cols):
                cv2.line(frame, (j * w // cols, 0), (j * w // cols, h), (255, 0, 0), 2)

            # Print numbers from the matrix in the corresponding sections if a person is detected
            for i in range(rows):
                for j in range(cols):
                    section_startX = j * w // cols
                    section_endX = (j + 1) * w // cols
                    section_startY = i * h // rows
                    section_endY = (i + 1) * h // rows
                    for (startX, startY, endX, endY) in current_positions:
                        centerX = (startX + endX) // 2
                        centerY = (startY + endY) // 2
                        if section_startX <= centerX < section_endX and section_startY <= centerY < section_endY:
                            number = self.matrix[i][j]
                            cv2.putText(frame, str(number), (section_startX + 10, section_startY + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                            # Update the section in person_tracks
                            for pid, track in self.person_tracks.items():
                                if track['position'] == (startX, startY, endX, endY):
                                    track['section'] = number

            # Display the frame with bounding boxes, labels, and grid lines
            cv2.imshow("Frame", frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("All Persons tracked : ", self.person_tracks)

    def run(self):
        video_thread = Thread(target=self.process_video)
        db_thread = Thread(target=self.update_db)

        video_thread.start()
        db_thread.start()

        video_thread.join()
        db_thread.join()

if __name__ == "__main__":
    tracker = PersonTracker(r"idle-detection\jump.mp4")
    tracker.run()