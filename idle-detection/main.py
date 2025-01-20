import cv2
import numpy as np

# Load pre-trained model and configuration file for person detection
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    person_tracks = {}
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
        net.setInput(blob)
        detections = net.forward()

        current_positions = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.4:  # Lowered confidence threshold
                idx = int(detections[0, 0, i, 1])
                if idx == 15:  # Class ID for person in MobileNet-SSD
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    current_positions.append((startX, startY, endX, endY))
                    print(f"Detected person with confidence {confidence}: {startX, startY, endX, endY}")

        print(f"Current positions: {current_positions}")
        print(f"Person tracks before update: {person_tracks.items()}")

        # Update tracks
        for pos in current_positions:
            matched = False
            for pid, track in person_tracks.items():
                if np.linalg.norm(np.array(pos) - np.array(track['position'])) < 50:  # Threshold for matching
                    track['position'] = pos
                    track['frames'] += 1
                    track['positions'].append(pos)
                    if len(track['positions']) > 10:  # Keep the last 10 positions
                        track['positions'].pop(0)
                    if np.linalg.norm(np.array(pos) - np.array(track['positions'][0])) < 5: # Threshold for idle
                        track['idle_frames'] += 1
                    else:
                        track['idle_frames'] = 0
                    track['prev_position'] = pos
                    matched = True
                    break
            if not matched:
                person_tracks[person_id] = {
                    'position': pos,
                    'prev_position': pos,
                    'frames': 1,
                    'idle_frames': 0,
                    'missed_frames': 0,  # Add a counter for missed frames
                    'positions': [pos]  # Store the last 10 positions
                }
                person_id += 1

        # Increment missed frames for all tracks
        for pid, track in person_tracks.items():
            if track['position'] not in current_positions:
                track['missed_frames'] += 1
            else:
                track['missed_frames'] = 0

        print(f"Person tracks after update: {person_tracks.items()}")

        # Remove old tracks that have not been updated for a certain number of frames
        person_tracks = {pid: track for pid, track in person_tracks.items() if track['missed_frames'] < 5}

        # Draw bounding boxes and labels
        for pid, track in person_tracks.items():
            (startX, startY, endX, endY) = track['position']
            # Ensure coordinates are within bounds
            startX = max(0, min(startX, w - 1))
            startY = max(0, min(startY, h - 1))
            endX = max(0, min(endX, w - 1))
            endY = max(0, min(endY, h - 1))
            label = f"ID {pid}: {'Idle' if track['idle_frames'] > 10 else 'Not Idle'}"
            print(f"Drawing box for ID {pid}: {startX, startY, endX, endY} with label {label}")
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame with bounding boxes and labels
        cv2.imshow("Frame", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video("girl.mp4")