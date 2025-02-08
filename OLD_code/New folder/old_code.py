import cv2
from ultralytics import YOLO
from deepface import DeepFace
from norfair import Detection, Tracker
import mediapipe as mp
import torch
from torchvision import transforms
import numpy as np
import os
import time
from PIL import Image

# Step 1: Download the YouTube video
def download_with_ytdlp(video_url, output_filename="youtube_video.mp4"):
    os.system(f'yt-dlp -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4" {video_url} -o "{output_filename}"')
    if not os.path.exists(output_filename):
        print(f"Error: File {output_filename} was not created. Check yt-dlp logs.")
        return None
    print(f"Downloaded video as: {output_filename}, Size: {os.path.getsize(output_filename)} bytes")
    return output_filename

# Provide the YouTube video URL
#video_url = 'https://www.youtube.com/watch?v=09R8_2nJtjg'  # Replace with your video URL
#video_path = download_with_ytdlp(video_url)


video_path = 'people--walking----background-video.mp4'  # I see this file in your directory

# Check if the video exists
if not os.path.exists(video_path):
    print(f"Error: Video file not found at {video_path}.")
    exit()

# Step 2: Initialize YOLO model
model = YOLO('yolov8x.pt')  # YOLOv8 for object detection

# Initialize Norfair Tracker
tracker = Tracker(distance_function="euclidean", distance_threshold=30)  # Tracks detected objects

# Initialize Mediapipe Pose Estimation
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize a Lightweight Action Recognition Model
from torchvision.models.video import r3d_18

action_model = r3d_18(pretrained=True)  # Lightweight ResNet3D model for video classification
action_model.eval()

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load known faces
def load_known_faces(folder='known_faces'):
    known_faces = {}
    for filename in os.listdir(folder):
        if filename.endswith(('.jpg', '.png')):
            known_faces[filename] = os.path.join(folder, filename)
    return known_faces

known_faces = load_known_faces()
print(f"Loaded {len(known_faces)} known faces.")

# Function to convert YOLO detections to Norfair detections
def yolo_to_norfair(detections):
    norfair_detections = []
    for box in detections:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        score = box.conf[0].item() if isinstance(box.conf, torch.Tensor) else box.conf
        norfair_detections.append(Detection(points=center, scores=np.array([score])))
    return norfair_detections

# Optimized function for face recognition
def perform_face_recognition(face_region, temp_face_path, known_faces):
    name = "Unknown"
    cv2.imwrite(temp_face_path, face_region)
    for known_name, known_face_path in known_faces.items():
        try:
            result = DeepFace.verify(img1_path=temp_face_path, img2_path=known_face_path, model_name="VGG-Face")
            if result['verified']:
                name = os.path.splitext(known_name)[0]
                break
        except Exception as e:
            print(f"Error verifying face: {e}")
    return name

# Function to handle clustering threshold and new entries
def should_do_face_recognition(tracked_objects, obj_id, last_recognition_time):
    current_time = time.time()
    if obj_id not in last_recognition_time:
        # New person, perform recognition
        last_recognition_time[obj_id] = current_time
        return True
    elif current_time - last_recognition_time[obj_id] > 3:  # Check if last recognition was >3 seconds ago
        # Perform recognition again
        last_recognition_time[obj_id] = current_time
        return True
    return False

# Function to recognize actions using ResNet3D
def recognize_action_with_r3d(frames):
    # Preprocess and stack frames
    processed_frames = torch.stack([transform(Image.fromarray(frame)) for frame in frames])
    batch = processed_frames.unsqueeze(0)  # Shape: [1, T, C, H, W]
    batch = batch.permute(0, 2, 1, 3, 4)  # Convert to [N, C, T, H, W] for ResNet3D

    with torch.no_grad():
        outputs = action_model(batch)
        predicted_class = torch.argmax(outputs, dim=1).item()
    action_labels = {0: "Walking", 1: "Lifting", 2: "Sitting"}  # Replace with your dataset's action labels
    return action_labels.get(predicted_class, "Unknown")

# Step 3: Open the video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open the video at {video_path}.")
    exit()

# Output file configuration
output_path = 'output_identified.mp4'  # Save as MP4
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Frame width
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Frame height

print(f"Video properties - FPS: {fps}, Width: {frame_width}, Height: {frame_height}")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Margin for better face cropping
margin = 20

# Track last face recognition timestamps
last_recognition_time = {}

# Frame buffer for action recognition
buffered_frames = []
'''
# Process video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("No more frames to process. End of video.")
        break

    results = model(frame, conf=0.6)
    detections = results[0].boxes
    norfair_detections = yolo_to_norfair(detections)
    tracked_objects = tracker.update(detections=norfair_detections)

    # Buffer frames for action recognition
    buffered_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if len(buffered_frames) == 16:  # Process every 16 frames for action recognition
        action = recognize_action_with_r3d(buffered_frames)
        buffered_frames = []

    for obj in tracked_objects:
        if obj.estimate is not None and len(obj.estimate) >= 2:
            x, y = map(int, obj.estimate[:2])  # Take only the first two elements for x and y
            id_text = f"ID: {obj.id}"

            # Extract bounding box for the tracked person
            for box in detections:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Perform face recognition only for people
                face_region = frame[
                    max(0, y1 - margin):min(frame.shape[0], y2 + margin),
                    max(0, x1 - margin):min(frame.shape[1], x2 + margin)
                ]
                if should_do_face_recognition(tracked_objects, obj.id, last_recognition_time):
                    temp_face_path = f"temp_face_{obj.id}.jpg"
                    name = perform_face_recognition(face_region, temp_face_path, known_faces)
                else:
                    name = "Recognized Earlier"

                # Draw bounding box, ID, name, and action
                color = (0, 255, 0) if name != "Unknown" else (255, 0, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, id_text, (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, name, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, f"Action: {action}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        else:
            print(f"Warning: obj.estimate has an unexpected shape or is None: {obj.estimate}")

    out.write(frame)
'''

# Inside the frame processing loop

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("No more frames to process. End of video.")
        break

    # Perform YOLO detection
    results = model(frame, conf=0.6)
    detections = results[0].boxes

    # Convert YOLO detections to Norfair detections
    norfair_detections = yolo_to_norfair(detections)
    tracked_objects = tracker.update(detections=norfair_detections)

    # Buffer frames for action recognition
    buffered_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if len(buffered_frames) == 16:  # Process every 16 frames for action recognition
        action = recognize_action_with_r3d(buffered_frames)
        buffered_frames = []

    for obj in tracked_objects:
        if obj.estimate is not None and len(obj.estimate) >= 2:
            # Draw tracked object ID
            x, y = map(int, obj.estimate[:2])  # Coordinates of tracked object
            id_text = f"ID: {obj.id}"

            # Match YOLO bounding box to the tracked object
            for box in detections:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Perform face recognition if needed
                face_region = frame[
                    max(0, y1 - margin):min(frame.shape[0], y2 + margin),
                    max(0, x1 - margin):min(frame.shape[1], x2 + margin)
                ]
                if should_do_face_recognition(tracked_objects, obj.id, last_recognition_time):
                    temp_face_path = f"temp_face_{obj.id}.jpg"
                    name = perform_face_recognition(face_region, temp_face_path, known_faces)
                else:
                    name = "Recognized Earlier"

                # Draw bounding box, ID, name, and action
                color = (0, 255, 0) if name != "Unknown" else (255, 0, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, id_text, (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, name, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, f"Action: {action}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Write the frame to the output video
    out.write(frame)



# Release resources
cap.release()
out.release()
print("Processing complete. Output saved to:", output_path)
