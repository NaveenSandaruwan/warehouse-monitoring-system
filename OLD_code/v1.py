import cv2
import torch
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image

# Load YOLO model for person detection
yolo_model = YOLO("yolov8n.pt")  # You can use yolov8s.pt or yolov8m.pt for better accuracy

# Load DeepSORT tracker
tracker = DeepSort(max_age=30)  # Adjust max_age for tracking duration

# Load pre-trained action recognition model (e.g., a 3D CNN)
class ActionRecognitionModel(torch.nn.Module):
    def __init__(self, num_classes):
        super(ActionRecognitionModel, self).__init__()
        self.conv1 = torch.nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = torch.nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv2 = torch.nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = torch.nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.fc1 = torch.nn.Linear(128 * 8 * 8 * 8, 512)
        self.fc2 = torch.nn.Linear(512, num_classes)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load pre-trained action recognition model
action_model = ActionRecognitionModel(num_classes=3)  # Replace with your number of actions
action_model.load_state_dict(torch.load("warehouse_action_recognition_model.pth"))
action_model.eval()

# Preprocess function for action recognition
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to predict action for a single worker
def predict_action(worker_frame):
    worker_frame = Image.fromarray(worker_frame)
    worker_frame = transform(worker_frame).unsqueeze(0)
    with torch.no_grad():
        output = action_model(worker_frame)
        _, predicted = torch.max(output, 1)
        return predicted.item()

# Function to process a video frame
def process_frame(frame):
    # Detect workers using YOLO
    results = yolo_model(frame)
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = box.cls[0]
            if cls == 0:  # Class 0 is 'person' in YOLO
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

    # Update tracker with detections
    tracks = tracker.update_tracks(detections, frame=frame)

    # For each tracked worker, predict action
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()  # Get bounding box coordinates
        x1, y1, x2, y2 = map(int, ltrb)

        # Crop the worker from the frame
        worker_frame = frame[y1:y2, x1:x2]

        # Predict action
        action = predict_action(worker_frame)

        # Draw bounding box and action label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}, Action: {action}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame

# Main loop to process video
cap = cv2.VideoCapture(r"C:\Users\anura\Downloads\OLD_code\test_vidoes\3105196-uhd_3840_2160_30fps.mp4")  # Replace with your video path
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = process_frame(frame)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()