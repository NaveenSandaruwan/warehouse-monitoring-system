# import cv2
# from ultralytics import YOLO

# # Load the YOLOv8 model (pre-trained)
# model = YOLO("yolov8n.pt")  # You can replace 'yolov8n.pt' with the path to your YOLO model file

# # Load the video
# cap = cv2.VideoCapture("testing/examplevideo.mp4")

# # Check if video is opened
# if not cap.isOpened():
#     print("Error: Could not open video file.")
#     exit()

# # Get the frame width, height, and FPS
# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))
# fps = int(cap.get(5))

# # Set up the VideoWriter to save the processed video
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' for MP4 files
# output_path = 'testing/output_video.mp4'
# out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# # Process the video frame by frame
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Perform object detection with YOLOv8
#     results = model(frame)  # Detect objects in the current frame

#     # Filter results to keep only detections for the "person" class
#     person_results = [result for result in results[0].boxes if result.cls == 0]  # Class ID 0 is for "person"

#     # Annotate the frame with bounding boxes and labels for persons only
#     for person in person_results:
#         x1, y1, x2, y2 = map(int, person.xyxy[0])  # Bounding box coordinates
#         confidence = person.conf.item()  # Convert tensor to float
#         label = f"Person {confidence:.2f}"
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
#         cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     # Optionally, display the frame (for debugging)
#     cv2.imshow('Frame', frame)

#     # Write the processed frame to the output video
#     out.write(frame)

#     # Exit on 'q' key press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# out.release()
# cv2.destroyAllWindows()
import cv2
from ultralytics import YOLO

# Load the YOLOv8 model (pre-trained)
model = YOLO("yolov8n.pt")  # You can replace 'yolov8n.pt' with the path to your YOLO model file

# Load the video
cap = cv2.VideoCapture("testing/examplevideo.mp4")

# Check if video is opened
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get the frame width, height, and FPS
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(5))

# Set up the VideoWriter to save the processed video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' for MP4 files
output_path = 'testing/output_video.mp4'
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection with YOLOv8
    results = model(frame)  # Detect objects in the current frame

    # Filter results to keep only detections for the "person" class
    person_results = [result for result in results[0].boxes if result.cls == 0]  # Class ID 0 is for "person"

    # Annotate the frame with bounding boxes and labels for persons only
    for person in person_results:
        x1, y1, x2, y2 = map(int, person.xyxy[0])  # Bounding box coordinates
        confidence = person.conf.item()  # Convert tensor to float
        label = f"Person {confidence:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the count of humans on the frame
    human_count = len(person_results)
    count_label = f"Person detected: {human_count}"
    cv2.putText(frame, count_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Optionally, display the frame (for debugging)
    cv2.imshow('Frame', frame)

    # Write the processed frame to the output video
    out.write(frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
