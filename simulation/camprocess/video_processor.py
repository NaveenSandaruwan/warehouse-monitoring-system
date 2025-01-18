import cv2
from ultralytics import YOLO
from file_writer import write_log

def process_video(input_video_path, output_video_path, log_file_path, camcoordinates,rawcoordinates,shared_data,cam_id):
    # Load the YOLOv8 model (pre-trained)
    model = YOLO("yolov8n.pt")  # Replace 'yolov8n.pt' with your YOLO model file if needed

    # Load the video
    cap = cv2.VideoCapture(input_video_path)

    # Check if video is opened
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Get the frame width, height, and FPS
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(5))

    # Calculate row boundaries
    row_height = frame_height // 3
    row_boundaries = [(0, row_height), (row_height, 2 * row_height), (2 * row_height, frame_height)]

    # Set up the VideoWriter to save the processed video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' for MP4 files
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Clear the log file if it exists
    with open(log_file_path, 'w') as file:
        file.write("Row-wise person count log\n")
        file.write("====================================\n")

    # Process the video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection with YOLOv8
        results = model(frame)  # Detect objects in the current frame

        # Filter results to keep only detections for the "person" class
        person_results = [result for result in results[0].boxes if result.cls == 0]  # Class ID 0 is for "person"

        # Initialize counts for each row
        row_counts = [0, 0, 0]

        # Draw blue horizontal lines to divide the screen into three rows
        for i in range(1, 3):
            y = row_height * i
            cv2.line(frame, (0, y), (frame_width, y), (255, 0, 0), 2)  # Blue line

        # Process detections and assign to rows based on the bottom line of the bounding box
        for person in person_results:
            x1, y1, x2, y2 = map(int, person.xyxy[0])  # Bounding box coordinates
            bottom_y = y2  # Use the bottom Y-coordinate of the bounding box

            # Determine which row the person belongs to
            for i, (start, end) in enumerate(row_boundaries):
                if start <= bottom_y < end:
                    row_counts[i] += 1
                    break

            # Draw bounding box and label on the frame
            confidence = person.conf.item()  # Convert tensor to float
            label = f"Person {confidence:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Annotate the frame with row counts
        for i, count in enumerate(row_counts):
            label = f"Row {i + 1}: {count} persons"
            cv2.putText(frame, label, (10, 50 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Write row counts to the log file
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        write_log(log_file_path, frame_number, row_counts,camcoordinates,rawcoordinates,shared_data,cam_id)

        # Write the processed frame to the output video
        out.write(frame)

        # Optionally, display the frame (for debugging)
        cv2.imshow('Frame', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
