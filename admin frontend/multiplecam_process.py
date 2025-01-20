import cv2

def start_camera(camera_id):
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow(f'Camera {camera_id}', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()