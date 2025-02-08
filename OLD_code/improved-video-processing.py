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
from torchvision.models.video import r3d_18
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    """Configuration parameters for video processing"""
    face_recognition_interval: float = 3.0  # seconds between face recognition attempts
    action_recognition_frames: int = 16     # frames needed for action recognition
    detection_confidence: float = 0.6       # YOLO detection confidence threshold
    margin: int = 20                        # margin for face detection
    frame_buffer_size: int = 16             # size of frame buffer for action recognition

class VideoProcessor:
    def __init__(self, model_path: str = 'yolov8x.pt'):
        """Initialize all models and trackers"""
        self.config = ProcessingConfig()
        
        # Initialize models
        self.yolo_model = YOLO(model_path)
        self.tracker = Tracker(distance_function="euclidean", distance_threshold=30)
        #self.mp_pose = mp.solutions.pose.Pose()
        
        # Initialize action recognition
        self.action_model = self._setup_action_model()
        self.transform = self._setup_transforms()
        
        # State tracking
        self.last_recognition_time: Dict[int, float] = {}
        self.buffered_frames: List[np.ndarray] = []
        self.current_action: str = "Unknown"
        
        logger.info("VideoProcessor initialized successfully")
    '''
    def _setup_action_model(self) -> torch.nn.Module:
        """Setup and configure the action recognition model"""
        model = r3d_18(pretrained=True)
        model.eval()
        return model
    '''

    def _setup_action_model(self) -> torch.nn.Module:
        """Setup and configure the action recognition model"""
        model = r3d_18(pretrained=True)
        model.eval()  # Set to evaluation mode
        
        # Optional: Move to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
        
        return model
    
    # def _setup_transforms(self) -> transforms.Compose:
    #     """Setup image transformations pipeline"""
    #     return transforms.Compose([
    #         transforms.Resize((112, 112)),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    #     ])
    
    def _setup_transforms(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.43216, 0.394666, 0.37645],
                std=[0.22803, 0.22145, 0.216989]
        )
    ])


    def load_known_faces(self, folder: str = 'known_faces') -> Dict[str, str]:
        """Load known faces from directory"""
        known_faces = {}
        if not os.path.exists(folder):
            logger.warning(f"Known faces folder {folder} does not exist")
            return known_faces
            
        for filename in os.listdir(folder):
            if filename.endswith(('.jpg', '.png')):
                known_faces[filename] = os.path.join(folder, filename)
        logger.info(f"Loaded {len(known_faces)} known faces")
        return known_faces

    def process_video(self, input_path: str, output_path: str) -> None:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Video file not found: {input_path}")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {input_path}")

        writer = self._setup_video_writer(cap, output_path)
        known_faces = self.load_known_faces()

        frame_count = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                
                # Create a copy of the frame for processing
                processed_frame = frame.copy()
                
                # Process the frame
                processed_frame = self._process_single_frame(
                    processed_frame, frame_count, known_faces)
                
                # Write the processed frame
                writer.write(processed_frame)
                
                # Optional: Display the frame (for debugging)
                cv2.imshow('Processing', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            print(f"Error processing video: {str(e)}")
            raise
        finally:
            cap.release()
            writer.release()
            cv2.destroyAllWindows()
    '''
    def _process_single_frame(self, frame: np.ndarray, frame_count: int, 
                         known_faces: Dict[str, str]) -> np.ndarray:
        """Process a single frame of video"""
        # YOLO detection
        results = self.yolo_model(frame, conf=self.config.detection_confidence)
        detections = results[0]
        
        # Filter for person detections (class 0 is person in YOLO)
        person_detections = []
        for det in detections.boxes:
            if int(det.cls) == 0:  # 0 is the class ID for person
                person_detections.append(det)
        
        # Convert to Norfair format and update tracking
        norfair_detections = self._yolo_to_norfair(person_detections)
        tracked_objects = self.tracker.update(detections=norfair_detections)

        # Draw detection boxes first
        for det in person_detections:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            conf = float(det.conf)
            
            # Draw detection box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Draw confidence score
            conf_text = f"Conf: {conf:.2f}"
            cv2.putText(frame, conf_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Process each tracked object
        for tracked_obj in tracked_objects:
            if tracked_obj.estimate is not None:
                # Find matching detection for this tracked object
                matched_detection = None
                for det in person_detections:
                    x1, y1, x2, y2 = map(int, det.xyxy[0])
                    det_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
                    if np.linalg.norm(tracked_obj.estimate - det_center) < 50:  # Distance threshold
                        matched_detection = det
                        break

                if matched_detection is not None:
                    x1, y1, x2, y2 = map(int, matched_detection.xyxy[0])
                    
                    # Perform face recognition
                    name = self._handle_face_recognition(
                        frame, tracked_obj.id, (x1, y1, x2, y2), known_faces)
                    
                    # Draw bounding box and labels with better visibility
                    color = (0, 255, 0) if name != "Unknown" else (0, 165, 255)
                    
                    # Draw filled rectangle for text background
                    cv2.rectangle(frame, (x1, y1-60), (x1+200, y1), (0, 0, 0), -1)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw text
                    cv2.putText(frame, f"ID: {tracked_obj.id}", (x1, y1-40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(frame, name, (x1, y1-25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(frame, f"Action: {self.current_action}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return frame
    '''
    def _process_single_frame(self, frame: np.ndarray, frame_count: int, 
                         known_faces: Dict[str, str]) -> np.ndarray:
        """Process a single frame of video"""
        # YOLO detection
        results = self.yolo_model(frame, conf=self.config.detection_confidence)
        detections = results[0]
        
        # Filter for person detections (class 0 is person in YOLO)
        person_detections = []
        for det in detections.boxes:
            if int(det.cls) == 0:  # 0 is the class ID for person
                person_detections.append(det)
        
        # Convert to Norfair format and update tracking
        norfair_detections = self._yolo_to_norfair(person_detections)
        tracked_objects = self.tracker.update(detections=norfair_detections)

        # Update action recognition buffer with RGB frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.buffered_frames.append(rgb_frame)
        
        # Perform action recognition when enough frames are collected
        if len(self.buffered_frames) >= self.config.action_recognition_frames:
            self.current_action = self._recognize_action(self.buffered_frames)
            self.buffered_frames = []  # Clear buffer after recognition
            print(f"Detected action: {self.current_action}")  # Debug print

        # Draw detection boxes first
        for det in person_detections:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            conf = float(det.conf)
            
            # Draw detection box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Draw confidence score
            conf_text = f"Conf: {conf:.2f}"
            cv2.putText(frame, conf_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Process each tracked object
        for tracked_obj in tracked_objects:
            if tracked_obj.estimate is not None:
                # Find matching detection for this tracked object
                matched_detection = None
                for det in person_detections:
                    x1, y1, x2, y2 = map(int, det.xyxy[0])
                    det_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
                    if np.linalg.norm(tracked_obj.estimate - det_center) < 50:  # Distance threshold
                        matched_detection = det
                        break

                if matched_detection is not None:
                    x1, y1, x2, y2 = map(int, matched_detection.xyxy[0])
                    
                    # Perform face recognition
                    name = self._handle_face_recognition(
                        frame, tracked_obj.id, (x1, y1, x2, y2), known_faces)
                    
                    # Draw bounding box and labels with better visibility
                    color = (0, 255, 0) if name != "Unknown" else (0, 165, 255)
                    
                    # Draw filled rectangle for text background
                    cv2.rectangle(frame, (x1, y1-60), (x1+200, y1), (0, 0, 0), -1)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw text
                    cv2.putText(frame, f"ID: {tracked_obj.id}", (x1, y1-40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(frame, name, (x1, y1-25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(frame, f"Action: {self.current_action}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return frame


    def _recognize_action(self, frames: List[np.ndarray]) -> str:
        """Recognize action from a sequence of frames"""
        try:
            # Process only the required number of frames
            frames_to_process = frames[:self.config.action_recognition_frames]
            
            # Convert frames to tensors
            processed_frames = []
            for frame in frames_to_process:
                # Convert numpy array to PIL Image
                pil_image = Image.fromarray(frame)
                # Apply transformations
                tensor = self.transform(pil_image)
                processed_frames.append(tensor)
            
            # Stack frames into a single tensor
            processed_frames = torch.stack(processed_frames)
            
            # Add batch dimension and reorder dimensions
            batch = processed_frames.unsqueeze(0)  # Add batch dimension
            batch = batch.permute(0, 2, 1, 3, 4)  # Reorder to [B, C, T, H, W]
            
            # Move to GPU if available
            if torch.cuda.is_available():
                batch = batch.cuda()
                self.action_model = self.action_model.cuda()
            
            # Perform inference
            with torch.no_grad():
                outputs = self.action_model(batch)
                predicted_class = torch.argmax(outputs, dim=1).item()
            
            # Map class index to action label
            action_labels = {
                0: "Walking",
                1: "Running",
                2: "Standing",
                3: "Sitting",
                4: "Jumping",
                5: "Dancing"
            }
            
            return action_labels.get(predicted_class, "Unknown")
            
        except Exception as e:
            print(f"Action recognition error: {str(e)}")
            return "Unknown"

    def _update_action_recognition(self, frame: np.ndarray) -> None:
        """Update action recognition buffer and predict action"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.buffered_frames.append(rgb_frame)
        
        if len(self.buffered_frames) >= self.config.action_recognition_frames:
            self.current_action = self._recognize_action(self.buffered_frames)
            self.buffered_frames = []

    def _process_tracked_object(self, frame: np.ndarray, tracked_obj: Detection, 
                              detections: List, known_faces: Dict[str, str]) -> np.ndarray:
        """Process a single tracked object in the frame"""
        if tracked_obj.estimate is None or len(tracked_obj.estimate) < 2:
            return frame

        # Extract object location and ID
        x, y = map(int, tracked_obj.estimate[:2])
        obj_id = tracked_obj.id

        # Find matching YOLO detection
        for box in detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Perform face recognition if needed
            name = self._handle_face_recognition(
                frame, obj_id, (x1, y1, x2, y2), known_faces)
            
            # Draw visual elements
            frame = self._draw_annotations(
                frame, (x1, y1, x2, y2), obj_id, name)

        return frame

    def _handle_face_recognition(self, frame: np.ndarray, obj_id: int, 
                               bbox: Tuple[int, int, int, int], 
                               known_faces: Dict[str, str]) -> str:
        """Handle face recognition for a tracked object"""
        x1, y1, x2, y2 = bbox
        current_time = time.time()
        
        # Check if we should perform recognition
        if not self._should_recognize_face(obj_id, current_time):
            return "Recognized Earlier"

        # Extract and process face region
        face_region = frame[
            max(0, y1 - self.config.margin):min(frame.shape[0], y2 + self.config.margin),
            max(0, x1 - self.config.margin):min(frame.shape[1], x2 + self.config.margin)
        ]

        # Perform recognition
        temp_face_path = f"temp_face_{obj_id}.jpg"
        try:
            return self._recognize_face(face_region, temp_face_path, known_faces)
        finally:
            if os.path.exists(temp_face_path):
                os.remove(temp_face_path)

    def _should_recognize_face(self, obj_id: int, current_time: float) -> bool:
        """Determine if face recognition should be performed"""
        if obj_id not in self.last_recognition_time:
            self.last_recognition_time[obj_id] = current_time
            return True
            
        if current_time - self.last_recognition_time[obj_id] > self.config.face_recognition_interval:
            self.last_recognition_time[obj_id] = current_time
            return True
            
        return False

    def _recognize_face(self, face_region: np.ndarray, temp_path: str, 
                       known_faces: Dict[str, str]) -> str:
        """Perform face recognition"""
        cv2.imwrite(temp_path, face_region)
        
        for known_name, known_face_path in known_faces.items():
            try:
                result = DeepFace.verify(
                    img1_path=temp_path,
                    img2_path=known_face_path,
                    model_name="VGG-Face"
                )
                if result['verified']:
                    return os.path.splitext(known_name)[0]
            except Exception as e:
                logger.warning(f"Face recognition error: {str(e)}")
                
        return "Unknown"

    def _recognize_action(self, frames: List[np.ndarray]) -> str:
        """Recognize action from a sequence of frames"""
        try:
            processed_frames = torch.stack([
                self.transform(Image.fromarray(frame)) 
                for frame in frames
            ])
            
            batch = processed_frames.unsqueeze(0).permute(0, 2, 1, 3, 4)
            
            with torch.no_grad():
                outputs = self.action_model(batch)
                predicted_class = torch.argmax(outputs, dim=1).item()
                
            action_labels = {
                0: "Walking",
                1: "Lifting",
                2: "Sitting"
            }
            return action_labels.get(predicted_class, "Unknown")
            
        except Exception as e:
            logger.error(f"Action recognition error: {str(e)}")
            return "Unknown"

    def _draw_annotations(self, frame: np.ndarray, bbox: Tuple[int, int, int, int],
                        obj_id: int, name: str) -> np.ndarray:
        """Draw bounding boxes and annotations on frame"""
        x1, y1, x2, y2 = bbox
        color = (0, 255, 0) if name != "Unknown" else (255, 0, 0)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw text annotations
        texts = [
            (f"ID: {obj_id}", (x1, y1 - 40)),
            (name, (x1, y1 - 20)),
            (f"Action: {self.current_action}", (x1, y2 + 20))
        ]
        
        for text, position in texts:
            cv2.putText(frame, text, position, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame

    def _setup_video_writer(self, cap: cv2.VideoCapture, 
                          output_path: str) -> cv2.VideoWriter:
        """Setup video writer with proper parameters"""
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        return cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    def _yolo_to_norfair(self, detections) -> List[Detection]:
        """Convert YOLO detections to Norfair format"""
        norfair_detections = []
        for box in detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            score = box.conf[0].item() if isinstance(box.conf, torch.Tensor) else box.conf
            norfair_detections.append(Detection(points=center, scores=np.array([score])))
        return norfair_detections

# Usage example
if __name__ == "__main__":
    processor = VideoProcessor()
    processor.process_video(
        input_path='free-no-copyright-content---people-walking-free-stock-footage-royalty.mp4',
        output_path='output_2.mp4'
    )
