import cv2
from ultralytics import YOLO
from deepface import DeepFace
from norfair import Detection, Tracker
import torch
from torchvision import transforms
import numpy as np
import os
import time
from PIL import Image
from torchvision.models.video import r3d_18
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
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
        
        # Initialize action recognition
        self.action_model = self._setup_action_model()
        self.transform = self._setup_transforms()
        
        # State tracking
        self.last_recognition_time: Dict[int, float] = {}
        self.object_actions: Dict[int, Dict[str, Any]] = {}
        
        logger.info("VideoProcessor initialized successfully")
    
    def _setup_action_model(self) -> torch.nn.Module:
        """Setup and configure the action recognition model"""
        model = r3d_18(pretrained=True)
        model.eval()  # Set to evaluation mode
        
        # Optional: Move to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
            logger.info("Action recognition model moved to GPU")
        
        return model
    
    def _setup_transforms(self) -> transforms.Compose:
        """Setup image transformations pipeline"""
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
                    logger.info("Processing terminated by user")
                    break

        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            raise
        finally:
            cap.release()
            writer.release()
            cv2.destroyAllWindows()
            logger.info("Released video resources and closed all windows")

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

        # Process each tracked object
        for tracked_obj in tracked_objects:
            obj_id = tracked_obj.id

            if obj_id not in self.object_actions:
                # Initialize action tracking for new object
                self.object_actions[obj_id] = {
                    'buffered_frames': [],
                    'current_action': "Unknown"
                }
                logger.info(f"Initialized action tracking for object ID {obj_id}")

            # Get the current bounding box for the tracked object
            matched_detection = self._find_matched_detection(tracked_obj, person_detections)
            if matched_detection is not None:
                x1, y1, x2, y2 = map(int, matched_detection.xyxy[0])

                # Extract the region of interest (ROI) for the object
                roi = frame[max(0, y1 - self.config.margin):min(frame.shape[0], y2 + self.config.margin),
                            max(0, x1 - self.config.margin):min(frame.shape[1], x2 + self.config.margin)]

                if roi.size == 0:
                    logger.warning(f"Empty ROI for object ID {obj_id}, skipping action recognition")
                    continue

                # Update frame buffer for action recognition
                rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                self.object_actions[obj_id]['buffered_frames'].append(rgb_roi)
                logger.debug(f"Appended frame to buffer for object ID {obj_id}")

                # Perform action recognition if enough frames are buffered
                if len(self.object_actions[obj_id]['buffered_frames']) >= self.config.action_recognition_frames:
                    action = self._recognize_action(self.object_actions[obj_id]['buffered_frames'])
                    self.object_actions[obj_id]['current_action'] = action
                    self.object_actions[obj_id]['buffered_frames'] = []  # Clear buffer after recognition
                    logger.info(f"Detected action for ID {obj_id}: {action}")

                # Perform face recognition
                name = self._handle_face_recognition(
                    frame, obj_id, (x1, y1, x2, y2), known_faces)
                
                # Draw annotations
                color = (0, 255, 0) if name != "Unknown" else (0, 165, 255)
                # Draw filled rectangle for text background
                cv2.rectangle(frame, (x1, y1-60), (x1+200, y1), (0, 0, 0), -1)
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                # Draw text
                cv2.putText(frame, f"ID: {obj_id}", (x1, y1-40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(frame, name, (x1, y1-25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(frame, f"Action: {self.object_actions[obj_id]['current_action']}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            else:
                logger.warning(f"No matching detection found for object ID {obj_id}")

        return frame

    def _find_matched_detection(self, tracked_obj: Detection, detections: List) -> Optional[Detection]:
        """Find the detection that matches the tracked object based on proximity"""
        if tracked_obj.estimate is None:
            logger.warning(f"No estimate available for tracked object ID {tracked_obj.id}")
            return None

        for det in detections:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            det_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            if np.linalg.norm(tracked_obj.estimate - det_center) < 50:  # Distance threshold
                return det
        logger.warning(f"No matched detection within threshold for object ID {tracked_obj.id}")
        return None

    def _recognize_action(self, frames: List[np.ndarray]) -> str:
        """Recognize action from a sequence of frames"""
        try:
            # Ensure exactly the required number of frames
            if len(frames) != self.config.action_recognition_frames:
                logger.warning(f"Insufficient frames for action recognition: {len(frames)}")
                return "Unknown"
            
            # Convert frames to tensors
            processed_frames = []
            for frame in frames:
                # Convert numpy array to PIL Image
                pil_image = Image.fromarray(frame)
                # Apply transformations
                tensor = self.transform(pil_image)
                processed_frames.append(tensor)
            
            # Stack frames into a single tensor
            processed_frames = torch.stack(processed_frames)  # Shape: [T, C, H, W]
            
            # Add batch dimension and reorder dimensions to [B, C, T, H, W]
            batch = processed_frames.unsqueeze(0).permute(0, 1, 2, 3, 4)
            
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
                5: "Dancing",
                # Add more mappings as necessary
            }
            action = action_labels.get(predicted_class, "Unknown")
            logger.info(f"Model predicted class {predicted_class}: {action}")
            return action
            
        except Exception as e:
            logger.error(f"Action recognition error: {str(e)}")
            return "Unknown"

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

        if face_region.size == 0:
            logger.warning(f"Empty face region for object ID {obj_id}")
            return "Unknown"

        # Perform recognition
        temp_face_path = f"temp_face_{obj_id}.jpg"
        try:
            cv2.imwrite(temp_face_path, face_region)
            return self._recognize_face(face_region, temp_face_path, known_faces)
        except Exception as e:
            logger.error(f"Error handling face recognition for ID {obj_id}: {str(e)}")
            return "Unknown"
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
        try:
            # Save the cropped face region temporarily
            cv2.imwrite(temp_path, face_region)
            
            # Iterate through known faces and verify
            for known_name, known_face_path in known_faces.items():
                try:
                    result = DeepFace.verify(
                        img1_path=temp_path,
                        img2_path=known_face_path,
                        model_name="VGG-Face",
                        enforce_detection=False  # Continue even if face not detected
                    )
                    if result['verified']:
                        logger.info(f"Face recognized: {os.path.splitext(known_name)[0]} for ID {known_name}")
                        return os.path.splitext(known_name)[0]
                except Exception as e:
                    logger.warning(f"Face recognition error between {temp_path} and {known_face_path}: {str(e)}")
                    
            logger.info(f"Face not recognized for object ID associated with {temp_path}")
            return "Unknown"
        except Exception as e:
            logger.error(f"Error during face recognition: {str(e)}")
            return "Unknown"

    def _setup_video_writer(self, cap: cv2.VideoCapture, 
                            output_path: str) -> cv2.VideoWriter:
        """Setup video writer with proper parameters"""
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        logger.info(f"Video writer initialized with FPS: {fps}, Width: {width}, Height: {height}")
        return writer

    def _yolo_to_norfair(self, detections) -> List[Detection]:
        """Convert YOLO detections to Norfair format"""
        norfair_detections = []
        for box in detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            score = box.conf[0].item() if isinstance(box.conf, torch.Tensor) else box.conf
            norfair_detections.append(Detection(points=center, scores=np.array([score])))
        logger.debug(f"Converted {len(norfair_detections)} detections to Norfair format")
        return norfair_detections

# Usage example
if __name__ == "__main__":
    processor = VideoProcessor()
    processor.process_video(
        input_path='free-no-copyright-content---people-walking-free-stock-footage-royalty.mp4',
        output_path='output_3.mp4'
    )
