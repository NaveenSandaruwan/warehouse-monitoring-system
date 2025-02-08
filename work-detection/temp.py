import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from typing import List, Tuple, Dict
from dataclasses import dataclass
import time

@dataclass
class Landmark:
    x: float
    y: float
    z: float
    visibility: float

@dataclass
class PersonDetection:
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    confidence: float
    landmarks: List[Landmark] = None
    
class MultiPersonWorkDetector:
    def __init__(self,
                 movement_threshold: float = 15.0,
                 pose_movement_threshold: float = 20.0,
                 size_change_threshold: float = 0.1,
                 time_window: int = 10):
        """
        Initialize work detector with both YOLO and Pose estimation.
        
        Args:
            movement_threshold: Threshold for person movement detection
            pose_movement_threshold: Threshold for skeleton movement
            size_change_threshold: Threshold for size change detection
            time_window: Number of frames to consider for analysis
        """
        # Initialize YOLO
        self.yolo_model = YOLO('yolov8n.pt')
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=1
        )
        
        # Parameters
        self.movement_threshold = movement_threshold
        self.pose_movement_threshold = pose_movement_threshold
        self.size_change_threshold = size_change_threshold
        self.time_window = time_window
        
        # Tracking data
        self.tracks = {}  # Format: {track_id: {history: [], working_status: bool, ...}}
        self.next_track_id = 0
        
    def _calculate_bbox_movement(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """Calculate movement between two bounding boxes."""
        center1 = (bbox1[0] + bbox1[2]/2, bbox1[1] + bbox1[3]/2)
        center2 = (bbox2[0] + bbox2[2]/2, bbox2[1] + bbox2[3]/2)
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def _calculate_size_change(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """Calculate relative size change between two bounding boxes."""
        area1 = bbox1[2] * bbox1[3]
        area2 = bbox2[2] * bbox2[3]
        return abs(area1 - area2) / area1
    
    def _calculate_pose_movement(self, landmarks1: List[Landmark], landmarks2: List[Landmark]) -> float:
        """Calculate weighted movement of key pose points."""
        if not landmarks1 or not landmarks2:
            return 0.0
        
        key_points = {
            11: 1.5,  # left shoulder
            12: 1.5,  # right shoulder
            13: 2.0,  # left elbow
            14: 2.0,  # right elbow
            15: 2.0,  # left wrist
            16: 2.0,  # right wrist
            23: 1.0,  # left hip
            24: 1.0,  # right hip
        }
        
        total_movement = 0
        total_weight = 0
        
        for point, weight in key_points.items():
            if (point < len(landmarks1) and point < len(landmarks2) and
                landmarks1[point].visibility > 0.6 and
                landmarks2[point].visibility > 0.6):
                p1, p2 = landmarks1[point], landmarks2[point]
                movement = np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
                total_movement += movement * weight
                total_weight += weight
        
        return total_movement / total_weight if total_weight > 0 else 0
    
    def _match_detections_to_tracks(self, 
                                  detections: List[PersonDetection],
                                  frame_shape: Tuple[int, int]) -> Dict[int, PersonDetection]:
        """Match new detections to existing tracks."""
        height, width = frame_shape
        matched_tracks = {}
        used_detections = set()
        
        # Calculate IoU matrix between all tracks and detections
        for track_id, track in self.tracks.items():
            if 'last_bbox' not in track:
                continue
                
            best_iou = 0
            best_detection = None
            
            for i, detection in enumerate(detections):
                if i in used_detections:
                    continue
                    
                # Calculate IoU
                bbox1 = track['last_bbox']
                bbox2 = detection.bbox
                
                x1 = max(bbox1[0], bbox2[0])
                y1 = max(bbox1[1], bbox2[1])
                x2 = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
                y2 = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])
                
                if x2 <= x1 or y2 <= y1:
                    continue
                    
                intersection = (x2 - x1) * (y2 - y1)
                area1 = bbox1[2] * bbox1[3]
                area2 = bbox2[2] * bbox2[3]
                union = area1 + area2 - intersection
                iou = intersection / union if union > 0 else 0
                
                if iou > best_iou and iou > 0.3:  # IOU threshold
                    best_iou = iou
                    best_detection = i
            
            if best_detection is not None:
                matched_tracks[track_id] = detections[best_detection]
                used_detections.add(best_detection)
        
        # Create new tracks for unmatched detections
        for i, detection in enumerate(detections):
            if i not in used_detections:
                new_track_id = self.next_track_id
                self.next_track_id += 1
                self.tracks[new_track_id] = {
                    'history': [],
                    'working_status': False,
                    'last_seen': time.time()
                }
                matched_tracks[new_track_id] = detection
        
        return matched_tracks
    
    def process_frame(self, frame: np.ndarray) -> Dict[int, bool]:
        """
        Process a frame and return working status for each person.
        
        Returns:
            Dictionary mapping track IDs to working status
        """
        height, width = frame.shape[:2]
        
        # 1. Detect persons using YOLO
        yolo_results = self.yolo_model(frame, classes=[0])  # class 0 is person
        
        # 2. Process detections
        current_detections = []
        for result in yolo_results:
            boxes = result.boxes
            for box in boxes:
                if box.conf[0] > 0.5:  # Confidence threshold
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1
                    detection = PersonDetection(
                        bbox=(x1, y1, w, h),
                        confidence=float(box.conf[0])
                    )
                    current_detections.append(detection)
        
        # 3. Match detections to tracks
        matched_tracks = self._match_detections_to_tracks(current_detections, frame.shape[:2])
        
        # 4. Update tracks and determine working status
        current_time = time.time()
        work_status = {}
        
        for track_id, detection in matched_tracks.items():
            track = self.tracks[track_id]
            track['last_seen'] = current_time
            
            # Extract pose landmarks
            person_frame = frame[detection.bbox[1]:detection.bbox[1]+detection.bbox[3],
                               detection.bbox[0]:detection.bbox[0]+detection.bbox[2]]
            if person_frame.size > 0:
                pose_results = self.pose.process(cv2.cvtColor(person_frame, cv2.COLOR_BGR2RGB))
                if pose_results.pose_landmarks:
                    landmarks = []
                    for landmark in pose_results.pose_landmarks.landmark:
                        landmarks.append(Landmark(
                            x=landmark.x * detection.bbox[2] + detection.bbox[0],
                            y=landmark.y * detection.bbox[3] + detection.bbox[1],
                            z=landmark.z,
                            visibility=landmark.visibility
                        ))
                    detection.landmarks = landmarks
            
            # Update history
            track['history'].append(detection)
            if len(track['history']) > self.time_window:
                track['history'].pop(0)
            
            # Determine working status based on multiple factors
            if len(track['history']) >= 2:
                prev_detection = track['history'][-2]
                curr_detection = track['history'][-1]
                
                # Check different types of movement
                bbox_movement = self._calculate_bbox_movement(
                    prev_detection.bbox, curr_detection.bbox)
                
                size_change = self._calculate_size_change(
                    prev_detection.bbox, curr_detection.bbox)
                
                pose_movement = 0
                if prev_detection.landmarks and curr_detection.landmarks:
                    pose_movement = self._calculate_pose_movement(
                        prev_detection.landmarks, curr_detection.landmarks)
                
                # Combine different movement indicators
                is_working = (
                    bbox_movement > self.movement_threshold or
                    pose_movement > self.pose_movement_threshold or
                    size_change > self.size_change_threshold
                )
                
                track['working_status'] = is_working
            
            work_status[track_id] = track.get('working_status', False)
            track['last_bbox'] = detection.bbox
        
        # Remove old tracks
        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            if current_time - track['last_seen'] > 1.0:  # Remove after 1 second
                tracks_to_remove.append(track_id)
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
        
        return work_status
    
    def visualize_results(self, frame: np.ndarray, work_status: Dict[int, bool]) -> np.ndarray:
        """Draw detection results on frame."""
        output_frame = frame.copy()
        
        for track_id, track in self.tracks.items():
            if 'last_bbox' in track:
                bbox = track['last_bbox']
                is_working = work_status.get(track_id, False)
                
                # Draw bounding box
                color = (0, 255, 0) if is_working else (0, 0, 255)
                cv2.rectangle(output_frame, 
                            (bbox[0], bbox[1]),
                            (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                            color, 2)
                
                # Draw status text
                status_text = f"ID {track_id}: {'Working' if is_working else 'Not Working'}"
                cv2.putText(output_frame, status_text,
                          (bbox[0], bbox[1] - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw pose landmarks if available
                if track['history'] and track['history'][-1].landmarks:
                    landmarks = track['history'][-1].landmarks
                    for landmark in landmarks:
                        if landmark.visibility > 0.6:
                            cv2.circle(output_frame,
                                     (int(landmark.x), int(landmark.y)),
                                     2, (255, 0, 0), -1)
        
        return output_frame

# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = MultiPersonWorkDetector(
        movement_threshold=15.0,
        pose_movement_threshold=20.0,
        size_change_threshold=0.1,
        time_window=10
    )
    
    # Process video
    video_path = r"C:\Users\anura\Downloads\OLD_code\test_vidoes\free-no-copyright-content---people-walking-free-stock-footage-royalty.mp4"
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Create output video writer
        output_path = 'output_enhanced_detection.mp4'
        out = cv2.VideoWriter(output_path, 
                            cv2.VideoWriter_fourcc(*'mp4v'),
                            fps, 
                            (frame_width, frame_height))
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            work_status = detector.process_frame(frame)
            
            # Visualize results
            viz_frame = detector.visualize_results(frame, work_status)
            
            # Write and display frame
            out.write(viz_frame)
            cv2.imshow('Enhanced Work Detection', viz_frame)
            
            # Print status every 30 frames
            if frame_count % 30 == 0:
                print(f"\nFrame {frame_count}:")
                for track_id, is_working in work_status.items():
                    print(f"Person {track_id}: {'Working' if is_working else 'Not Working'}")
            
            frame_count += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error processing video: {e}")