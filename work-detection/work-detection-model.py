import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from typing import List, Tuple, Dict
from collections import defaultdict
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
                movement_threshold: float = 15.0,  # Overall movement threshold --
                pose_movement_threshold: float = 20.0,  # Skeleton movement threshold --
                size_change_threshold: float = 0.1,  # Size change threshold   --
                time_window: int = 10,  # Number of frames to analyze ..
                min_detection_confidence: float = 0.7,  # Minimum confidence for detections ..
                min_visibility_threshold: float = 0.5,  # Minimum visibility for landmarks ..
                working_confidence_threshold: float = 0.6  # Threshold to declare as working --
                ):
        """
        Initialize work detector with both YOLO and Pose estimation.
        """
        # Store all initialization parameters as instance variables
        self.movement_threshold = movement_threshold
        self.pose_movement_threshold = pose_movement_threshold
        self.size_change_threshold = size_change_threshold
        self.time_window = time_window
        self.min_detection_confidence = min_detection_confidence
        self.min_visibility_threshold = min_visibility_threshold
        self.working_confidence_threshold = working_confidence_threshold
        
        # Initialize YOLO
        self.yolo_model = YOLO('yolov8n.pt')
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=0.7,
            model_complexity=1
        )
        
        # Initialize tracking data
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
    
    # def _calculate_pose_movement(self, landmarks1: List[Landmark], landmarks2: List[Landmark]) -> float:
    #     """Calculate weighted movement of key pose points."""
    #     if not landmarks1 or not landmarks2:
    #         return 0.0
        
    #     key_points = {
    #         11: 1.5,  # left shoulder
    #         12: 1.5,  # right shoulder
    #         13: 2.0,  # left elbow
    #         14: 2.0,  # right elbow
    #         15: 2.0,  # left wrist
    #         16: 2.0,  # right wrist
    #         23: 1.0,  # left hip
    #         24: 1.0,  # right hip
    #     }
        
    #     total_movement = 0
    #     total_weight = 0
        
    #     for point, weight in key_points.items():
    #         if (point < len(landmarks1) and point < len(landmarks2) and
    #             landmarks1[point].visibility > 0.6 and
    #             landmarks2[point].visibility > 0.6):
    #             p1, p2 = landmarks1[point], landmarks2[point]
    #             movement = np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    #             total_movement += movement * weight
    #             total_weight += weight
        
    #     return total_movement / total_weight if total_weight > 0 else 0
    
    def _calculate_pose_movement(self, landmarks1: List[Landmark], landmarks2: List[Landmark]) -> Dict[str, float]:
        """Calculate weighted movement of all body parts."""
        if not landmarks1 or not landmarks2:
            return {'total': 0.0, 'parts': {}}
        
        # Define body parts and their weights
        body_parts = {
            'arms': {
                'points': [11,12,13,14,15,16],  # shoulders, elbows, wrists
                'weight': 3.0
            },
            'legs': {
                'points': [23,24,25,26,27,28,29,30,31,32],  # hips, knees, ankles
                'weight': 4.0
            },
            'torso': {
                'points': [11,12,23,24],  # shoulders and hips
                'weight': 3.0
            },
            'head': {
                'points': [0,1,2,3,4,5,6,7,8,9,10],  # face landmarks
                'weight': 0.5
            }
        }
        
        movements = {'parts': {}}
        total_weighted_movement = 0
        total_weight = 0
        
        for part_name, part_info in body_parts.items():
            part_movement = 0
            valid_points = 0
            
            for point in part_info['points']:
                if (point < len(landmarks1) and point < len(landmarks2) and
                    landmarks1[point].visibility > self.min_visibility_threshold and
                    landmarks2[point].visibility > self.min_visibility_threshold):
                    
                    p1, p2 = landmarks1[point], landmarks2[point]
                    movement = np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
                    part_movement += movement
                    valid_points += 1
            
            if valid_points > 0:
                avg_part_movement = part_movement / valid_points
                weighted_movement = avg_part_movement * part_info['weight']
                movements['parts'][part_name] = avg_part_movement
                total_weighted_movement += weighted_movement
                total_weight += part_info['weight']
        
        movements['total'] = total_weighted_movement / total_weight if total_weight > 0 else 0
        return movements

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
    
    # def _determine_working_status(self, track: Dict, curr_detection: PersonDetection) -> Tuple[bool, Dict]:
    #     """Determine if a person is working based on multiple movement factors."""
    #     movement_scores = {
    #         'bbox_movement': 0.0,
    #         'pose_movement': 0.0,
    #         'size_change': 0.0,
    #         'parts_movement': {}
    #     }
        
    #     if len(track['history']) >= 2:
    #         prev_detection = track['history'][-2]
            
    #         # Calculate bounding box movement
    #         bbox_movement = self._calculate_bbox_movement(
    #             prev_detection.bbox, curr_detection.bbox)
    #         movement_scores['bbox_movement'] = bbox_movement
            
    #         # Calculate size change
    #         size_change = self._calculate_size_change(
    #             prev_detection.bbox, curr_detection.bbox)
    #         movement_scores['size_change'] = size_change
            
    #         # Calculate pose movements
    #         if prev_detection.landmarks and curr_detection.landmarks:
    #             pose_movements = self._calculate_pose_movement(
    #                 prev_detection.landmarks, curr_detection.landmarks)
    #             movement_scores['pose_movement'] = pose_movements['total']
    #             movement_scores['parts_movement'] = pose_movements['parts']
            
    #         # Determine working status based on combined factors
    #         is_working = False
    #         confidence = 0.0
            
    #         # Check different movement types
    #         if bbox_movement > self.movement_threshold:
    #             confidence += 0.3
            
    #         if size_change > self.size_change_threshold:
    #             confidence += 0.2
            
    #         # Check body parts movements
    #         if movement_scores['parts_movement']:
    #             for part, movement in movement_scores['parts_movement'].items():
    #                 if movement > self.pose_movement_threshold:
    #                     if part in ['arms', 'legs']:
    #                         confidence += 0.25
    #                     elif part == 'torso':
    #                         confidence += 0.15
    #                     elif part == 'head':
    #                         confidence += 0.1
            
    #         is_working = confidence >= self.working_confidence_threshold
            
    #         return is_working, movement_scores
        
    #     return False, movement_scores

    # def process_frame(self, frame: np.ndarray) -> Dict[int, bool]:
    #     """
    #     Process a frame and return working status for each person.
        
    #     Returns:
    #         Dictionary mapping track IDs to working status
    #     """
    #     height, width = frame.shape[:2]
        
    #     # 1. Detect persons using YOLO
    #     yolo_results = self.yolo_model(frame, classes=[0])  # class 0 is person
        
    #     # 2. Process detections
    #     current_detections = []
    #     for result in yolo_results:
    #         boxes = result.boxes
    #         for box in boxes:
    #             if box.conf[0] > 0.5:  # Confidence threshold
    #                 x1, y1, x2, y2 = map(int, box.xyxy[0])
    #                 w, h = x2 - x1, y2 - y1
    #                 detection = PersonDetection(
    #                     bbox=(x1, y1, w, h),
    #                     confidence=float(box.conf[0])
    #                 )
    #                 current_detections.append(detection)
        
    #     # 3. Match detections to tracks
    #     matched_tracks = self._match_detections_to_tracks(current_detections, frame.shape[:2])
        
    #     # 4. Update tracks and determine working status
    #     current_time = time.time()
    #     work_status = {}
        
    #     for track_id, detection in matched_tracks.items():
    #         track = self.tracks[track_id]
    #         is_working, movement_scores = self._determine_working_status(track, detection)
            
    #         work_status[track_id] = {
    #             'is_working': is_working,
    #             'movement_scores': movement_scores
    #         }
            
    #         track['working_status'] = is_working
    #         track['last_movement_scores'] = movement_scores
        
    #     return work_status
    
    def visualize_results(self, frame: np.ndarray, work_status: Dict[int, Dict]) -> np.ndarray:
        output_frame = frame.copy()
        
        for track_id, status in work_status.items():
            if track_id in self.tracks and 'last_bbox' in self.tracks[track_id]:
                bbox = self.tracks[track_id]['last_bbox']
                is_working = status['is_working']
                scores = status['movement_scores']
                
                # Draw bounding box
                color = (0, 255, 0) if is_working else (0, 0, 255)
                cv2.rectangle(output_frame, 
                            (bbox[0], bbox[1]),
                            (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                            color, 2)
                
                # Draw status and movement information
                y_offset = bbox[1] - 10
                for i, (part, score) in enumerate(scores['parts_movement'].items()):
                    text = f"{part}: {score:.2f}"
                    y_offset -= 20
                    cv2.putText(output_frame, text,
                            (bbox[0], y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # Draw overall status
                status_text = f"ID {track_id}: {'Working' if is_working else 'Not Working'}"
                cv2.putText(output_frame, status_text,
                        (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return output_frame

    # def visualize_results(self, frame: np.ndarray, work_status: Dict[int, Dict]) -> np.ndarray:
    #     """
    #     Visualize detection results with colored boxes, skeletons, and working status.
        
    #     Args:
    #         frame: Input frame
    #         work_status: Dictionary containing working status and movement scores for each person
            
    #     Returns:
    #         Annotated frame with visualizations
    #     """
    #     output_frame = frame.copy()
        
    #     for track_id, status in work_status.items():
    #         if track_id in self.tracks and 'last_bbox' in self.tracks[track_id]:
    #             bbox = self.tracks[track_id]['last_bbox']
    #             is_working = status['is_working']
    #             scores = status['movement_scores']
                
    #             # Colors in BGR format
    #             WORKING_COLOR = (0, 255, 0)      # Green for working
    #             NOT_WORKING_COLOR = (0, 0, 255)   # Red for not working
    #             SKELETON_COLOR = (255, 165, 0)    # Orange for skeleton
    #             TEXT_COLOR = (255, 255, 255)      # White for text
                
    #             # Draw bounding box with thicker lines
    #             color = WORKING_COLOR if is_working else NOT_WORKING_COLOR
    #             cv2.rectangle(output_frame, 
    #                         (int(bbox[0]), int(bbox[1])),
    #                         (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
    #                         color, 3)
                
    #             # Create a semi-transparent overlay for text background
    #             overlay = output_frame.copy()
    #             alpha = 0.3
                
    #             # Draw status text with background
    #             status_text = f"ID {track_id}: {'Working' if is_working else 'Not Working'}"
    #             text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    #             text_x = int(bbox[0])
    #             text_y = int(bbox[1] - 10)
                
    #             # Draw text background
    #             cv2.rectangle(overlay,
    #                         (text_x - 5, text_y - text_size[1] - 5),
    #                         (text_x + text_size[0] + 5, text_y + 5),
    #                         (0, 0, 0), -1)
    #             output_frame = cv2.addWeighted(overlay, alpha, output_frame, 1 - alpha, 0)
                
    #             # Draw status text
    #             cv2.putText(output_frame, status_text,
    #                     (text_x, text_y),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)
                
    #             # Draw movement scores
    #             y_offset = text_y - 20
    #             for part, score in scores.get('parts_movement', {}).items():
    #                 score_text = f"{part}: {score:.2f}"
    #                 cv2.putText(output_frame, score_text,
    #                         (text_x, y_offset),
    #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
    #                 y_offset -= 20
                
    #             # Draw skeleton if landmarks exist
    #             if 'last_landmarks' in self.tracks[track_id]:
    #                 landmarks = self.tracks[track_id]['last_landmarks']
    #                 if landmarks:
    #                     # Define skeleton connections
    #                     skeleton_connections = [
    #                         # Upper body
    #                         (11, 12),  # shoulders
    #                         (11, 13), (13, 15),  # left arm
    #                         (12, 14), (14, 16),  # right arm
    #                         (11, 23), (12, 24),  # torso
    #                         # Lower body
    #                         (23, 24),  # hips
    #                         (23, 25), (25, 27),  # left leg
    #                         (24, 26), (26, 28),  # right leg
    #                         # Face
    #                         (0, 1), (1, 2), (2, 3), (3, 4),  # face outline
    #                         (5, 6), (6, 7), (7, 8),          # eyebrows
    #                         (9, 10)                          # mouth
    #                     ]
                        
    #                     # Draw skeleton connections
    #                     for start_idx, end_idx in skeleton_connections:
    #                         if (start_idx < len(landmarks) and end_idx < len(landmarks) and
    #                             landmarks[start_idx].visibility > self.min_visibility_threshold and
    #                             landmarks[end_idx].visibility > self.min_visibility_threshold):
                                
    #                             start_point = (
    #                                 int(landmarks[start_idx].x * frame.shape[1]),
    #                                 int(landmarks[start_idx].y * frame.shape[0])
    #                             )
    #                             end_point = (
    #                                 int(landmarks[end_idx].x * frame.shape[1]),
    #                                 int(landmarks[end_idx].y * frame.shape[0])
    #                             )
                                
    #                             # Draw lines with anti-aliasing
    #                             cv2.line(output_frame, start_point, end_point, SKELETON_COLOR, 2, cv2.LINE_AA)
                        
    #                     # Draw landmark points
    #                     for landmark in landmarks:
    #                         if landmark.visibility > self.min_visibility_threshold:
    #                             point = (
    #                                 int(landmark.x * frame.shape[1]),
    #                                 int(landmark.y * frame.shape[0])
    #                             )
    #                             cv2.circle(output_frame, point, 4, SKELETON_COLOR, -1, cv2.LINE_AA)
    #                             cv2.circle(output_frame, point, 2, (255, 255, 255), -1, cv2.LINE_AA)
        
    #     return output_frame

    # def _determine_working_status(self, track: Dict, curr_detection: PersonDetection) -> Tuple[bool, Dict]:
    #     """
    #     Determine if a person is working based on movement analysis across multiple frames.
    #     Uses weighted scoring system for different types of movements.
    #     """
    #     NUM_HISTORY_FRAMES = 5  # Number of frames to analyze for movement patterns
        
    #     movement_scores = {
    #         'bbox_movement': 0.0,
    #         'pose_movement': 0.0,
    #         'size_change': 0.0,
    #         'parts_movement': {},
    #         'overall_score': 0.0
    #     }
        
    #     # Return early if not enough history
    #     if len(track['history']) < NUM_HISTORY_FRAMES:
    #         return False, movement_scores
        
    #     # Analyze movements across multiple recent frames
    #     recent_history = track['history'][-NUM_HISTORY_FRAMES:]
    #     movements_over_time = {
    #         'bbox': [],
    #         'size': [],
    #         'parts': defaultdict(list)
    #     }
        
    #     # Calculate movements between consecutive frames
    #     for i in range(len(recent_history) - 1):
    #         prev_detection = recent_history[i]
    #         next_detection = recent_history[i + 1]
            
    #         # Bounding box movement
    #         bbox_mov = self._calculate_bbox_movement(prev_detection.bbox, next_detection.bbox)
    #         movements_over_time['bbox'].append(bbox_mov)
            
    #         # Size change
    #         size_change = self._calculate_size_change(prev_detection.bbox, next_detection.bbox)
    #         movements_over_time['size'].append(size_change)
            
    #         # Pose movements
    #         if prev_detection.landmarks and next_detection.landmarks:
    #             pose_movements = self._calculate_pose_movement(
    #                 prev_detection.landmarks, next_detection.landmarks)
                
    #             for part, movement in pose_movements['parts'].items():
    #                 movements_over_time['parts'][part].append(movement)
        
    #     # Calculate average movements
    #     avg_bbox_movement = np.mean(movements_over_time['bbox']) if movements_over_time['bbox'] else 0
    #     avg_size_change = np.mean(movements_over_time['size']) if movements_over_time['size'] else 0
        
    #     # Updated movement thresholds
    #     THRESHOLDS = {
    #         'bbox': {
    #             'slight': 5.0,   # pixels per frame
    #             'significant': 15.0
    #         },
    #         'size': {
    #             'slight': 0.05,  # relative change
    #             'significant': 0.15
    #         },
    #         'pose': {
    #             'slight': 0.02,  # relative movement
    #             'significant': 0.08
    #         }
    #     }
        
    #     # Score initialization
    #     confidence_score = 0.0
        
    #     # Score bbox movement (30% weight)
    #     if avg_bbox_movement < THRESHOLDS['bbox']['slight']:
    #         confidence_score += 0.0
    #     elif avg_bbox_movement < THRESHOLDS['bbox']['significant']:
    #         confidence_score += 0.15
    #     else:
    #         confidence_score += 0.3
        
    #     # Score size change (20% weight)
    #     if avg_size_change < THRESHOLDS['size']['slight']:
    #         confidence_score += 0.0
    #     elif avg_size_change < THRESHOLDS['size']['significant']:
    #         confidence_score += 0.1
    #     else:
    #         confidence_score += 0.2
        
    #     # Score pose movements (50% weight)
    #     pose_score = 0.0
    #     part_weights = {
    #         'arms': 0.3,    # 30% of pose score
    #         'legs': 0.2,    # 20% of pose score
    #         'torso': 0.1,   # 10% of pose score
    #         'head': 0.1     # 10% of pose score
    #     }
        
    #     for part, movements in movements_over_time['parts'].items():
    #         if movements:
    #             avg_movement = np.mean(movements)
    #             movement_scores['parts_movement'][part] = avg_movement
                
    #             if avg_movement < THRESHOLDS['pose']['slight']:
    #                 part_score = 0.0
    #             elif avg_movement < THRESHOLDS['pose']['significant']:
    #                 part_score = 0.5
    #             else:
    #                 part_score = 1.0
                
    #             pose_score += part_score * part_weights.get(part, 0.0)
        
    #     confidence_score += pose_score * 0.5  # Add pose score (50% weight)
        
    #     # Update movement scores for visualization
    #     movement_scores.update({
    #         'bbox_movement': avg_bbox_movement,
    #         'size_change': avg_size_change,
    #         'pose_movement': pose_score,
    #         'overall_score': confidence_score
    #     })
        
    #     # Determine working status based on weighted score
    #     # Require higher confidence for "working" classification
    #     is_working = confidence_score >= 0.6  # Increased threshold
        
    #     # Additional check for sustained movement
    #     if is_working:
    #         # Check if movement has been consistent across frames
    #         movement_consistency = np.std(movements_over_time['bbox']) / (np.mean(movements_over_time['bbox']) + 1e-6)
    #         if movement_consistency > 1.5:  # High variance in movement
    #             is_working = False
        
    #     return is_working, movement_scores

    # def _determine_working_status(self, track: Dict, curr_detection: PersonDetection) -> Tuple[bool, Dict]:
    #     """
    #     Determine if a person is working based on movement analysis across multiple frames.
    #     Uses weighted scoring system for different types of movements.
    #     """
    #     NUM_HISTORY_FRAMES = 5  # Reduced from 10 to be more responsive
        
    #     movement_scores = {
    #         'bbox_movement': 0.0,
    #         'pose_movement': 0.0,
    #         'size_change': 0.0,
    #         'parts_movement': {},
    #         'overall_score': 0.0
    #     }
        
    #     # Return early if not enough history
    #     if len(track['history']) < 2:  # Changed from NUM_HISTORY_FRAMES to 2 for quicker response
    #         return False, movement_scores
        
    #     # Get previous and current detection
    #     prev_detection = track['history'][-2]
        
    #     # Calculate basic movements
    #     bbox_movement = self._calculate_bbox_movement(prev_detection.bbox, curr_detection.bbox)
    #     size_change = self._calculate_size_change(prev_detection.bbox, curr_detection.bbox)
        
    #     # Updated thresholds - made more sensitive
    #     THRESHOLDS = {
    #         'bbox': {
    #             'slight': 3.0,    # Reduced from 5.0
    #             'significant': 8.0 # Reduced from 15.0
    #         },
    #         'size': {
    #             'slight': 0.03,   # Reduced from 0.05
    #             'significant': 0.1 # Reduced from 0.15
    #         },
    #         'pose': {
    #             'slight': 0.01,   # Reduced from 0.02
    #             'significant': 0.05 # Reduced from 0.08
    #         }
    #     }
        
    #     # Initialize confidence score
    #     confidence_score = 0.0
        
    #     # Score bbox movement (30% weight)
    #     if bbox_movement > THRESHOLDS['bbox']['significant']:
    #         confidence_score += 0.3
    #     elif bbox_movement > THRESHOLDS['bbox']['slight']:
    #         confidence_score += 0.15
        
    #     # Score size change (20% weight)
    #     if size_change > THRESHOLDS['size']['significant']:
    #         confidence_score += 0.2
    #     elif size_change > THRESHOLDS['size']['slight']:
    #         confidence_score += 0.1
        
    #     # Score pose movements (50% weight)
    #     if prev_detection.landmarks and curr_detection.landmarks:
    #         pose_movements = self._calculate_pose_movement(
    #             prev_detection.landmarks, curr_detection.landmarks)
            
    #         # Updated part weights to focus more on arms and torso
    #         part_weights = {
    #             'arms': 0.4,    # Increased from 0.3
    #             'legs': 0.1,    # Decreased from 0.2
    #             'torso': 0.2,   # Increased from 0.1
    #             'head': 0.1     # Kept the same
    #         }
            
    #         for part, movement in pose_movements['parts'].items():
    #             movement_scores['parts_movement'][part] = movement
                
    #             if movement > THRESHOLDS['pose']['significant']:
    #                 confidence_score += part_weights.get(part, 0.0)
    #             elif movement > THRESHOLDS['pose']['slight']:
    #                 confidence_score += part_weights.get(part, 0.0) * 0.5
        
    #     # Update movement scores for visualization
    #     movement_scores.update({
    #         'bbox_movement': bbox_movement,
    #         'size_change': size_change,
    #         'overall_score': confidence_score
    #     })
        
    #     # Lower the working threshold
    #     is_working = confidence_score >= 0.3  # Reduced from 0.6
        
    #     return is_working, movement_scores

    def _determine_working_status(self, track: Dict, curr_detection: PersonDetection) -> Tuple[bool, Dict]:
        """
        Determine if a person is working based on movement analysis across multiple frames.
        Only significant movements are considered as working activity.
        """
        movement_scores = {
            'bbox_movement': 0.0,
            'pose_movement': 0.0,
            'size_change': 0.0,
            'parts_movement': {},
            'overall_score': 0.0
        }
        
        # Require at least 3 frames of history for better movement analysis
        if len(track['history']) < 3:
            return False, movement_scores
        
        # Get previous and current detection
        prev_detection = track['history'][-2]
        
        # Calculate basic movements
        bbox_movement = self._calculate_bbox_movement(prev_detection.bbox, curr_detection.bbox)
        size_change = self._calculate_size_change(prev_detection.bbox, curr_detection.bbox)
        
        # Stricter thresholds - only significant movements count
        THRESHOLDS = {
            'bbox': {
                'working': 4.0,    # Significant movement threshold in pixels
            },
            'size': {
                'working': 0.12,    # Significant size change threshold
            },
            'pose': {
                'working': 0.02,    # Significant pose movement threshold
            }
        }
        
        # Initialize confidence score
        confidence_score = 0.0
        
        # Score bbox movement (30% weight) - only count significant movements
        if bbox_movement > THRESHOLDS['bbox']['working']:
            confidence_score += 0.3
        
        # Score size change (20% weight) - only count significant changes
        if size_change > THRESHOLDS['size']['working']:
            confidence_score += 0.2
        
        # Score pose movements (50% weight)
        if prev_detection.landmarks and curr_detection.landmarks:
            pose_movements = self._calculate_pose_movement(
                prev_detection.landmarks, curr_detection.landmarks)
            
            # Focus heavily on work-related body parts
            part_weights = {
                'arms': 0.3,     # Heavy weight on arm movement
                'torso': 0.2,    # Moderate weight on torso movement
                'legs': 0.4,     # Less weight on leg movement
                'head': 0.1      # !Ignore head movement for work detection
            }
            
            for part, movement in pose_movements['parts'].items():
                movement_scores['parts_movement'][part] = movement
                
                # Only count movements above the working threshold
                if movement > THRESHOLDS['pose']['working']:
                    confidence_score += part_weights.get(part, 0.0)
        
        # Update movement scores for visualization
        movement_scores.update({
            'bbox_movement': bbox_movement,
            'size_change': size_change,
            'overall_score': confidence_score
        })
        
        # Require a higher confidence score for working status
        is_working = confidence_score >= 0.5  # Increased threshold
        
        # Additional check: require sustained movement
        if is_working and len(track['history']) >= 3:
            # Check if movement has been consistent for the last 3 frames
            recent_movements = [
                self._calculate_bbox_movement(
                    track['history'][i-1].bbox,
                    track['history'][i].bbox
                )
                for i in range(len(track['history'])-1, len(track['history'])-3, -1)
            ]
            
            # If any recent movement is below threshold, consider as not working
            if any(move < THRESHOLDS['bbox']['working']/2 for move in recent_movements):
                is_working = False
        
        return is_working, movement_scores



    def process_frame(self, frame: np.ndarray) -> Dict[int, bool]:
        """
        Process a frame and return working status for each person.
        Updated to maintain history for better movement analysis.
        """
        height, width = frame.shape[:2]
        
        # YOLO detection
        yolo_results = self.yolo_model(frame, classes=[0])
        
        # Process detections
        current_detections = []
        for result in yolo_results:
            boxes = result.boxes
            for box in boxes:
                if box.conf[0] > self.min_detection_confidence:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1
                    
                    # Create detection object
                    detection = PersonDetection(
                        bbox=(x1, y1, w, h),
                        confidence=float(box.conf[0])
                    )
                    
                    # Process pose if needed
                    roi = frame[y1:y2, x1:x2]
                    if roi.size > 0:
                        pose_results = self.pose.process(roi)
                        if pose_results.pose_landmarks:
                            landmarks = []
                            for landmark in pose_results.pose_landmarks.landmark:
                                landmarks.append(Landmark(
                                    x=landmark.x,
                                    y=landmark.y,
                                    z=landmark.z,
                                    visibility=landmark.visibility
                                ))
                            detection.landmarks = landmarks
                    
                    current_detections.append(detection)
        
        # Match detections to tracks
        matched_tracks = self._match_detections_to_tracks(current_detections, frame.shape[:2])
        
        # Update tracks and determine working status
        work_status = {}
        current_time = time.time()
        
        for track_id, detection in matched_tracks.items():
            track = self.tracks[track_id]
            
            # Update track history
            track['history'].append(detection)
            if len(track['history']) > self.time_window:
                track['history'] = track['history'][-self.time_window:]
            
            # Update last seen bbox and landmarks
            track['last_bbox'] = detection.bbox
            track['last_landmarks'] = detection.landmarks
            track['last_seen'] = current_time
            
            # Determine working status
            is_working, movement_scores = self._determine_working_status(track, detection)
            
            work_status[track_id] = {
                'is_working': is_working,
                'movement_scores': movement_scores
            }
            
            track['working_status'] = is_working
            track['last_movement_scores'] = movement_scores
        
        # Clean up old tracks
        current_tracks = list(self.tracks.keys())
        for track_id in current_tracks:
            if current_time - self.tracks[track_id]['last_seen'] > 1.0:  # Remove after 1 second
                del self.tracks[track_id]
        
        return work_status


# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = MultiPersonWorkDetector(
        movement_threshold=8.0,           # Reduced from 15.0 --
        pose_movement_threshold=0.05,     # Reduced from 0.08 --
        size_change_threshold=0.1,        # Reduced from 0.15 -- 
        time_window=5,                    # Reduced from 10 ..
        min_detection_confidence=0.6,     # Kept the same ..
        min_visibility_threshold=0.4,     # Kept the same ..
        working_confidence_threshold=0.3   # Reduced from 0.6 --
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
            # if frame_count % 30 == 0:
            #     print(f"\nFrame {frame_count}:")
            #     for track_id, is_working in work_status.items():
            #         print(f"Person {track_id}: {'Working' if is_working else 'Not Working'}")
            
            
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