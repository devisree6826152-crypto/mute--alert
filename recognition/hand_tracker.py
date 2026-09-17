try:
    import cv2
except ImportError:
    cv2 = None

import numpy as np

try:
    import mediapipe as mp
except ImportError:
    mp = None

class HandTracker:
    def __init__(self, static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands if (mp and hasattr(mp, 'solutions')) else None
        self.mp_drawing = mp.solutions.drawing_utils if (mp and hasattr(mp, 'solutions')) else None
        if self.mp_hands and cv2:
            try:
                self.hands = self.mp_hands.Hands(
                    static_image_mode=static_image_mode,
                    max_num_hands=max_num_hands,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence
                )
            except Exception:
                self.hands = None
        else:
            self.hands = None

    def process_frame(self, frame):
        if frame is None:
            return None, None, None, False

        annotated_frame = frame.copy() if hasattr(frame, 'copy') else frame
        if not self.hands or not cv2:
            # Fallback dummy features if cv2 or mediapipe unavailable
            return annotated_frame, None, None, False

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if not results or not results.multi_hand_landmarks:
            return annotated_frame, None, None, False

        hand_landmarks = results.multi_hand_landmarks[0]
        if self.mp_drawing:
            self.mp_drawing.draw_landmarks(
                annotated_frame, 
                hand_landmarks, 
                self.mp_hands.HAND_CONNECTIONS
            )

        # Extract normalized 63 coordinate features (21 landmarks x 3 xyz) + 10 pairwise distance features = 73 features
        raw_coords = []
        for lm in hand_landmarks.landmark:
            raw_coords.extend([lm.x, lm.y, lm.z])

        # Normalize coordinates relative to wrist (landmark 0)
        coords_arr = np.array(raw_coords).reshape(-1, 3)
        wrist = coords_arr[0]
        norm_coords = (coords_arr - wrist).flatten()

        # Additional scale invariance (distance between wrist and middle finger MCP 0 -> 9)
        ref_dist = np.linalg.norm(coords_arr[9] - coords_arr[0])
        if ref_dist > 0:
            norm_coords = norm_coords / ref_dist

        # Compute key finger joint distances (10 extra features)
        finger_tips = [4, 8, 12, 16, 20]
        wrist_pt = norm_coords[:3]
        extra_dists = []
        for tip_idx in finger_tips:
            tip_pt = norm_coords[tip_idx*3 : tip_idx*3 + 3]
            dist = np.linalg.norm(tip_pt - wrist_pt)
            extra_dists.append(dist)
        
        # Inter-finger tip distances
        for i in range(len(finger_tips)-1):
            pt1 = norm_coords[finger_tips[i]*3 : finger_tips[i]*3 + 3]
            pt2 = norm_coords[finger_tips[i+1]*3 : finger_tips[i+1]*3 + 3]
            extra_dists.append(np.linalg.norm(pt1 - pt2))

        features = np.concatenate([norm_coords, np.array(extra_dists)])
        raw_landmarks_list = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]

        return annotated_frame, features, raw_landmarks_list, True
