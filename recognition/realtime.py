import os
import joblib
import numpy as np
from recognition.hand_tracker import HandTracker

class RealtimeRecognizer:
    def __init__(self, model_path='models/sign_model.pkl'):
        self.model_path = model_path
        self.tracker = HandTracker()
        self.model = None
        self.classes = []
        self.history = []
        self.load_model()

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                data = joblib.load(self.model_path)
                self.model = data.get('model')
                self.classes = data.get('classes', [])
                print(f"Model loaded successfully with {len(self.classes)} classes.")
            except Exception as e:
                print(f"Error loading model: {e}")
                self.model = None
        else:
            print("Model file not found.")
            self.model = None

    def predict_frame(self, frame, confidence_threshold=0.65):
        annotated_frame, features, raw_landmarks, hand_found = self.tracker.process_frame(frame)

        if not hand_found or features is None:
            return {
                'hand_detected': False,
                'prediction': 'No Hand Detected',
                'confidence': 0.0,
                'stable_prediction': None,
                'raw_landmarks': [],
                'annotated_frame': annotated_frame
            }

        if self.model is None:
            return {
                'hand_detected': True,
                'prediction': 'MODEL NOT TRAINED',
                'confidence': 0.0,
                'stable_prediction': None,
                'raw_landmarks': raw_landmarks,
                'annotated_frame': annotated_frame
            }

        probs = self.model.predict_proba([features])[0]
        max_idx = np.argmax(probs)
        confidence = float(probs[max_idx])
        prediction = self.classes[max_idx] if confidence >= confidence_threshold else "NO SIGN"

        # Debounce for stable prediction string
        self.history.append(prediction)
        if len(self.history) > 10:
            self.history.pop(0)

        stable_pred = None
        if len(self.history) >= 5:
            recent = self.history[-5:]
            if recent.count(recent[0]) == 5 and recent[0] not in ["NO SIGN", "No Hand Detected", "MODEL NOT TRAINED"]:
                stable_pred = recent[0]

        return {
            'hand_detected': True,
            'prediction': prediction,
            'confidence': confidence,
            'stable_prediction': stable_pred,
            'raw_landmarks': raw_landmarks,
            'annotated_frame': annotated_frame
        }
