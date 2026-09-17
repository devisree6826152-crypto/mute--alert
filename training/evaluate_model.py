import os
import joblib

def evaluate_active_model(model_path='models/sign_model.pkl'):
    if not os.path.exists(model_path):
        return {'model_loaded': False, 'message': 'No trained model found.'}

    data = joblib.load(model_path)
    return {
        'model_loaded': True,
        'accuracy': data.get('accuracy', 0.0),
        'classes': data.get('classes', []),
        'metrics': data.get('metrics', {}),
        'trained_at': data.get('trained_at', 'N/A')
    }
