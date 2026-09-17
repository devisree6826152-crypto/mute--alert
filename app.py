import os
import io
import base64
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify, Response, send_file
from gtts import gTTS

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from PIL import Image
except ImportError:
    Image = None

# Local imports
from database.database import (
    init_db, add_history_record, get_history, 
    delete_history_record, clear_history, get_dataset_stats,
    add_sos_event, get_sos_events, resolve_sos_event,
    add_contact, update_contact, get_contacts, delete_contact
)
from recognition.hand_tracker import HandTracker
from recognition.realtime import RealtimeRecognizer

app = Flask(__name__)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(PROJECT_ROOT, 'data', 'signs.csv')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'sign_model.pkl')

# Initialize DB & recognizer singleton
init_db()

# Global recognizer instance
recognizer = None

def get_recognizer():
    global recognizer
    if recognizer is None:
        recognizer = RealtimeRecognizer(model_path=MODEL_PATH)
    return recognizer

# Ensure starter dataset and trained model exist on app boot
def bootstrap_system():
    try:
        if not os.path.exists(DATASET_PATH):
            print("Initializing starter dataset...")
            from training.generate_starter_data import generate_starter_dataset
            generate_starter_dataset(csv_path=DATASET_PATH)
        if not os.path.exists(MODEL_PATH):
            print("Training initial ML model...")
            from training.train_model import train_sign_model
            train_sign_model(csv_path=DATASET_PATH, model_path=MODEL_PATH)
    except Exception as e:
        print(f"Warning: System bootstrap skipped or failed (likely read-only environment): {e}")

bootstrap_system()

# Helper function to decode base64 image string to numpy image array
def decode_base64_image(base64_str):
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]
    image_bytes = base64.b64decode(base64_str)
    if cv2:
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return frame
    elif Image:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        return np.array(img)
    return None

# Helper function to encode image frame to base64 string
def encode_image_base64(frame):
    if frame is None:
        return ""
    if cv2:
        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        base64_str = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_str}"
    elif Image:
        img = Image.fromarray(frame)
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=80)
        base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_str}"
    return ""


# ---------------------- HTML ROUTE HANDLERS ----------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recognition')
def recognition_page():
    return render_template('recognition.html')

@app.route('/dataset')
def dataset_page():
    return render_template('dataset.html')

@app.route('/training')
def training_page():
    return render_template('training.html')

@app.route('/history')
def history_page():
    return render_template('history.html')


# ---------------------- API ROUTE HANDLERS ----------------------

@app.route('/api/predict_frame', methods=['POST'])
def api_predict_frame():
    """Process a webcam frame sent via JSON base64 from browser JS."""
    try:
        data = request.get_json(force=True)
        base64_image = data.get('frame')
        threshold = float(data.get('threshold', 0.65))

        if not base64_image:
            return jsonify({'success': False, 'error': 'No frame provided'}), 400

        frame = decode_base64_image(base64_image)
        if frame is None:
            return jsonify({'success': False, 'error': 'Invalid frame format'}), 400

        # Mirror frame horizontally for intuitive self-view
        frame = np.fliplr(frame)
        
        # Resize to standard 320x240 for ultra-fast landmark inference
        h, w, _ = frame.shape
        if w > 320:
            if cv2:
                frame = cv2.resize(frame, (320, 240))
            elif Image:
                img = Image.fromarray(frame)
                img = img.resize((320, 240))
                frame = np.array(img)

        rec = get_recognizer()
        result = rec.predict_frame(frame, confidence_threshold=threshold)

        return jsonify({
            'success': True,
            'hand_detected': result['hand_detected'],
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'stable_prediction': result['stable_prediction'],
            'raw_landmarks': result['raw_landmarks']
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/collect_frame', methods=['POST'])
def api_collect_frame():
    """Extract hand landmark features from a frame and append to class dataset."""
    try:
        data = request.get_json(force=True)
        base64_image = data.get('frame')
        sign_name = data.get('sign_name', '').strip()

        if not base64_image or not sign_name:
            return jsonify({'success': False, 'error': 'Missing frame or sign_name'}), 400

        frame = decode_base64_image(base64_image)
        if frame is None:
            return jsonify({'success': False, 'error': 'Invalid image format'}), 400

        frame = np.fliplr(frame)

        # Use HandTracker to extract normalized 73 features
        rec = get_recognizer()
        annotated_frame, features, _, hand_found = rec.tracker.process_frame(frame)

        if not hand_found or features is None:
            return jsonify({
                'success': False, 
                'hand_detected': False,
                'message': 'No hand detected in camera frame. Position hand clearly.'
            }), 400

        from training.collect_data import save_sample_to_csv
        save_sample_to_csv(sign_name, features, csv_path=DATASET_PATH)
        annotated_base64 = encode_image_base64(annotated_frame)

        # Count total samples for this class using python csv module
        import csv
        class_count = 0
        total_dataset_size = 0
        if os.path.exists(DATASET_PATH):
            with open(DATASET_PATH, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if row:
                        total_dataset_size += 1
                        if row[0].strip().upper() == sign_name.upper():
                            class_count += 1

        return jsonify({
            'success': True,
            'hand_detected': True,
            'sign_name': sign_name.upper(),
            'class_sample_count': class_count,
            'total_dataset_size': total_dataset_size,
            'annotated_frame': annotated_base64,
            'message': f"Sample captured for '{sign_name.upper()}' (Total: {class_count})"
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/train', methods=['GET', 'POST'])
def api_train_model():
    """Trigger model retraining and reload active recognizer."""
    try:
        from training.train_model import train_sign_model
        res = train_sign_model(csv_path=DATASET_PATH, model_path=MODEL_PATH)
        if res.get('success'):
            global recognizer
            if recognizer is not None:
                recognizer.load_model()
            else:
                recognizer = RealtimeRecognizer(model_path=MODEL_PATH)
        return jsonify(res)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def api_stats():
    """Return dataset overview, total samples, class counts, and model metrics."""
    try:
        import csv
        stats_db = get_dataset_stats()
        
        total_samples = 0
        class_counts = {}
        if os.path.exists(DATASET_PATH):
            with open(DATASET_PATH, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if row:
                        total_samples += 1
                        cname = row[0].strip().upper()
                        class_counts[cname] = class_counts.get(cname, 0) + 1

        model_loaded = False
        accuracy = 0.0
        classes = []
        metrics = {}
        trained_at = "Not Trained"

        if os.path.exists(MODEL_PATH):
            model_data = joblib.load(MODEL_PATH)
            model_loaded = True
            accuracy = model_data.get('accuracy', 0.0)
            classes = model_data.get('classes', [])
            metrics = model_data.get('metrics', {})
            trained_at = model_data.get('trained_at', 'N/A')

        return jsonify({
            'success': True,
            'total_samples': total_samples,
            'num_classes': len(class_counts),
            'class_counts': class_counts,
            'model_loaded': model_loaded,
            'accuracy': accuracy,
            'classes': classes,
            'metrics': metrics,
            'trained_at': trained_at,
            'db_stats': stats_db
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/history', methods=['GET', 'POST', 'DELETE'])
def api_history():
    """Manage SQLite history records."""
    if request.method == 'GET':
        records = get_history(limit=100)
        return jsonify({'success': True, 'history': records})

    elif request.method == 'POST':
        data = request.get_json(force=True)
        text = data.get('text', '').strip()
        confidence = data.get('confidence', 1.0)
        sign_type = data.get('sign_type', 'sentence')

        if not text:
            return jsonify({'success': False, 'error': 'Empty text'}), 400

        rec_id = add_history_record(text, confidence, sign_type)
        return jsonify({'success': True, 'inserted_id': rec_id, 'message': 'Record saved to history'})

    elif request.method == 'DELETE':
        record_id = request.args.get('id')
        if record_id:
            delete_history_record(int(record_id))
            return jsonify({'success': True, 'message': f'Record {record_id} deleted'})
        else:
            clear_history()
            return jsonify({'success': True, 'message': 'All history cleared'})


# ---------------------- EMERGENCY CONTACTS API ----------------------

@app.route('/api/contacts', methods=['GET', 'POST'])
def api_contacts():
    """Manage user's real emergency contacts."""
    if request.method == 'GET':
        contacts = get_contacts()
        return jsonify({'success': True, 'contacts': contacts})

    elif request.method == 'POST':
        try:
            data = request.get_json(force=True)
            name = (data.get('name') or '').strip()
            phone = (data.get('phone') or '').strip()
            relationship = (data.get('relationship') or 'Contact').strip()

            if not name or not phone:
                return jsonify({'success': False, 'error': 'Name and phone number are required'}), 400

            inserted_id = add_contact(name, phone, relationship)
            return jsonify({
                'success': True,
                'inserted_id': inserted_id,
                'contact': {'id': inserted_id, 'name': name, 'phone': phone, 'relationship': relationship},
                'message': f"Emergency contact '{name}' saved successfully."
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/contacts/<int:contact_id>', methods=['PUT', 'DELETE'])
def api_contact_detail(contact_id):
    """Update or delete a specific emergency contact."""
    if request.method == 'PUT':
        try:
            data = request.get_json(force=True)
            name = (data.get('name') or '').strip()
            phone = (data.get('phone') or '').strip()
            relationship = (data.get('relationship') or 'Contact').strip()

            if not name or not phone:
                return jsonify({'success': False, 'error': 'Name and phone number are required'}), 400

            success = update_contact(contact_id, name, phone, relationship)
            if success:
                return jsonify({'success': True, 'message': f'Contact #{contact_id} updated successfully.'})
            else:
                return jsonify({'success': False, 'error': 'Contact not found or update failed.'}), 444
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

    elif request.method == 'DELETE':
        try:
            success = delete_contact(contact_id)
            if success:
                return jsonify({'success': True, 'message': f'Contact #{contact_id} deleted.'})
            else:
                return jsonify({'success': False, 'error': 'Contact not found.'}), 404
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500


# ---------------------- REAL SOS EMERGENCY API ----------------------

@app.route('/api/sos', methods=['GET', 'POST'])
def api_sos():
    """Manage real SOS emergency events."""
    if request.method == 'GET':
        events = get_sos_events(limit=50)
        return jsonify({'success': True, 'sos_events': events})

    elif request.method == 'POST':
        try:
            data = request.get_json(force=True)
            sos_id = data.get('sos_id')
            if not sos_id:
                return jsonify({'success': False, 'error': 'sos_id is required'}), 400

            res = add_sos_event(data)
            return jsonify({
                'success': True,
                'event_id': res['id'],
                'sos_id': res['sos_id'],
                'is_duplicate': res['is_duplicate'],
                'status': res['status'],
                'message': 'Duplicate SOS request ignored.' if res['is_duplicate'] else 'Real SOS Event logged successfully.'
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/sos/resolve', methods=['PUT'])
def api_sos_resolve():
    """Mark an active SOS event as resolved."""
    try:
        data = request.get_json(force=True)
        sos_id = data.get('sos_id')
        if not sos_id:
            return jsonify({'success': False, 'error': 'sos_id is required'}), 400

        success = resolve_sos_event(sos_id)
        if success:
            return jsonify({'success': True, 'message': f'SOS Event {sos_id} marked as RESOLVED.'})
        else:
            return jsonify({'success': False, 'error': 'SOS Event not found or already resolved.'}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500



@app.route('/api/tts', methods=['POST'])
def api_tts():
    """Convert input text to spoken audio (gTTS fallback)."""
    try:
        data = request.get_json(force=True)
        text = data.get('text', '').strip()

        if not text:
            return jsonify({'success': False, 'error': 'No text provided'}), 400

        tts = gTTS(text=text, lang='en', slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)

        return send_file(fp, mimetype='audio/mpeg', download_name='speech.mp3')
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/video_feed')
def video_feed():
    """Direct OpenCV camera MJPEG stream for desktop preview."""
    def generate():
        cap = cv2.VideoCapture(0)
        rec = get_recognizer()
        while True:
            success, frame = cap.read()
            if not success:
                break
            frame = cv2.flip(frame, 1)
            result = rec.predict_frame(frame)
            ret, buffer = cv2.imencode('.jpg', result['annotated_frame'])
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        cap.release()

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    print("=" * 60)
    print(" SIGNSPEAK AI - REAL-TIME SIGN LANGUAGE SYSTEM")
    print(" Running Flask App on http://127.0.0.1:5000")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=True)
