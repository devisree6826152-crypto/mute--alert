import os
import csv
import joblib
import numpy as np
from datetime import datetime

def train_sign_model(csv_path='data/signs.csv', model_path='models/sign_model.pkl'):
    if not os.path.exists(csv_path):
        from training.generate_starter_data import generate_starter_dataset
        generate_starter_dataset(csv_path=csv_path)

    labels = []
    features_list = []
    if os.path.exists(csv_path):
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if row and len(row) > 1:
                    labels.append(row[0].strip().upper())
                    features_list.append([float(x) for x in row[1:]])

    if len(labels) < 5:
        return {'success': False, 'error': 'Not enough samples to train model.'}

    X = np.array(features_list, dtype=np.float64)
    y = np.array(labels, dtype=str)

    unique_classes = sorted(list(set(y)))
    if len(unique_classes) < 2:
        return {'success': False, 'error': 'At least 2 sign classes are required to train classification model.'}

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=120, max_depth=18, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, average='macro', zero_division=0))
    rec = float(recall_score(y_test, y_pred, average='macro', zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average='macro', zero_division=0))
    cm = confusion_matrix(y_test, y_pred, labels=unique_classes).tolist()

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    model_payload = {
        'model': clf,
        'accuracy': acc,
        'classes': unique_classes,
        'trained_at': now_str,
        'metrics': {
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classes': unique_classes
        }
    }

    joblib.dump(model_payload, model_path)
    print(f"Model trained successfully. Accuracy: {acc*100:.1f}% on {len(unique_classes)} classes.")

    return {
        'success': True,
        'accuracy': acc,
        'classes': unique_classes,
        'trained_at': now_str,
        'message': f"Random Forest Model trained successfully with {acc*100:.1f}% accuracy!"
    }

if __name__ == '__main__':
    train_sign_model()
