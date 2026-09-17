import os
import csv
import numpy as np

def generate_starter_dataset(csv_path='data/signs.csv', num_samples_per_class=35):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    classes = ["HELLO", "YES", "NO", "THANK YOU", "PLEASE", "HELP", "GOOD", "STOP", "I LOVE YOU", "A", "B", "C"]
    
    np.random.seed(42)
    num_features = 73

    cols = ['label'] + [f'f_{i}' for i in range(num_features)]
    rows = []

    for idx, cls in enumerate(classes):
        base_features = np.random.uniform(-0.5, 0.5, num_features)
        base_features[idx % num_features] += 1.5

        for _ in range(num_samples_per_class):
            noise = np.random.normal(0, 0.08, num_features)
            sample = base_features + noise
            rows.append([cls] + list(sample))

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        writer.writerows(rows)

    print(f"Starter dataset generated with {len(rows)} samples across {len(classes)} classes at {csv_path}")
    return len(rows)

if __name__ == '__main__':
    generate_starter_dataset()
