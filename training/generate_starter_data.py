import os
import pandas as pd
import numpy as np

def generate_starter_dataset(csv_path='data/signs.csv', num_samples_per_class=35):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    classes = ["HELLO", "YES", "NO", "THANK YOU", "PLEASE", "HELP", "GOOD", "STOP", "I LOVE YOU", "A", "B", "C"]
    
    np.random.seed(42)
    rows = []
    num_features = 73

    for idx, cls in enumerate(classes):
        base_features = np.random.uniform(-0.5, 0.5, num_features)
        base_features[idx % num_features] += 1.5

        for _ in range(num_samples_per_class):
            noise = np.random.normal(0, 0.08, num_features)
            sample = base_features + noise
            rows.append([cls] + list(sample))

    cols = ['label'] + [f'f_{i}' for i in range(num_features)]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(csv_path, index=False)
    print(f"Starter dataset generated with {len(df)} samples across {len(classes)} classes at {csv_path}")
    return df

if __name__ == '__main__':
    generate_starter_dataset()
