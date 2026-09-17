import os
import csv
from database.database import update_class_sample_count

def save_sample_to_csv(sign_name, features, csv_path='data/signs.csv'):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    sign_name = sign_name.upper().strip()

    row_data = [sign_name] + list(features)

    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            cols = ['label'] + [f'f_{i}' for i in range(len(features))]
            writer.writerow(cols)
        writer.writerow(row_data)

    count = 0
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if row and row[0].strip().upper() == sign_name:
                count += 1

    update_class_sample_count(sign_name, count)
    return count
