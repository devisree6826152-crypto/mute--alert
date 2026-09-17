import os
import pandas as pd
from database.database import update_class_sample_count

def save_sample_to_csv(sign_name, features, csv_path='data/signs.csv'):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    sign_name = sign_name.upper().strip()

    row_data = [sign_name] + list(features)
    columns = ['label'] + [f'f_{i}' for i in range(len(features))]

    if not os.path.exists(csv_path):
        df = pd.DataFrame([row_data], columns=columns)
        df.to_csv(csv_path, index=False)
    else:
        df_existing = pd.read_csv(csv_path)
        df_new = pd.DataFrame([row_data], columns=columns)
        df_combined = pd.concat([df_existing, df_new], ignore_index=False)
        df_combined.to_csv(csv_path, index=False)

    df_current = pd.read_csv(csv_path)
    count = int((df_current.iloc[:, 0].astype(str).str.upper() == sign_name).sum())
    update_class_sample_count(sign_name, count)
    return count
