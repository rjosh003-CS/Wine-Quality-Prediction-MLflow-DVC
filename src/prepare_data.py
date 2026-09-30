# src/prepare_data.py

import pandas as pd
from pathlib import Path

# 1. Load the wine quality dataset straight from MLflow's official repository
data_url = "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
df = pd.read_csv(data_url, sep=";")

# 2. Define the target directory and file path using pathlib (capital 'P')
cwd = Path.cwd()
data_dir = cwd / "data"

# Ensure the 'data' directory exists before saving
data_dir.mkdir(parents=True, exist_ok=True)

file_path = data_dir / "winequality-red.csv"

# 3. Save the dataframe to the full CSV file path (excluding index column to keep it clean)
df.to_csv(file_path, index=False)

print(f"Dataset successfully downloaded and saved to: {file_path}")