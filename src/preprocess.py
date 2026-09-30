import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import os
from pathlib import Path

# Load params
with open("params.yaml") as f:
    params = yaml.safe_load(f)

# Load raw data tracked by DVC
df = pd.read_csv("data/winequality-red.csv", sep=",")
df.columns = df.columns.str.strip()

print(f"df columns list: {df.columns.tolist()}")

# Split dataset
train, test = train_test_split(
    df,
    test_size=params["base"]["test_size"],
    random_state=params["base"]["random_state"]
)

# Save to processed folder
processed_folder_path = Path("data/processed")
os.makedirs(processed_folder_path, exist_ok=True)
train.to_csv(processed_folder_path / "train.csv", index=False)
test.to_csv(processed_folder_path / "test.csv", index=False)
print("Data preprocessing and splitting completed sucessfully!")
