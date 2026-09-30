import argparse
import warnings
import sys

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

import mlflow
import mlflow.sklearn

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

# Start MLflow Tracking experiment
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("DVC_MLFlow_Local_Pipeline")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    parser = argparse.ArgumentParser()
    # Fixed: Changed argparse to add_argument
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--l1_ratio", type=float, default=0.5)
    args = parser.parse_args()

    # Load data locally (DVC ensures this file is present and matches the tracked version)
    data_path = "data/winequality-red.csv"
    try:
        # Fixed: Wine quality dataset uses semicolon (;) delimiter
        df = pd.read_csv(data_path, sep=",")

        # Clean up any accidental whitespace in column names
        df.columns = df.columns.str.strip()

        # Debug print to verify columns if it happens again
        print("Loaded columns:", df.columns.tolist())

    except Exception as e:
        print(f"Error Loading dataset from {data_path}: {e}", file=sys.stderr)
        sys.exit(1)

    # Split dataset
    train, test = train_test_split(df, test_size=0.25, random_state=42)
    target_features = ["quality"]
    train_x = train.drop(target_features, axis=1)
    test_x = test.drop(target_features, axis=1)
    train_y = train[target_features]
    test_y = test[target_features]
    

    with mlflow.start_run():
        # Train model - Fixed: ElasticNet takes l1_ratio, not l2_ratio
        lr = ElasticNet(alpha=args.alpha, l1_ratio=args.l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        # Evaluate
        predicted_qualities = lr.predict(test_x)
        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print(f"Model trained with DVC data (alpha={args.alpha}, l1_ratio={args.l1_ratio}):")
        print(f"  RMSE: {rmse}")
        print(f"  R2: {r2}")

        # Log Parameters, metrics, and dataset info to MLflow Tracking
        mlflow.log_param("alpha", args.alpha)
        mlflow.log_param("l1_ratio", args.l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_param("data_file", "winequality-red.csv")

        # Save model using modern MLflow format
        mlflow.sklearn.log_model(lr, name="model")
        print("Pipeline execution and MLflow Logging completed successfully!")