import pandas as pd
import numpy as np
import yaml
import joblib
import os
from pathlib import Path

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import ElasticNet
import mlflow
import mlflow.sklearn

with open("params.yaml") as f:
    params = yaml.safe_load(f)

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    alpha = params["train"]["alpha"]
    l1_ratio = params["train"]["l1_ratio"]

    # Load the processed data
    processed_data_path = Path("data/processed")
    train = pd.read_csv(processed_data_path /"train.csv")
    test = pd.read_csv(processed_data_path / "test.csv")

    target = "quality"
    train_x = train.drop(target, axis = 1)
    test_x = test.drop(target, axis = 1)
    train_y = train[[target]]
    test_y = test[[target]]

    mlflow.set_experiment("DVC_MLflow_Local_Pipeline")

    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)
        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        # Log to MLflow
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        # Save model locally for DVC output tracking
        os.makedirs("models", exist_ok=True)
        joblib.dump(lr, "models/model.pkl")
        mlflow.sklearn.log_model(
            lr,
            name = "model"
        )
        print("Pipeline execution and MLflow Logging completed successfully!")