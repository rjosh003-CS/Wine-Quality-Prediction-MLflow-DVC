import pandas as pd
import numpy as np
import yaml
import joblib
import os
from pathlib import Path

import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import ElasticNet

import mlflow
import mlflow.sklearn

with open("params.yaml") as f:
    params = yaml.safe_load(f)


def objective(trial):
    # Suggest hyperparameters to explore
    alpha = trial.suggest_float("alpha", 0.01, 1.0)
    l1_ratio = trial.suggest_float("l1_ratio", 0.1, 1.0)

    train = pd.read_csv("data/processed/train.csv")
    test = pd.read_csv("data/processed/test.csv")

    target = "quality"
    train_x, train_y = train.drop(target, axis=1), train[[target]]
    test_x, test_y = test.drop(target, axis=1), test[[target]]

    with mlflow.start_run(nested=True):
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        preds = lr.predict(test_x)
        rmse = np.sqrt(mean_squared_error(test_y, preds))

        # Log parameters and metrics for this trial
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)

        return rmse

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def optuna_tuning():
    mlflow.set_experiment("DVC_MLflow_Optimization")
    with mlflow.start_run(run_name="Optuna_Optimization_Parent"):
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=20)

        # Log best parameters found
        mlflow.log_params(study.best_params)
        print("Best Hyperparameters:", study.best_params)

        # ----- Retrain Final Model With Best Params for DVC OUTPUT -----
        best_alpha = study.best_params["alpha"]
        best_l1_ratio = study.best_params["l1_ratio"]

        train = pd.read_csv("data/processed/train.csv")
        test = pd.read_csv("data/processed/test.csv")

        target = "quality"
        train_x, train_y = train.drop(target, axis=1), train[[target]]
        test_x, test_y = test.drop(target, axis=1), test[[target]]

        # Train final model on best parameters
        best_lr = ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio, random_state=42)
        best_lr.fit(train_x, train_y)

        # Save model locally so DVC finds 'models/model.pkl'
        os.makedirs("models", exist_ok=True)
        joblib.dump(best_lr, "models/model.pkl")

        # Log final model artifact to MLflow
        mlflow.sklearn.log_model(best_lr, name="model")
        print("Final best model trained and saved to models/model.pkl sucessfully!")


def custom_hyper_parameter_tuning():
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
        # loging the model in the mlflow
        mlflow.sklearn.log_model(
            lr,
            name = "model"
        )
        print("Pipeline execution and MLflow Logging completed successfully!")


if __name__ == "__main__":
    use_optuna=True
    optuna_tuning() if use_optuna else custom_hyper_parameter_tuning()