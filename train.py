"""
ML Experiment Tracking with MLflow
House Price Prediction on the California Housing dataset.

Trains the same model (GradientBoostingRegressor) 3 times with different
hyperparameters and tracks everything with MLflow:
parameters, metrics, and the trained model (as an artifact).
"""

import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
RANDOM_SEED = 42
TEST_SIZE = 0.2          # 80% train / 20% validation
N_ESTIMATORS = 100       # fixed for all runs, so only max_depth & lr change
EXPERIMENT_NAME = "California-House-Price-Prediction"
TRACKING_URI = "sqlite:///mlflow.db"

# Run name -> hyperparameters (from the task table)
RUN_CONFIGS = [
    {"run_name": "Run 1", "max_depth": 3, "learning_rate": 0.1},
    {"run_name": "Run 2", "max_depth": 5, "learning_rate": 0.05},
    {"run_name": "Run 3", "max_depth": 7, "learning_rate": 0.01},
]


# ----------------------------------------------------------------------
# Part 1: Prepare the dataset
# ----------------------------------------------------------------------
def load_and_split():
    data = fetch_california_housing(as_frame=True)
    X, y = data.data, data.target  # y = median house value (in $100k)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    print(f"Train size: {len(X_train)} | Validation size: {len(X_val)}")
    return X_train, X_val, y_train, y_val


# ----------------------------------------------------------------------
# Parts 2 & 3: Train + track one run
# ----------------------------------------------------------------------
def run_experiment(cfg, X_train, X_val, y_train, y_val):
    with mlflow.start_run(run_name=cfg["run_name"]):
        # 1. Train
        model = GradientBoostingRegressor(
            max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"],
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_SEED,
        )
        model.fit(X_train, y_train)

        # 2. Validation metrics
        preds = model.predict(X_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
        mae = float(mean_absolute_error(y_val, preds))
        r2 = float(r2_score(y_val, preds))

        # 3. Log parameters
        mlflow.log_param("max_depth", cfg["max_depth"])
        mlflow.log_param("learning_rate", cfg["learning_rate"])
        mlflow.log_param("n_estimators", N_ESTIMATORS)  # extra, for clarity

        # 4. Log metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # 5. Log the trained model as an artifact
        signature = infer_signature(X_val, preds)
        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            input_example=X_val.head(5),
            # MLflow 3 saves sklearn models with skops; Tree is safe here
            # because we trained this model ourselves.
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        print(
            f"{cfg['run_name']}: max_depth={cfg['max_depth']}, "
            f"lr={cfg['learning_rate']} -> "
            f"RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}"
        )


# ----------------------------------------------------------------------
# Part 4: Compare runs and select the best (lowest RMSE)
# ----------------------------------------------------------------------
def summarize():
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.rmse ASC"],
    )

    table = runs[
        [
            "tags.mlflow.runName",
            "params.max_depth",
            "params.learning_rate",
            "metrics.rmse",
            "metrics.mae",
            "metrics.r2",
            "run_id",
        ]
    ].rename(
        columns={
            "tags.mlflow.runName": "Run",
            "params.max_depth": "Max Depth",
            "params.learning_rate": "Learning Rate",
            "metrics.rmse": "RMSE",
            "metrics.mae": "MAE",
            "metrics.r2": "R2",
            "run_id": "Run ID",
        }
    )

    print("\n=== Comparison (sorted by RMSE, best first) ===")
    print(table.drop(columns=["Run ID"]).round(4).to_string(index=False))

    best = table.iloc[0]
    print(
        f"\nBest model: {best['Run']} "
        f"(max_depth={best['Max Depth']}, learning_rate={best['Learning Rate']}) "
        f"with RMSE={best['RMSE']:.4f}"
    )
    print(f"Best run ID: {best['Run ID']}")

    # Handy for copying numbers into the README
    table.sort_values("Run").round(4).to_csv("results.csv", index=False)
    print("Saved comparison to results.csv")


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_val, y_train, y_val = load_and_split()

    for cfg in RUN_CONFIGS:
        run_experiment(cfg, X_train, X_val, y_train, y_val)

    summarize()


if __name__ == "__main__":
    main()
