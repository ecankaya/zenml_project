import mlflow
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.svm import SVC
from zenml import step
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker


@step(enable_cache=False, experiment_tracker=experiment_tracker.name)
def svc_trainer_mlflow(
        x_train: pd.DataFrame,
        y_train: pd.Series,
        gamma=0.01,
) -> ClassifierMixin:
    """Train a sklearn SVC classifier and log to MLflow."""
    mlflow.sklearn.autolog()  # log all model hparams and metrics to MLflow
    model = SVC(gamma=gamma)
    model.fit(x_train.to_numpy(), y_train.to_numpy())
    return model
