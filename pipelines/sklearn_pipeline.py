from zenml import pipeline
from zenml.integrations.mlflow.steps.mlflow_registry import mlflow_register_model_step

from steps.flower_predictions import infer
from steps.score import test_score
from steps.sklearn_trainer import svc_trainer_mlflow
from steps.sklearn_dataloader import training_data_loader
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step


@pipeline(enable_cache=False)
def sklearn_pipeline():
    x_train, x_test, y_train, y_test = infer()
    # x_train, x_test, y_train, y_test = training_data_loader()
    model = svc_trainer_mlflow(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    accuracy = test_score(x_test, y_test, model)
    model_register = mlflow_register_model_step(model, "iris-model")
    deployment = mlflow_model_deployer_step(model)


sklearn_pipeline()
