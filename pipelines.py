from zenml import pipeline
from steps import training_data_loader,tf_trainer,evaluator
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
model_deployer=mlflow_model_deployer_step



@pipeline(enable_cache=False)
def training_pipeline(
):
    """Train, evaluate, and deploy a model."""
    x_train, x_test, y_train, y_test = training_data_loader()
    model = tf_trainer(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    test_acc = evaluator(x_test=x_test, y_test=y_test, model=model)

training_pipeline()