from zenml import pipeline
from steps import training_data_loader, tf_trainer, evaluator


@pipeline(enable_cache=False)
def training_pipeline(
):
    """Train, evaluate, and deploy a model."""
    x_train, x_test, y_train, y_test = training_data_loader()
    model = tf_trainer(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    test_acc = evaluator(x_test=x_test, y_test=y_test, model=model)


training_pipeline()
