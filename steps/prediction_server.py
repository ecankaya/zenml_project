from zenml.services import BaseService
from zenml import step
from zenml.client import Client
import numpy as np
from zenml.steps import Output
from zenml.integrations.mlflow.services import MLFlowDeploymentService

import cv2
from matplotlib import pyplot as plt
from zenml import step
from zenml.steps import Output


@step(enable_cache=False)
def prediction_service_loader() -> BaseService:
    """Load the model service of our train_evaluate_deploy_pipeline."""
    client = Client()
    model_deployer = client.active_stack.model_deployer
    services = model_deployer.find_model_server(
        pipeline_name="image_training_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",  # geht auch ohne
        running=True,
    )
    service = services[0]
    return service


@step(enable_cache=False)
def predictor(
        data: np.ndarray,
        service: BaseService
) -> Output(predictions=np.int64):
    labels = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    """Run a inference request against a prediction service"""

    service.start(timeout=10)  # should be a NOP if already started
    prediction = service.predict(data)
    prediction = prediction.argmax()
    print(f"Prediction is: {labels[prediction]}")
    return prediction
