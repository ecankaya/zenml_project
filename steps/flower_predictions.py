import numpy as np
import pandas as pd
from zenml import step
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from zenml.client import Client
from zenml.services import BaseService
from zenml.steps import Output


@step
def infer() -> (Output(
    x_train=pd.DataFrame,
    x_test=pd.DataFrame,
    y_train=pd.Series,
    y_test=pd.Series
)):
    iris = load_iris(as_frame=True)
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, shuffle=True, random_state=42
    )

    return x_train, x_test, y_train, y_test


@step
def prediction_service() -> BaseService:
    client = Client()
    model_deployer = client.active_stack.model_deployer
    services = model_deployer.find_model_server(
        pipeline_name="sklearn_pipeline",
        running=True,
    )
    service = services[0]
    return service


@step
def predict(
        data: pd.DataFrame
) -> Output(predictions=pd.DataFrame):
    client = Client()
    model_deployer = client.active_stack.model_deployer
    services = model_deployer.find_model_server(
        pipeline_name="sklearn_pipeline",
        running=True,
    )
    service = services[0]
    inference = data
    inference = inference.to_numpy()
    prediction = service.predict(inference)
    # prediction = prediction.argmax(axis=-1)
    print(prediction)
    prediction = pd.DataFrame(prediction, columns=["series"])
    return prediction
