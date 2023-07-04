import pandas as pd
from zenml import step
from zenml.client import Client
from zenml.steps import Output

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
    prediction = pd.DataFrame(prediction, columns=["series"])
    return prediction