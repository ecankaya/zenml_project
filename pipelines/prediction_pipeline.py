from zenml import pipeline

from steps.inference import inference_data_loader
from steps.prediction_server import predictor, prediction_service_loader


@pipeline(enable_cache=False)
def prediction_pipeline():
    data = inference_data_loader(),
    service = prediction_service_loader(),
    prediction = predictor(service,data)


prediction_pipeline()
