from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

print('mlflow ui --backend-store-uri="' + get_tracking_uri() + '"')