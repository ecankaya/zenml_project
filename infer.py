import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from zenml.client import Client

iris = load_iris(as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, shuffle=True, random_state=42
)
#X_train = X_train[["sepal length (cm)","sepal width (cm)"]]
X_train = X_train[(X_train["sepal length (cm)"] < 6) & (X_train["sepal width (cm)"] > 3)]
X_test = X_test[(X_test["sepal length (cm)"] < 6) & (X_test["sepal width (cm)"] > 3)]
y_train = y_train[:35]
y_test = y_test[:35]
print(y_train)



client = Client()
model_deployer = client.active_stack.model_deployer
services = model_deployer.find_model_server(
    pipeline_name="sklearn_pipeline",
    running=True,
)
service = services[0]

Data = X_train.to_numpy()
#print(Data.tolist())
prediction = service.predict(Data)
#prediction = prediction.argmax(axis=-1)
df = pd.DataFrame(prediction, columns=["target"])
print(prediction)
print(df)