import pandas as pd
from zenml import step
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
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
