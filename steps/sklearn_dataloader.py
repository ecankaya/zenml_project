import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from zenml import step
from zenml.steps import Output


@step
def training_data_loader() -> (
        Output(
            X_train=pd.DataFrame,
            X_test=pd.DataFrame,
            y_train=pd.Series,
            y_test=pd.Series,
        )
):
    """Load the iris dataset as tuple of Pandas DataFrame / Series."""
    iris = load_iris(as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, shuffle=True, random_state=42
    )
    X__train = X_train[(X_train["sepal length (cm)"] < 6.5) & (X_train["sepal width (cm)"] > 3)]
    y_train = y_train[(X_train["sepal length (cm)"] < 6.5) & (X_train["sepal width (cm)"] > 3)]
    X_train = X__train
    return X_train, X_test, y_train, y_test
