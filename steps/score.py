import pandas as pd
from sklearn.base import ClassifierMixin
from zenml import step
import numpy as np
import tensorflow as tf


@step
def test_score(
        x_test: pd.DataFrame,
        y_test: pd.Series,
        model: ClassifierMixin,
) -> float:
    """Calculate the accuracy on the test set"""
    accuracy = model.score(x_test.to_numpy(), y_test.to_numpy())
    return accuracy
