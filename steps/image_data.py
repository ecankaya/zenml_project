import numpy as np
from zenml import step
from zenml.steps import Output
import tensorflow as tf


@step
def training_data_loader() -> Output(
    x_train=np.ndarray,
    x_test=np.ndarray,
    y_train=np.ndarray,
    y_test=np.ndarray,
):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = x_train / 255, x_test / 255
    return x_train, x_test, y_train, y_test
