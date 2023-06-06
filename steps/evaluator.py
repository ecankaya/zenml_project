from zenml import step
import numpy as np
import tensorflow as tf


@step
def evaluator(
        x_test: np.ndarray,
        y_test: np.ndarray,
        model: tf.keras.Model,
) -> float:
    """Calculate the accuracy on the test set"""
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {accuracy}")
    return accuracy
