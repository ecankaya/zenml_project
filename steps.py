import numpy as np
import tensorflow as tf
from keras import Sequential, layers
from zenml import step
from zenml.steps import Output


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

import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker
mlflow.set_registry_uri()

@step(experiment_tracker=experiment_tracker.name)
def tf_trainer(
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,

) -> tf.keras.Model:
    model = Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # train model

    mlflow.tensorflow.autolog() # log info into mlflow
    model.fit(x_train, y_train, epochs=6
              , validation_data=(x_test, y_test)
              )

    # log additional information to MLflow explicitly if needed
    return model


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


@step
def deployment_trigger(accuracy: float) -> bool:
    """Only deploy if the test accuracy > 50%."""
    return accuracy > 0.5
