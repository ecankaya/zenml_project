import cv2
import numpy as np
from matplotlib import pyplot as plt
from zenml import step
from zenml.steps import Output

img_path = '/home/emre/PycharmProjects/pythonProject/frog.png'  # set path to your image


@step(enable_cache=False)
def inference_data_loader() -> Output(data=np.ndarray):
    """Load some (random) inference data."""

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # plt.imshow(img, cmap=plt.cm.binary)
    # plt.show()
    data = np.array([img]) / 255
    return data
