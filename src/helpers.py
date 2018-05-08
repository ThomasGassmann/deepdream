import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def plot_image_from_array(a):
    a = np.uint8(np.clip(a, 0, 1) * 255)
    plt.imshow(a)
    plt.show()