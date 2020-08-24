import numpy as np
import matplotlib.pyplot as plt


def viz_sample(sample):
    """ Visualise single sample of the Avalanche dataset"""
    image, aval = sample
    i = image[:, :, 0:3]
    i[:, :, 0] += 0.4 * aval
    i[:, :, 1] -= 0.4 * aval
    i[:, :, 2] -= 0.4 * aval

    plt.imshow(i)
    plt.show()
