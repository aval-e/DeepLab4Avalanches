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

def overlay_and_plot_avalanches_by_certainty(image, aval_images):
    """ Plots image and overlays avalanches in different colors according to their certainty

    :param image: background satellite image as numpy array
    :param aval_images: list of 3 rasterised avalanche shapes from certain to uncertain
    """
    i = image[:, :, 0:3]

    # overlay certain avalanches in green
    i[:, :, 0] -= 0.4 * aval_images[0]
    i[:, :, 1] += 0.4 * aval_images[0]
    i[:, :, 2] -= 0.4 * aval_images[0]

    # overlay estimated avalanches in orange
    i[:, :, 0] += 0.4 * aval_images[1]
    i[:, :, 1] += 0.1 * aval_images[1]
    i[:, :, 2] -= 0.4 * aval_images[1]

    # overlay guessed avalanches in red
    i[:, :, 0] += 0.4 * aval_images[2]
    i[:, :, 1] -= 0.4 * aval_images[2]
    i[:, :, 2] -= 0.4 * aval_images[2]

    plt.imshow(image)
    plt.show()
