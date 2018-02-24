# import the necessary packages
from skimage.segmentation import slic, mark_boundaries
from skimage.io import imread, imsave
from skimage import img_as_ubyte, img_as_float
import numpy as np

image = imread('test.png')
segments = slic(image, n_segments=100)
channel1 = mark_boundaries(np.zeros(image.shape), segments, color=(1, 1, 1))[:, :, 0]
segments = slic(image, n_segments=200)
channel2 = mark_boundaries(np.zeros(image.shape), segments, color=(1, 1, 1))[:, :, 0]
segments = slic(image, n_segments=300)
channel3 = mark_boundaries(np.zeros(image.shape), segments, color=(1, 1, 1))[:, :, 0]
image = np.dstack((channel1, channel2, channel3))

imsave('superpixel.png', image)
