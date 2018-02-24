# import the necessary packages
from skimage.segmentation import slic, mark_boundaries
from skimage.io import imread, imsave
from skimage import img_as_ubyte, img_as_float
import numpy as np

image = imread('test.png')
segments = slic(image, n_segments=300)
image = mark_boundaries(np.zeros(image.shape), segments)
for x in image[:, :, 0]:
    print(x)
    break
imsave('superpixel.png', image)
