# import the necessary packages
from skimage.segmentation import slic, mark_boundaries
from skimage.io import imread, imsave

image = imread('test.png')
segments = slic(image, n_segments=300)
image = mark_boundaries(image, segments)
imsave('superpixel.png', image)
