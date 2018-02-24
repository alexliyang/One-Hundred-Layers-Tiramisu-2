# import the necessary packages
from skimage.segmentation import slic, mark_boundaries
from skimage.io import imread, imsave
from skimage import img_as_ubyte, img_as_float
import numpy as np
import argparse
import os
import timeit

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--directory', required=True, help='Path of images')
directory = vars(ap.parse_args())['directory']
directory = os.path.abspath(directory)
save_directory = os.path.join(directory, 'superpixel')

if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# profile
start = timeit.default_timer()
count = 0

for filename in os.listdir(directory):
    if filename.endswith('.png'):
        count += 1

        image = imread(os.path.join(directory, filename))
        segments = slic(image, n_segments=100)
        channel1 = mark_boundaries(np.zeros(image.shape), segments, color=(1, 1, 1))[:, :, 0]
        segments = slic(image, n_segments=200)
        channel2 = mark_boundaries(np.zeros(image.shape), segments, color=(1, 1, 1))[:, :, 0]
        segments = slic(image, n_segments=300)
        channel3 = mark_boundaries(np.zeros(image.shape), segments, color=(1, 1, 1))[:, :, 0]
        image = np.dstack((channel1, channel2, channel3))
        imsave(os.path.join(save_directory, filename), image)

stop = timeit.default_timer()
duration = stop - start
print('total spent ' + str(duration) + 's')
duration /= count
print(str(duration) + 's per image')
