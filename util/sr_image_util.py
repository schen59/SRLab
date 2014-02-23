__author__ = 'Sherwin'

import numpy as np
from PIL import Image
from sr_exception.sr_exception import SRException

INVALID_PATCH_SIZE_ERR = "Invalid patch size %s, patch should be square with odd size."
INVALID_PATCH_DIMENSION = "Invalid patch dimension %s, patch should be square with odd size."

def createSize(size, ratio):
    newSize = map(int, [dimension*ratio for dimension in size])
    return newSize

def gaussianKernel(size=5, sigma=1):
    radius = size / 2
    x, y = np.mgrid[-radius:radius+1, -radius:radius+1]
    unnormalized_kernel = np.exp(-(x**2 + y**2)//2*sigma*sigma)
    return unnormalized_kernel / np.sum(unnormalized_kernel)

def _validPatchSize(patch_size):
    if len(patch_size) != 2:
        return False

    height, width = patch_size
    return height == width and height % 2 != 0

def _validPatchDimension(patch_dimension):
    patch_height = int(patch_dimension**(.5))
    patch_width = patch_dimension / patch_height
    return _validPatchSize([patch_height, patch_width])

def patchify(array, patch_size, interval=1):
    if not _validPatchSize(patch_size):
        raise SRException(INVALID_PATCH_SIZE_ERR % patch_size)

    patch_width = patch_size[0]
    patch_dimension = patch_width * patch_width
    patch_radius = patch_width / 2
    patch_y, patch_x = np.mgrid[-patch_radius:patch_radius+1:interval, -patch_radius:patch_radius+1:interval]
    pad_array = np.pad(array, (patch_radius, patch_radius), 'reflect')
    padded_height, padded_width = np.shape(pad_array)
    patches_y, patches_x = np.mgrid[patch_radius:padded_width-patch_radius, patch_radius:padded_height-patch_radius]
    patches_number = len(patches_y.flat)
    patches_y_vector = np.tile(patch_y.flatten(), (patches_number, 1)) + np.tile(patches_y.flatten(), (patch_dimension, 1)).transpose()
    patches_x_vector = np.tile(patch_x.flatten(), (patches_number, 1)) + np.tile(patches_x.flatten(), (patch_dimension, 1)).transpose()
    index = np.ravel_multi_index([patches_y_vector, patches_x_vector], (padded_height, padded_width))
    return np.reshape(pad_array.flatten()[index], (patches_number, patch_dimension))

def unpatchify(patches, output_array_size, overlap=1):
    patches_number, patch_dimension = np.shape(patches)
    if not _validPatchDimension(patch_dimension):
        raise SRException(INVALID_PATCH_DIMENSION % patch_dimension)

    patch_width = int(patch_dimension**(.5))
    patch_radius = patch_width / 2;
    interval = patch_width - overlap
    padded_array_size = [d+patch_width for d in output_array_size]
    padded_array_height, padded_array_width = padded_array_size
    padded_array = np.zeros(padded_array_size)
    gaussian_kernel = gaussianKernel(patch_width, 1)
    weight = np.zeros(padded_array_size)
    patches_y, patches_x = np.mgrid[patch_radius:padded_array_width-patch_radius:interval,
                           patch_radius:padded_array_height-patch_radius:interval]
    h, w = np.shape(patches_x)
    patch_idx = 0
    for i in range(h):
        for j in range(w):
            patch_x = patches_x[i, j]
            patch_y = patches_y[i, j]
            padded_array[patch_y-patch_radius:patch_y+patch_radius+1, patch_x-patch_radius:patch_x+patch_radius+1]

    pass






