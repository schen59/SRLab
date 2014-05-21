__author__ = 'Sherwin'

import numpy as np
from PIL import Image
from sr_exception.sr_exception import SRException

DEFAULT_PATCH_SIZE = [5, 5]
INVALID_PATCH_SIZE_ERR = "Invalid patch size %s, patch should be square with odd size."
INVALID_PATCH_DIMENSION = "Invalid patch dimension %s, patch should be square with odd size."

def create_size(size, ratio):
    """Create a new size which the new size equals size*ratio.

    @param size: original size
    @type size: list
    @param ratio:
    @type ratio: float
    @return: new size
    @rtype: list
    """
    new_size = map(int, [dimension*ratio for dimension in size])
    return new_size

def create_gaussian_kernel(radius=2, sigma=1.0):
    """Create a 2D gaussian kernel.

    @param radius: the radius of the kernel
    @type radius: int
    @param sigma: the sigma of the kernel
    @type sigma: float
    @return: a normalized gaussian kernel
    @rtype: L{numpy.array}
    """
    y, x = np.mgrid[-radius:radius+1, -radius:radius+1]
    unnormalized_kernel = np.exp(-(x**2 + y**2)//2*sigma*sigma)
    return unnormalized_kernel / np.sum(unnormalized_kernel)

def _valid_patch_size(patch_size):
    """Check if the given patch size is valid. A valid patch size should be odd square with two dimension.

    @param patch_size: size of the patch
    @type patch_size: list
    @return: if the patch size if valid
    @rtype: bool
    """
    if len(patch_size) != 2:
        return False

    height, width = patch_size
    return height == width and height % 2 != 0

def _valid_patch_dimension(patch_dimension):
    """Check if the given patch dimension is valid or not. A patch dimension is the size of the flatten patch.

    @param patch_dimension:
    @type patch_dimension: int
    @return: if the patch dimension is valid or not
    @rtype: bool
    """
    patch_height = int(patch_dimension**(.5))
    patch_width = patch_dimension / patch_height
    return _valid_patch_size([patch_height, patch_width])

def patchify(array, patch_size, interval=1):
    """Create a list of patches by sampling the given array in row-major order with the given patch size
    and interval between patches.

    @param array: given array to patchify
    @type array: L{numpy.array}
    @param patch_size: a two dimension list which gives the size of width and height of the patch
    @type patch_size: list
    @param interval: interval between patches
    @type interval: int
    @return: list of sampled patches
    @rtype: L{numpy.array}
    """
    if not _valid_patch_size(patch_size):
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

def unpatchify(patches, output_array_size, kernel, overlap=1):
    """Create an array from the given patches by merging them together with the given kernel and overlap size.

    @param patches: given patches array
    @type patches: L{numpy.array}
    @param output_array_size: size of output array
    @type output_array_size: list
    @param kernel: kernel to merge the patches into array
    @type kernel: L{numpy.array}
    @param overlap: overlap size between patches
    @type overlap: int
    @return: merged array from the given patches
    @rtype: L{numpy.array}
    """
    patches_number, patch_dimension = np.shape(patches)
    if not _valid_patch_dimension(patch_dimension):
        raise SRException(INVALID_PATCH_DIMENSION % patch_dimension)
    if patch_dimension != np.shape(np.flatten())[0]:
        raise "Invalid kernel size, kernel size should be equal to patch size."
    patch_width = int(patch_dimension**(.5))
    patch_radius = patch_width / 2;
    interval = patch_width - overlap
    padded_array_size = [d+patch_width for d in output_array_size]
    padded_array_height, padded_array_width = padded_array_size
    padded_array = np.zeros(padded_array_size)
    weight = np.zeros(padded_array_size)
    patches_y, patches_x = np.mgrid[patch_radius:padded_array_width-patch_radius:interval,
                           patch_radius:padded_array_height-patch_radius:interval]
    h, w = np.shape(patches_x)
    patch_idx = 0
    for i in range(h):
        for j in range(w):
            patch_x = patches_x[i, j]
            patch_y = patches_y[i, j]
            padded_array[patch_y-patch_radius:patch_y+patch_radius+1, patch_x-patch_radius:patch_x+patch_radius+1] = \
                np.reshape(patches[patch_idx], [patch_width, patch_width]) * kernel
            weight[patch_y-patch_radius:patch_y+patch_radius+1, patch_x-patch_radius:patch_x+patch_radius+1] += kernel
            patch_idx += 1
    padded_array /= weight
    return padded_array[patch_radius:padded_array_height-patch_radius+1, patch_radius:padded_array_width-patch_radius+1]

def normalize(array):
    """Normalize the row vector of a 2D array.

    @param array: 2D array
    @type array: L{numpy.array}
    @return: normalized 2D array
    @rtype: L{numpy.array}
    """
    return array.astype(float) / np.sum(array, axis=1)[:, np.newaxis]

def get_patches_without_dc(self, sr_image):
    """Get patches without dc from the given SR image.

    @param sr_image: SR image
    @type sr_image: L{sr_image.SRImage}
    @return: patches from the given SR image without DC component
    @rtype: L{numpy.array}
    """
    patches = sr_image.patchify(DEFAULT_PATCH_SIZE)
    patches_dc = get_dc(patches)
    return patches - patches_dc

def get_dc(self, patches):
    """Get the dc component of the row major order patches.

    @param patches: row major order patches
    @type patches: L{numpy.array}
    @return: dc component of all the patches
    @rtype: L{numpy.array}
    """
    h, w = np.shape(patches)
    patches_dc = np.mean(patches, axis=1)
    patches_dc = np.tile(patches_dc, [h, 1]).transpose()
    return patches_dc





