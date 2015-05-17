__author__ = 'Sherwin'

import numpy as np
from PIL import Image
from PIL import ImageFilter
from sr_util import sr_image_util
from sr_factory.sr_method_factory import SRMethodFactory
from operator import add
from operator import sub

class SRImage(object):

    def __init__(self, y_data, cb_data=None, cr_data=None):

        self._y_data = y_data
        self._cb_data, self._cr_data = cb_data, cr_data
        self._size = np.shape(self._y_data)

    @property
    def size(self):
        """Get the size of the SR image.

        @return: size of the SR image
        @rtype: list
        """
        return self._size

    def get_data(self):
        return self._y_data

    def reconstruct(self, ratio, method_type):
        """Reconstruct a SR image by the given SR method.

        @param ratio: reconstruct ratio
        @type ratio: float
        @param method_type: SR method type
        @type method_type: str
        @return: reconstructed SR image
        @rtype: L{sr_image.SRImage}
        """
        sr_method = SRMethodFactory.create_method(method_type)
        return sr_method.reconstruct(ratio, self)

    def resize(self, ratio):
        """Return a resized copy of the original SRImage by the given resize ratio.

        @param ratio: resize ratio
        @type ratio: float
        @return: a resized copy of the original SRImage
        @rtype: L{sr_image.SRImage}
        """
        size = sr_image_util.create_size(self.size, ratio)
        resized_image = sr_image_util.resize(self._y_data, size)
        return SRImage(resized_image, self._cb_data, self._cr_data)

    def putdata(self, data):
        """Update the SRImage instance by the given data.

        @param data: two dimensional image data
        @type data: L{numpy.array}
        """
        size = np.shape(data)
        if self._size != size:
            raise "Invalid image data, data size not equal to image size %s." % self._size
        self._y_data = data

    def upgrade(self, size, kernel):
        """Upgrade the image to the given size and blurred with given kernel.

        @param size: target upgrade size
        @param kernel: blur kernel
        @return: upgraded image
        @rtype: L{SRImage}
        """
        upgraded_image = sr_image_util.resize(self._y_data, size)
        blurred_image = sr_image_util.filter(upgraded_image, kernel)
        return SRImage(blurred_image, self._cb_data, self._cr_data)

    def downgrade(self, size, kernel):
        """Downgraded the image to given size and blurred with given kernel.

        @param size: target downgrade size
        @param kernel: blur kernel
        @return: downgraded image
        @rtype: L{SRImage}
        """
        blurred_image = sr_image_util.filter(self._y_data, kernel)
        downgraded_image = sr_image_util.resize(blurred_image, size)
        return SRImage(downgraded_image, self._cb_data, self._cr_data)

    def _downgrade(self, ratio, kernel):
        """Downgrade the original SR image with the given ratio and blur kernel.

        @param ratio: downgrade ratio
        @type ratio: float
        @param kernel: blur kernel
        @type kernel: L{numpy.array}
        @return: downgraded image with same size as original image
        @rtype: L{sr_image.SRImage}
        """
        size = sr_image_util.create_size(self.size, 1.0/ratio)
        blurred_image = sr_image_util.filter(self._y_data, kernel)
        downgraded_image = sr_image_util.resize(blurred_image, size)
        downgraded_image = sr_image_util.resize(downgraded_image, self._size)
        return SRImage(downgraded_image, self._cb_data, self._cr_data)

    def get_pyramid(self, level, ratio):
        """Get a pyramid of SR images from the original image.

        @param level: level of pyramid
        @type level: int
        @param ratio: ratio between two neighboring SR image
        @type ratio: float
        @return: image pyramid
        @rtype: list of SR image
        """
        pyramid = []
        r = 1.0
        ALPHA = 2 ** (1.0/3)
        for i in range(level):
            r *= ratio
            gaussian_kernel = sr_image_util.gaussian_kernel(sigma=(ALPHA**i)/3.0)
            pyramid.append(self._downgrade(r, gaussian_kernel))
        return pyramid

    def patchify(self, patch_size, interval=1):
        """Create an array of image patches with the given patch size.

        @param patch_size: size of the patch
        @type patch_size: two dimensional list of the patch size
        @return: an array contains all the patches from the image
        @rtype: L{numpy.array}
        """
        image_array = self._y_data
        return sr_image_util.patchify(image_array, patch_size, interval)

    def save(self, path, extension):
        if self._cb_data is not None and self._cr_data is not None:
            self._cb_data = sr_image_util.resize(self._cb_data, self._size)
            self._cr_data = sr_image_util.resize(self._cr_data, self._size)
        image = sr_image_util.compose(self._y_data, self._cb_data, self._cr_data)
        image.save(path, extension)

    def __add__(self, other_sr_image):
        my_image_data = self._y_data
        other_image_data = other_sr_image.get_data()
        image_data = my_image_data + other_image_data
        return SRImage(image_data, self._cb_data, self._cr_data)

    def __sub__(self, other_sr_image):
        my_image_data = self._y_data
        other_image_data = other_sr_image.get_data()
        image_data = my_image_data - other_image_data
        return SRImage(image_data, self._cb_data, self._cr_data)

    def __mul__(self, factor):
        my_image_data = self._y_data
        image_data = my_image_data * factor
        return SRImage(image_data, self._cb_data, self._cr_data)


