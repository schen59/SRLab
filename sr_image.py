__author__ = 'Sherwin'

import numpy as np
from PIL import Image
from sr_util import sr_image_util
from sr_factory.sr_method_factory import SRMethodFactory
from operator import add
from operator import sub

class SRImage(object):

    def __init__(self, image):
        self._image = image
        self._size = image.size

    @property
    def size(self):
        """Get the size of the SR image.

        @return: size of the SR image
        @rtype: list
        """
        return self._size

    def get_image(self):
        return self._image

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
        size = sr_image_util.create_size(self._size, ratio)
        resized_image = self._image.resize(size, Image.BILINEAR)
        return SRImage(resized_image)

    def putdata(self, data):
        """Update the SRImage instance by the given data.

        @param data: two dimensional image data
        @type data: L{numpy.array}
        """
        size = np.shape(data)
        if self._size != size:
            raise "Invalid image data, data size not equal to image size %s." % self._size
        self._image.putdata(list(data.flatten()))

    def upgrade(self, size, kernel):
        upgraded_image = self._image.resize(size, Image.BICUBIC)
        blurred_image = upgraded_image.filter(kernel)
        return SRImage(blurred_image)

    def downgrade(self, size, kernel):
        blurred_image = self._image.filter(kernel)
        downgraded_image = blurred_image.resize(size, Image.BICUBIC)
        return SRImage(downgraded_image)

    def _downgrade(self, ratio, kernel):
        """Downgrade the original SR image with the given ratio and blur kernel.

        @param ratio: downgrade ratio
        @type ratio: float
        @param kernel: blur kernel
        @type kernel: L{PIL.ImageFilter.Kernel}
        @return: downgraded image with same size as original image
        @rtype: L{sr_image.SRImage}
        """
        size = sr_image_util.create_size(self._size, ratio)
        blurred_image = self._image.filter(kernel)
        downgraded_image = blurred_image.resize(size, Image.BILINEAR)
        return SRImage(downgraded_image.resize(self._size, Image.BILINEAR))

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
        for _ in range(level):
            r *= ratio
            gaussian_kernel = sr_image_util.gaussian_kernel(sigma=r)
            pyramid.append(self._downgrade(r, gaussian_kernel))
        return pyramid

    def patchify(self, patch_size, interval=1):
        """Create an array of image patches with the given patch size.

        @param patch_size: size of the patch
        @type patch_size: two dimensional list of the patch size
        @return: an array contains all the patches from the image
        @rtype: L{numpy.array}
        """
        image_array = np.reshape(np.array(list(self._image.getdata())), self._size)
        return sr_image_util.patchify(image_array, patch_size, interval)

    def save(self, path, extension):
        self._image.save(path, extension)

    def __add__(self, other_sr_image):
        my_image_data = list(self._image.getdata())
        other_image_data = list(other_sr_image.get_image().getdata())
        image_data = map(add, my_image_data, other_image_data)
        image = Image.new("L", self._size)
        image.putdata(image_data)
        return SRImage(image)

    def __sub__(self, other_sr_image):
        my_image_data = list(self._image.getdata())
        other_image_data = list(other_sr_image.get_image().getdata())
        image_data = map(sub, my_image_data, other_image_data)
        image = Image.new("L", self._size)
        image.putdata(image_data)
        return SRImage(image)

    def __mul__(self, factor):
        my_image_data = np.array(list(self._image.getdata()))
        image_data = my_image_data * factor
        image = Image.new("L", self._size)
        image.putdata(list(image_data.flatten()))
        return SRImage(image)


