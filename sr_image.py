__author__ = 'Sherwin'

import numpy as np
from PIL import Image
from sr_util import sr_image_util
from sr_factory.sr_method_factory import SRMethodFactory

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
        sr_method = SRMethodFactory.create_method(method_type)
        return sr_method.reconstruct(ratio, self)

    def _downgrade(self, ratio):
        size = sr_image_util.create_size(self._size, ratio)
        downgraded_image = self._image.resize(size, Image.BILINEAR)
        return SRImage(downgraded_image.resize(self._size, Image.BILINEAR))

    def get_pyramid(self, level, ratio):
        pyramid = []
        r = 1.0
        for _ in range(level):
            r *= ratio
            pyramid.append(self._downgrade(r))
        return pyramid

    def patchify(self, patch_size):
        image_array = np.reshape(np.array(list(self._image.getdata())), self._size)
        return sr_image_util.patchify(image_array, patch_size)
        pass



