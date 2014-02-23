__author__ = 'Sherwin'

import numpy as np
from PIL import Image
from util import sr_image_util
from factory.sr_method_factory import SRMethodFactory

class SRImage(object):

    def __init__(self, image):
        self._image = image
        self._size = image.size

    @property
    def size(self):
        return self._size

    def getImage(self):
        return self._image

    def reconstruct(self, ratio, method_type):
        sr_method = SRMethodFactory.createMethod(method_type)
        return sr_method.reconstruct(ratio, self)

    def _downgrade(self, ratio):
        size = sr_image_util.createSize(self._size, ratio)
        downgraded_image = self._image.resize(size, Image.BILINEAR)
        return SRImage(downgraded_image.resize(self._size, Image.BILINEAR))

    def getPyramid(self):
        pyramid = []
        ratio = 1.0
        for level in range(1, 7):
            ratio *= 1.25
            pyramid.append(self._downgrade(ratio))
        return pyramid

    def patchify(self, patch_size):
        image_array = np.reshape(np.array(list(self._image.getdata())), self._size)
        return sr_image_util.patchify(image_array, patch_size)
        pass



