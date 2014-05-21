__author__ = 'Sherwin'

import numpy as np
from sr_dataset import SRDataSet

class ICCV09(object):

    def __init__(self):
        self._method_type = "iccv09"

    def get_method_type(self):
        return self._method_type

    def reconstruct(self, ratio, sr_image):
        """Reconstruct the SR image by the given ratio.

        @param ratio: reconstruct ratio
        @type ratio: float
        @param sr_image: original SR image
        @type sr_image: L{sr_image.SRImage}
        @return: reconstructed SR image
        @rtype: L{sr_image.SRImage}
        """
        sr_dataset = SRDataSet.from_sr_image(sr_image)
        pass

