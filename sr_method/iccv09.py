__author__ = 'Sherwin'

import numpy as np
from sr_dataset import SRDataSet
from sr_util import sr_image_util

DEFAULT_RECONSTRUCT_LEVEL = 6

class ICCV09(object):

    def __init__(self):
        self._method_type = "iccv09"
        self._kernel = sr_image_util.create_gaussian_kernel()

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
        reconstructed_sr_image = sr_image
        r = ratio ** (1.0/DEFAULT_RECONSTRUCT_LEVEL)
        for level in range(DEFAULT_RECONSTRUCT_LEVEL):
            print "\rReconstructing %.2f%%" % (float(level) / DEFAULT_RECONSTRUCT_LEVEL * 100)
            reconstructed_sr_image = self._reconstruct(r, reconstructed_sr_image, sr_dataset)
            reconstructed_sr_image = sr_image_util.back_project(reconstructed_sr_image, sr_image, 3)
            new_sr_dataset = SRDataSet.from_sr_image(reconstructed_sr_image)
            sr_dataset.merge(new_sr_dataset)
        return reconstructed_sr_image

    def _reconstruct(self, ratio, sr_image, sr_dataset):
        """Reconstruct a SRImage using the given SRDataset by the given ratio.

        @param ratio: reconstruct ratio
        @type ratio: float
        @param sr_image: original SRImage
        @type sr_image: L{sr_image.SRImage}
        @param sr_dataset:
        @type sr_dataset: L{sr_dataset.SRDataset}
        @return: reconstructed SRImage
        @rtype: L{sr_image.SRImage}
        """
        resized_sr_image = sr_image.resize(ratio)
        patches_without_dc, patches_dc = sr_image_util.get_patches_from(resized_sr_image, interval=4)
        high_res_patches_without_dc = sr_dataset.query(patches_without_dc, neighbors=9)
        high_res_patches = high_res_patches_without_dc + patches_dc
        high_res_data = sr_image_util.unpatchify(high_res_patches, resized_sr_image.size, self._kernel)
        resized_sr_image.putdata(high_res_data)
        return resized_sr_image

