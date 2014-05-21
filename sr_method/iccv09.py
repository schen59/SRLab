__author__ = 'Sherwin'

import numpy as np

DEFAULT_PATCH_SIZE = [5, 5]

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

        patches_without_dc = self._get_patches_without_dc(sr_image)

    def _get_patches_without_dc(self, sr_image):
        """Get patches without dc from the given SR image.

        @param sr_image: SR image
        @type sr_image: L{sr_image.SRImage}
        @return: patches from the given SR image without DC component
        @rtype: L{numpy.array}
        """
        patches = sr_image.patchify(DEFAULT_PATCH_SIZE)
        patches_dc = self._get_dc(patches)
        return patches - patches_dc

    def _get_dc(self, patches):
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