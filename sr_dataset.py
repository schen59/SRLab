__author__ = 'Sherwin'

import numpy as np
from scipy import spatial
from sr_util import sr_image_util

class SRDataSet(object):

    def __init__(self, low_res_patches, high_res_patches):
        self._low_res_patches = low_res_patches
        self._high_res_patches = high_res_patches
        self._kd_tree = spatial.KDTree(self.low_res_patches)

    @property
    def low_res_patches(self):
        return self._low_res_patches

    @property
    def high_res_patches(self):
        return self._high_res_patches

    def add(self, low_res_patches, high_res_patches):
        """Add low_res_patches -> high_res_patches mapping to the dataset.

        @param low_res_patches: low resolution patches
        @type low_res_patches: L{numpy.array}
        @param high_res_patches: high resolution patches
        @type high_res_patches: L{numpy.array}
        """
        self._low_res_patches = np.concatenate((self._low_res_patches, low_res_patches))
        self._high_res_patches = np.concatenate((self._high_res_patches, high_res_patches))
        self._kd_tree = spatial.KDTree(self._low_res_patches)

    def query(self, low_res_patches, neighbors=1):
        """Query the high resolution patches for the given low resolution patches.

        @param low_res_patches: low resolution patches
        @type low_res_patches: L{numpy.array}
        @param neighbors: number of neighbors to query for
        @type neighbors: int
        @return: high resolution patches for the given low resolution patches
        @rtype: L{numpy.array}
        """
        distances, indices = self._kd_tree.query(low_res_patches, neighbors)
        neighbor_patches = self.high_res_patches[indices]
        return self._merge_high_res_patches(neighbor_patches, distances) if neighbors > 1 else neighbor_patches

    def _merge_high_res_patches(self, neighbor_patches, distances):
        """Get the high resolution patches by merging the neighboring patches with the given distance as weight.

        @param neighbor_patches: neighboring high resolution patches
        @type neighbor_patches: L{numpy.array}
        @param distances: distance vector associate with the neighboring patches
        @type distances: L{numpy.array}
        @return: high resolution patches by merging the neighboring patches
        @rtype: L{numpy.array}
        """
        patch_number, neighbor_number, patch_dimension = np.shape(neighbor_patches)
        weights = sr_image_util.normalize(np.exp(distances))
        weights = weights[:, np.newaxis].reshape(patch_number, neighbor_number, 1)
        high_res_patches = np.sum(neighbor_patches*weights, axis=1)
        return high_res_patches
