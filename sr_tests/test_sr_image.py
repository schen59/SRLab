__author__ = 'Sherwin'

import unittest
import numpy as np
from PIL import Image
from factory.sr_image_factory import SRImageFactory

class TestSRImage(unittest.TestCase):

    def setUp(self):
        self.image = Image.open("test_data/babyface_4.png")
        self.sr_image = SRImageFactory.createSRImage(self.image)

    def testSize(self):
        self.assertEqual(self.image.size, self.sr_image.size)

    def testGetPyramid(self):
        pyramid = self.sr_image.getPyramid()
        self.assertEqual(len(pyramid), 6)

    def testPatchify(self):
        patches = self.sr_image.patchify([5, 5])
        height, width = self.sr_image.size
        patches_size = np.shape(patches)
        self.assertEqual(patches_size[0], height*width)
        self.assertEqual(patches_size[1], 25)

if __name__ == "__main__":
    unittest.main()