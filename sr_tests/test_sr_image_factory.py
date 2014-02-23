__author__ = 'Sherwin'

import unittest
from PIL import Image
from factory.sr_image_factory import SRImageFactory

class TestSRImageFactory(unittest.TestCase):

    def testCreateSRImageFrom(self):
        sr_image = SRImageFactory.createSRImageFrom("test_data/babyface_4.png")
        self.assertIsNotNone(sr_image.getImage())

    def testCreateSRImage(self):
        image = Image.open("test_data/babyface_4.png")
        sr_image = SRImageFactory.createSRImage(image)
        self.assertIsNotNone(sr_image.getImage())

if __name__ == "__main__":
    unittest.main()