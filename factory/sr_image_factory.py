__author__ = 'Sherwin'

from PIL import Image
from sr_image import SRImage

class SRImageFactory(object):

    @classmethod
    def createSRImageFrom(cls, image_path):
        image = Image.open(image_path)
        return SRImage(image.convert("L"))

    @classmethod
    def createSRImage(cls, image):
        return SRImage(image.convert("L"))
