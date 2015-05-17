__author__ = 'shaofeng'

from PIL import Image
from sr_factory.sr_image_factory import SRImageFactory

def letter_example():
    image = Image.open("../test_data/letter.png")
    sr_image = SRImageFactory.create_sr_image(image)
    reconstructed_sr_image = sr_image.reconstruct(3, 'iccv09')
    reconstructed_sr_image.save("../test_data/letter_3x.png", "png")

def babyface_example():
    image = Image.open("../test_data/babyface_4.png")
    sr_image = SRImageFactory.create_sr_image(image)
    reconstructed_sr_image = sr_image.reconstruct(4, 'iccv09')
    reconstructed_sr_image.save("../test_data/babyface_4x.png", "png")

def monarch_example():
    image = Image.open("../test_data/monarch.png")
    sr_image = SRImageFactory.create_sr_image(image)
    reconstructed_sr_image = sr_image.reconstruct(3, 'iccv09')
    reconstructed_sr_image.save("../test_data/monarch_3x.png", "png")

if __name__ == '__main__':
    letter_example()
    babyface_example()
    monarch_example()