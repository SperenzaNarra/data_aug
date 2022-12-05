from random import choice
from typing import List
from imgaug import augmenters as iaa
from numpy import ndarray

rotate          = iaa.Rotate((-40, -20))
shear           = iaa.ShearX((-10, 10))
gaussian        = iaa.GaussianBlur(sigma=(0.0, 4.0))
motion          = iaa.MotionBlur(k=15)
addictive       = iaa.AdditiveGaussianNoise(scale=(0, 0.2*255))
multiply        = iaa.Multiply((0.5, 1.5), per_channel=0.5)
spatter_orig    = [iaa.imgcorruptlike.Spatter(severity=i) for i in range(1, 4)]

class spatter_generator:
    def __init__(self, *args) -> None:
        self.fn = [iaa.Sequential([spatter_orig[i], *args]) for i in range(len(spatter_orig))]
    def __call__(self, image=None, images:List[ndarray]=None):
        if not images is None:
            result = []
            for image in images:
                aug = choice(self.fn)
                result.append(aug(image=image))
            return result
        
        elif not image is None:
            aug = choice(self.fn)
            return aug(image=image)