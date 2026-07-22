import random
import numbers
import numpy as np
import torch
from torchvision import transforms

def _is_numpy_image(img):
    return isinstance(img, np.ndarray)

class RandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(pic, output_size):
        w, h = pic.shape[:2]
        th, tw = output_size
        i = random.randint(0, w - tw)
        j = random.randint(0, h - th)
        return i, j, th, tw

    def __call__(self, pic):
        if not _is_numpy_image(pic):
            raise TypeError('img should be numpy array. Got {}'.format(type(pic)))
        if len(pic.shape) != 3:
            pic = pic.reshape(pic.shape[0], pic.shape[1], -1)
        i, j, th, tw = self.get_params(pic, self.size)
        return pic[i:i + th, j:j + tw, ...]

class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(pic, output_size):
        h, w = pic.shape[:2]
        th, tw = output_size
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))
        return i, j, th, tw

    def __call__(self, pic):
        if not _is_numpy_image(pic):
            raise TypeError('img should be numpy array. Got {}'.format(type(pic)))
        if len(pic.shape) != 3:
            pic = pic.reshape(pic.shape[0], pic.shape[1], -1)
        i, j, h, w = self.get_params(pic, self.size)
        return pic[i:i + h, j:j + w, :]

class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, pic):
        if not _is_numpy_image(pic):
            raise TypeError('img should be numpy array. Got {}'.format(type(pic)))
        if len(pic.shape) != 3:
            pic = pic.reshape(pic.shape[0], pic.shape[1], -1)
        if random.random() < self.prob:
            return pic[:, ::-1, :]
        return pic

class ToTensor(object):
    def __call__(self, pic):
        if not _is_numpy_image(pic):
            raise TypeError('img should be numpy array. Got {}'.format(type(pic)))
        if len(pic.shape) == 1:
            return torch.FloatTensor(pic.copy())
        return torch.FloatTensor(pic.transpose((2, 0, 1)).copy())

class Compose(transforms.Compose):
    pass
