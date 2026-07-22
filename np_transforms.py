import random
import numbers
import numpy as np
import torch
from torchvision import transforms

def _is_numpy_image(img):
    return isinstance(img, np.ndarray)

class RandomCrop(object):
    """
    Performs a random crop on a numpy array [H, W, C].
    Pads automatically if image dimensions are smaller than crop size.
    """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, pic):
        if not _is_numpy_image(pic):
            raise TypeError('img should be numpy array. Got {}'.format(type(pic)))
        if len(pic.shape) != 3:
            pic = pic.reshape(pic.shape[0], pic.shape[1], -1)

        h, w = pic.shape[:2]
        th, tw = self.size

        # Pad image if smaller than crop dimensions
        if h < th or w < tw:
            pad_h = max(0, th - h)
            pad_w = max(0, tw - w)
            pic = np.pad(pic, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
            h, w = pic.shape[:2]

        max_i = max(0, h - th)
        max_j = max(0, w - tw)

        i = random.randint(0, max_i) if max_i > 0 else 0
        j = random.randint(0, max_j) if max_j > 0 else 0

        return pic[i:i + th, j:j + tw, ...]

class CenterCrop(object):
    """
    Crops the center of a numpy array [H, W, C].
    Pads automatically if image dimensions are smaller than crop size.
    """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, pic):
        if not _is_numpy_image(pic):
            raise TypeError('img should be numpy array. Got {}'.format(type(pic)))
        if len(pic.shape) != 3:
            pic = pic.reshape(pic.shape[0], pic.shape[1], -1)

        h, w = pic.shape[:2]
        th, tw = self.size

        if h < th or w < tw:
            pad_h = max(0, th - h)
            pad_w = max(0, tw - w)
            pic = np.pad(pic, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
            h, w = pic.shape[:2]

        i = max(0, int(round((h - th) / 2.)))
        j = max(0, int(round((w - tw) / 2.)))

        return pic[i:i + th, j:j + tw, ...]

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
