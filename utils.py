import os
import copy
import numpy as np
import scipy.misc

class item_pool(object):
    def __init__(self, max_num=50):
        self.max_num = max_num
        self.num = 0
        self.items = []

    def __call__(self, in_items):
        """ in_items is a list of item"""
        if self.max_num == 0:
            return in_items
        return_items = []
        for in_item in in_items:
            if self.num < self.max_num:
                self.items.append(in_item)
                self.num = self.num + 1
                return_items.append(in_item)
            else:
                if np.random.rand() > 0.5:
                    idx = np.random.randint(0, self.max_num)
                    tmp = copy.copy(self.items[idx])
                    self.items[idx] = in_item
                    return_items.append(tmp)
                else:
                    return_items.append(in_item)
        return return_items


def mkdir(paths):
    if not isinstance(paths, list):
        paths = [paths]
    for path in paths:
        path_dir, _ = os.path.split(path)
        if not os.path.isdir(path_dir):
            os.makedirs(path_dir)


def immerge(images, row, col):
    """
    merge images into an image with (row * h) * (col * w)

    `images` is in shape of N * H * W(* C=1 or 3)
    """

    if images.ndim == 4:
        c = images.shape[3]
    elif images.ndim == 3:
        c = 1

    h, w = images.shape[1], images.shape[2]
    if c > 1:
        img = np.zeros((h * row, w * col, c))
    else:
        img = np.zeros((h * row, w * col))
    for idx, image in enumerate(images):
        i = idx % col
        j = idx // col
        img[j * h:j * h + h, i * w:i * w + w, ...] = image

    return img

def to_range(images, min_value=0.0, max_value=1.0, dtype=None):
    """
    transform images from [-1.0, 1.0] to [min_value, max_value] of dtype
    """
    assert \
        np.min(images) >= -1.0 - 1e-5 and np.max(images) <= 1.0 + 1e-5 \
        and (images.dtype == np.float32 or images.dtype == np.float64), \
        'The input images should be float64(32) and in the range of [-1.0, 1.0]!'
    if dtype is None:
        dtype = images.dtype
    return ((images + 1.) / 2. * (max_value - min_value) + min_value).astype(dtype)

def imwrite(image, path):
    """ save an [-1.0, 1.0] image """
    return scipy.misc.imsave(path, to_range(image, 0, 255, np.uint8))
