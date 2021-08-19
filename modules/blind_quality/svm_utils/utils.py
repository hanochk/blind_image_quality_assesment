from __future__ import division
import math

import numpy as np

from skimage import measure


def iota(a):
    if hasattr(a, '__len__'):
        N = len(a)
    else:
        try:
            N = int(a)
        except ValueError:
            raise ValueError(
                "Value is neither an integer nor an object with length")
    return np.arange(N)


def pad(rectangle, image_shape, padding):
    padded = dict(rectangle)
    padded['top'] = int(max(0, padded['top'] - padding))
    padded['bottom'] = int(min(image_shape[0], padded['bottom'] + padding))
    padded['left'] = int(max(0, padded['left'] - padding))
    padded['right'] = int(min(image_shape[1], padded['right'] + padding))
    padded['height'] = int(padded['bottom'] - padded['top'])
    padded['width'] = int(padded['right'] - padded['left'])
    return padded


def make_tiles(image, N, step=None):
    """ build possibly overlapping tiles of size N

    Tiles are all of size N and ceil(edge/N) are created in each direction.
    If N exactly divides edge, the tiles do not overlap. Otherwise, the overlap
    is (ceil(edge/N) - edge/N)
    """
    if step is None:
        step = N
    shape = (int(math.ceil((image.shape[0] - N) / step)) + 1,
             int(math.ceil((image.shape[1] - N) / step)) + 1,
             N, N, image.shape[2])
    strides = (int((image.shape[0] - N) / (shape[0] - 1)) * image.strides[0],
               int((image.shape[1] - N) / (shape[1] - 1)) * image.strides[1],
               image.strides[0], image.strides[1], image.strides[2])
    tiles = np.lib.stride_tricks.as_strided(
        image, shape=shape, strides=strides)
    return tiles


def tile_map(tile_shape, cutout_shape, outline):
    R, C = tile_shape
    H, W = cutout_shape
    # find centroid coordinates of tiles
    tile_C, tile_R = np.meshgrid(
        (np.arange(C) + 0.5) * W / C,
        (np.arange(R) + 0.5) * H / R)
    tile_centroids = np.concatenate(
        (tile_C[..., None], tile_R[..., None]), axis=-1)

    # select tiles whose centroid is inside the provided outline
    return measure.points_in_poly(
        tile_centroids.reshape((-1, 2)),
        [(pt.x, pt.y) for pt in outline]).reshape((R, C))
