from __future__ import division

import numpy as np

from .utils import make_tiles, tile_map

# from svm.autocorrelation_width import tiled_minwidth

AVAILABLE_PREDICTORS = [
    "intensity_histogram_normal_distance_35_masked_mean",
    "intensity_histogram_normal_distance_35_masked_variance",
    "intensity_histogram_normal_distance_42_masked_mean",
    "intensity_histogram_normal_distance_42_masked_variance",
    "frac_reflection_masked_mean",
    "frac_reflection_masked_variance",
    "r_masked_mean",
    "r_masked_variance",
    "g_masked_mean",
    "g_masked_variance",
    "b_masked_mean",
    "b_masked_variance",
    "acw_masked_mean",
    "acw_masked_variance",
    "vollath_f4_masked_mean",
    "vollath_f4_masked_variance"
]


# def minwidth(tiles):
#     w, p = tiled_minwidth(tiles)
#     return w.reshape((-1,)), p.reshape((-1,))


def array_of_tiles(ts):
    return ts.reshape((-1,) + ts.shape[2:])


def intensity_histogram_normal_distance(ar, bins=14, mu=0.42, sigma=0.15):
    hist, edges = np.histogram(ar, bins=bins, range=(0, 1))
    centers = (edges[:-1] + edges[1:]) / 2.0
    g = np.exp(-((centers - mu)**2 / sigma**2 / 2.0))
    distance = np.dot(hist / hist.sum(), g / g.sum())
    return distance


def frac_reflection(ar, f=0.9):
    white_count = (ar > f).sum()
    total_count = float(np.size(ar))
    return white_count / total_count


def vollath_f4(img):
    vol_f4 = (
        np.sum(img[:, :-1] * img[:, 1:]) - np.sum(img[:, :-2] * img[:, 2:]))
    return vol_f4


def predictor(f, ta):
    pv = np.array(list(map(f, ta)))
    return pv.mean(), pv.var()


def iterable(o):
    try:
        iter(o)
    except TypeError as te:
        return False
    return True


class blindSvmPredictors(object):
    """ Compute predictor values from the image

    Tries to cache intermediate work shared by multiple predictors.
    """
    def __init__(self, cutout, outline, tile_size=128, to_scale=True):
        """

        Arguments:
            cutout  Image data
            outline blind polygon in cutout coordinates (as returned by blind
                    finder)
        """
        self.tile_size = tile_size
        self.image_data = cutout
        self.outline = outline
        self.cache = {}
        self.to_scale = to_scale #HK

    @property
    def scaled_image_data(self):
        """ Image data scaled to 0-1.0 """
        if 'scaled_image' not in self.cache:
            if self.to_scale:
                self.cache['scaled_image'] = self.image_data / 255.0
            else:
                self.cache['scaled_image'] = self.image_data

        return self.cache['scaled_image']


    @property
    def tiles(self):
        """ Image data aranged in fixed size tiles """
        if 'tiles' not in self.cache:
            self.cache['tiles'] = make_tiles(
                self.scaled_image_data, self.tile_size)
        return self.cache['tiles']

    @property
    def tile_map(self):
        """ boolean map of tiles which are inside the blind outline """
        if 'tile_map' not in self.cache:
            self.cache['tile_map'] = tile_map(
                self.tiles.shape[:2], self.scaled_image_data.shape[:2],
                self.outline)
        return self.cache['tile_map']

    @property
    def tile_array(self):
        """ flattened array of tiles """
        if 'tile_array' not in self.cache:
            self.cache['tile_array'] = array_of_tiles(self.tiles)
        return self.cache['tile_array']

    @property
    def masked_tile_array(self):
        """ Flattened array of tiles inside the outline

        Position data is lost.
        """
        if 'masked_tile_array' not in self.cache:
            self.cache['masked_tile_array'] = self.tile_array[
                self.tile_map.reshape((-1,))]
        return self.cache['masked_tile_array']

    @property
    def cutout_area(self):
        return self.scaled_image_data.shape[0] * \
               self.scaled_image_data.shape[1]
    #
    # def __getitem__(self, predictor):
    #     """ Retrieve the named predictor
    #
    #     If called with a string, calculate the correct predictor.
    #     If called with an iterable, return a list of values for each named
    #     predictor in the iterable.
    #     """
    #     if type(predictor) != str and iterable(predictor):
    #         return [self[x] for x in predictor]
    #     if predictor in ["cutout_area"]:
    #         return getattr(self, predictor)
    #     split_predictor = predictor.split('_')
    #     summary_stat = split_predictor[-1]
    #     if summary_stat in ['mean', 'variance']:
    #         split_predictor.pop(-1)
    #     elif summary_stat == 'masked':
    #         summary_stat = None
    #     else:
    #         raise KeyError("Unknown predictor %s, summary not recognized" %
    #                        predictor)
    #     # method to call is 'var', need to rename for easy access
    #     summary_stat = {
    #         'mean': 'mean',
    #         'variance': 'var',
    #         None: None}[summary_stat]
    #     if split_predictor.pop(-1) != 'masked':
    #         raise KeyError(
    #             'Unknown predictor, expected "masked" %s' % predictor)
    #     normalized_predictor = '_'.join(split_predictor)
    #     if normalized_predictor not in self.cache:
    #         pred = self.calculate_predictor(split_predictor)
    #         if pred is None:
    #             raise KeyError('Unknown predictor %s' % predictor)
    #         if normalized_predictor in ['acw', 'acp']:
    #             self.cache['acw'], self.cache['acp'] = pred
    #         else:
    #             self.cache[normalized_predictor] = pred
    #     if summary_stat is None:
    #         # return masked array of values
    #         return self.cache[normalized_predictor]
    #     else:
    #         return getattr(self.cache[normalized_predictor], summary_stat)()
    #
    # def calculate_predictor(self, split_predictor):
    #     """ Split the provided name into parts and calculate the correct value
    #     """
    #     if (split_predictor[:-1] ==
    #             'intensity_histogram_normal_distance'.split('_')):
    #         mu = int(split_predictor[-1]) / 100.0
    #         return np.array(list(map(partial(
    #             intensity_histogram_normal_distance, mu=mu),
    #             self.masked_tile_array[..., 1])))
    #     if split_predictor[:2] == 'frac_reflection'.split('_'):
    #         return np.array(list(map(frac_reflection,
    #                                  self.masked_tile_array[..., 1])))
    #     if split_predictor[:2] == 'vollath_f4'.split('_'):
    #         return np.array(list(map(vollath_f4,
    #                                  self.masked_tile_array[..., 1])))
    #     if split_predictor[0] in ['acw', 'acp']:
    #         # only green channel
    #         acw, acp = minwidth(self.tiles[..., 1])
    #         # select tiles on blind
    #         return (acw[self.tile_map.reshape((-1,))],
    #                 acp[self.tile_map.reshape((-1,))])
    #     channels = ['r', 'g', 'b']
    #     if split_predictor[0] in channels:
    #         if 'mean' not in self.cache:
    #             self.cache['mean'] = self.masked_tile_array.mean(axis=(-2, -3))
    #         j = channels.index(split_predictor[0])
    #         return self.cache['mean'][..., j]
    #     return None
    #
    # def get_predictors(self, predictor_list):
    #     return np.array(self[predictor_list]).reshape((1, -1))
