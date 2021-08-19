from __future__ import print_function, division

import copy
import os
from dev.configuration import print_arguments
from dev.evaluation import evaluate_model_on_dataset
from dev.utillities.file import load_csv_xls_2_df
from PIL import Image
import pickle
import json
from dev.models import initialize_model

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from dev.dataset import prepare_test_dataloader_tile_based, clsss_quality_label_2_str, quality_to_labels_annotator_version, class_labels as quality_to_labels_basic_version, softmax_map_to_final_labels

# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from modules.blind_quality.svm_utils.blindSvmPredictors import blindSvmPredictors as blindSvmPredictors
from dev.inference import prepare_clargs_parser
from collections import namedtuple
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import PIL.ImageColor as ImageColor
from PIL import ImageEnhance
import matplotlib.colors as mcolors
import tqdm
import glob
from sklearn.preprocessing import label_binarize

Point = namedtuple('Point', ['x', 'y'])
from modules.blind_quality.quality_utils import tile_mapping
from dev.dataset import class_labels
from dev.configuration import add_clargs, CLARG_SETS

# clsss_quality_label_2_str = {1: 'bad', 2: 'marginal', 3: 'good', 4: 'unknown-tested'}
# quality_to_labels_annotator_version = {'poor': 1, 'marginal': 2, 'excellent': 3, 'unknown-tested' :4}
# quality_to_labels_basic_version = {'bad': 1, 'marginal': 2, 'good': 3, 'unknown-tested' :4}
#
# softmax_map_to_final_labels = {0 :1, 1: 3, 2: 2}


class FusionTiles(torch.nn.Module):
    def __init__(self, voting_method='concensus_and_vote', thr=np.arange(0, 1, 0.01), voting_percentile=np.arange(0, 105, 5),
                 *args, **kwargs):

        self.voting_method = voting_method  #'concensus_and_vote' or 'vote_4_good_class'
        self.thr = thr
        self.voting_percentile = voting_percentile
        self.tp_mat_acm = np.zeros([voting_percentile.shape[0], thr.shape[0]])
        self.fn_mat_acm = np.zeros([voting_percentile.shape[0], thr.shape[0]])
        self.tn_mat_acm = np.zeros([voting_percentile.shape[0], thr.shape[0]])
        self.fp_mat_acm = np.zeros([voting_percentile.shape[0], thr.shape[0]])
        self.tnr_cnt = 0
        self.tpr_cnt = 0

    def calc_fusion(self, label_name, predictions_good_cls):

        self.tp_mat = np.zeros([self.voting_percentile.shape[0], self.thr.shape[0]])
        self.fn_mat = np.zeros([self.voting_percentile.shape[0], self.thr.shape[0]])
        self.tn_mat = np.zeros([self.voting_percentile.shape[0], self.thr.shape[0]])
        self.fp_mat = np.zeros([self.voting_percentile.shape[0], self.thr.shape[0]])
        for i, th in enumerate(self.thr):  # threshold for the positive class
            for j, vote_p in enumerate(self.voting_percentile):
                # acc = np.sum(good_det.astype('int')) / all_targets.shape[0]
                dets = predictions_good_cls >= th  # softmax out #1 is good class decision criterion remains the same
                det = dets.astype('int')  # if actual=bad cls[=0] but still prob[good]>th =>fp

                if label_name == 'good':
                    tp = np.sum(det.astype('int')) / predictions_good_cls.shape[0]
                    fn = 1 - tp

                    fn_vote = 0
                    tp_vote = 0

                    if self.voting_method == 'concensus_and_vote':
                        if tp > fn: # pass the threshold criterion
                            tp_vote = (tp >= (vote_p / 100)).astype('int')
                            if not tp_vote: # is passed the voting criterion
                                fn_vote = 1
                        else:
                            fn_vote = 1
                    elif self.voting_method == 'vote_4_good_class':
                        # based on good class only if it passes the threshold then tru else false
                        tp_vote = (tp >= (vote_p / 100)).astype('int')
                        if tp_vote == 0:
                            fn_vote = 1
                    else:
                        raise ValueError("not supported type")
                    self.tp_mat[j, i] = tp_vote
                    self.fn_mat[j, i] = fn_vote
                else:
                    fp = np.sum(det.astype('int')) / predictions_good_cls.shape[0]
                    tn = 1 - fp

                    fp_vote = 0
                    tn_vote = 0

                    if self.voting_method == 'concensus_and_vote':
                        if fp > tn:
                            fp_vote = (fp >= (vote_p / 100)).astype('int')
                            if fp_vote == 0:
                                tn_vote = 1
                        else:
                            tn_vote = 1
                    elif self.voting_method == 'vote_4_good_class':
                        fp_vote = (fp >= (vote_p / 100)).astype('int')  # in the context of bad class the positive (good class) is FP in that case
                        if fp_vote == 0:
                            tn_vote = 1

                    else:
                        raise ValueError("not supported type")
                    self.fp_mat[j, i] = fp_vote
                    self.tn_mat[j, i] = tn_vote

        if label_name == 'good':
            self.tpr_cnt += 1
            self.tp_mat_acm = self.tp_mat_acm + self.tp_mat
            self.fn_mat_acm = self.fn_mat_acm + self.fn_mat
            # print("tpr_cnt {}".format(self.tpr_cnt))
        elif label_name == 'bad':
            self.tn_mat_acm = self.tn_mat_acm + self.tn_mat
            self.fp_mat_acm = self.fp_mat_acm + self.fp_mat
            self.tnr_cnt += 1
            # print("tnr_cnt {}".format(self.tnr_cnt))
        else:
            raise ValueError("Unrecognized label !!")

        return

    def save_results(self, path):
        stat_res = dict(tp_mat_acm=self.tp_mat_acm, tpr_cnt=self.tpr_cnt,
                            tn_mat_acm=self.tn_mat_acm, tnr_cnt=self.tnr_cnt, fn_mat_acm=self.fn_mat_acm,
                            fp_mat_acm=self.fp_mat_acm, thr=self.thr, voting_percentile=self.voting_percentile,
                            voting_method=self.voting_method)

        with open(path, 'wb') as f:
            pickle.dump(stat_res, f)

def voting_over_tiles(good_cls_ratio, image_fusion_voting_th, confidence_threshold,
                      image_fusion_voting_method='vote_4_good_class'):

    if image_fusion_voting_method == 'vote_4_good_class':
        if good_cls_ratio >= image_fusion_voting_th:  # (good_cls_ratio>0.5) n_good_tiles should be > bad_tiles 'concensus_and_vote'
            th_of_class = confidence_threshold
            acc = good_cls_ratio
            label = 3
        else:
            th_of_class = 1 - confidence_threshold  # Threshold for class good => for class bad it is 1-th
            label = 1
            acc = 1 - good_cls_ratio

    elif image_fusion_voting_method == 'concensus_and_vote':
        if good_cls_ratio >= image_fusion_voting_th and good_cls_ratio > 0.5:  # (good_cls_ratio>0.5) n_good_tiles should be > bad_tiles 'concensus_and_vote'
            th_of_class = confidence_threshold
            acc = good_cls_ratio
            label = 3
        else:  # bad quality
            th_of_class = 1 - confidence_threshold  # Threshold for class good => for class bad it is 1-th
            acc = 1 - good_cls_ratio
            label = 1
    else:
        raise

    return label, acc, th_of_class

def create_hsv_value_and_map_to_rgb(percentage_array, count_array):
    import matplotlib.colors as mcolors
    Hue_values = abs((1 / 3) * (percentage_array - np.nanmin(percentage_array)) / (
                np.nanmax(percentage_array) - np.nanmin(percentage_array)) - 1 / 3)
    saturation_values = count_array / count_array.max()
    value_matrix = np.ones([percentage_array.shape[0], percentage_array.shape[1]])
    array_hsv_colors = np.concatenate(
        (Hue_values[..., np.newaxis], saturation_values[..., np.newaxis], value_matrix[..., np.newaxis]), axis=2)
    array_rgb_colors = mcolors.hsv_to_rgb(array_hsv_colors)
    return array_rgb_colors


# fig, ax = plt.subplots()
# clist = [(0, "green"), (1. / 2., "yellow"), (1, "red")]
# cmap = matplotlib.colors.LinearSegmentedColormap.from_list("name", clist)
# ax = calvings_graph.plot_array_as_image(ax, array_rgb_colors, (-30 * 24 + 23) / 24, (50 * 24) / 24, y_min, y_max, cmap,
#                                         f"{farm_name} - Percentage of sick cows for each normalized action number {action_num}")
# plt.show()
def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color='red',
                               thickness=4,
                               display_str_list=(),
                               use_normalized_coordinates=True):
  """Adds a bounding box to an image.

  Bounding box coordinates can be specified in either absolute (pixel) or
  normalized coordinates by setting the use_normalized_coordinates argument.

  Each string in display_str_list is displayed on a separate line above the
  bounding box in black text on a rectangle filled with the input 'color'.
  If the top of the bounding box extends to the edge of the image, the strings
  are displayed below the bounding box.

  Args:
    image: a PIL.Image object.
    ymin: ymin of bounding box.
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box
                      (each to be shown on its own line).
    use_normalized_coordinates: If True (default), treat coordinates
      ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
      coordinates as absolute.
  """
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  if use_normalized_coordinates:
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
  else:
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
  draw.line([(left, top), (left, bottom), (right, bottom),
             (right, top), (left, top)], width=thickness, fill=color)
  try:
    font = ImageFont.truetype('arial.ttf', 24)
  except IOError:
    font = ImageFont.load_default()

  # If the total height of the display strings added to the top of the bounding
  # box exceeds the top of the image, stack the strings below the bounding box
  # instead of above.
  display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
  # Each display_str has a top and bottom margin of 0.05x.
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

  if top > total_display_str_height:
    text_bottom = top + total_display_str_height
  else:
    text_bottom = bottom # + total_display_str_height
  # Reverse list and print from bottom to top.
  for display_str in display_str_list[::-1]:
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle(
        [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                          text_bottom)],
        fill=color)
    draw.text(
        (left + margin, text_bottom - text_height - margin),
        display_str,
        fill='black',
        font=font)
    text_bottom -= text_height - 2 * margin

def plot_cls_tiles_box_over_blind(tile_size, pred, tiles_id, good_det, pil_img, path_save,
                             actual_class_name, file_tag=None, brightness_fact=1.0, softmax_map=False,
                             softmax_score_good_cls=None, box_or_dot='box', file_suffix='softmax_good_cls_overlay'):
    # ovlp_y = (int(pred.scaled_image_data.shape[0] / tile_size) + 1 - pred.scaled_image_data.shape[
    #     0] / tile_size) * tile_size
    # ovlp_x = (int(pred.scaled_image_data.shape[1] / tile_size) + 1 - pred.scaled_image_data.shape[
    #     1] / tile_size) * tile_size

    pilimg = pil_img.copy()
    if softmax_map:
        pilimg_soft = pil_img.copy()

    # model_label_to_index = class_labels[actual_class_name]
    # conf_class = soft_score[:, model_label_to_index]
    #
    # th_acc = 0.5
    if isinstance(tiles_id, torch.Tensor):
        tiles_id_ = tiles_id.numpy()
    else:
        tiles_id_ = tiles_id
        if isinstance(tiles_id_[0], str): # new method of ids is full tile name
            tiles_id_ = [int(tile.split('_tile_')[-1].split('.png')[0]) for tile in tiles_id]
        # if len(tiles_id.shape)>1:#when extracting full image blind processing tile NO are in (1,N) hence workarround
        #     tiles_id_ = tiles_id[0]

    if actual_class_name == 'good':
        color1 = [ImageColor.getrgb('green') if x == 1 else ImageColor.getrgb('yellow') for x in good_det.astype('int')]
        color_cont = [tuple(mcolors.hsv_to_rgb((0.33+(1-x), 1, 255)).astype('int')) for x in softmax_score_good_cls] # 0.33 stands for green in Hue scale
    else:
        color1 = [ImageColor.getrgb('red') if x == 1 else ImageColor.getrgb('orange') for x in good_det.astype('int')]
        color_cont = [tuple(mcolors.hsv_to_rgb((x, 1, 255)).astype('int')) for x in softmax_score_good_cls] # x=0 is red hence taking the good class likelihood for the bad actual class since  the more it is bad the more class good llr is 0 =>red

    thickness = 10
    det_str = ''
    det = dict()
    ovlap_tile_size_y = (pred.scaled_image_data.shape[0] - tile_size) / (pred.tile_map.shape[0] - 1)
    ovlap_tile_size_x = (pred.scaled_image_data.shape[1] - tile_size) / (pred.tile_map.shape[1] - 1)
    if 0:
        for idx, loc_tile in enumerate(tiles_id_):
            tile_x = int(loc_tile % pred.tile_map.shape[1])
            tile_y = int(loc_tile / pred.tile_map.shape[1])
            # print(tile_x, tile_y)
            # print(tile_x + tile_y * pred.tile_map.shape[0])
            # print(tile_x + tile_y * pred.tile_map.shape[1])
            det["bottom"] = int(tile_y * ovlap_tile_size_y + tile_size)
            det["left"] = int(tile_x * ovlap_tile_size_x)
            det["top"] = int(tile_y * ovlap_tile_size_y)
            det["right"] = int(tile_x * ovlap_tile_size_x + tile_size)
            if box_or_dot == 'box':
                draw_bounding_box_on_image(pilimg, det["bottom"], det["left"], det["top"], det["right"],
                                           color=color1[idx],
                                           thickness=thickness,
                                           display_str_list=det_str, use_normalized_coordinates=False)
            elif box_or_dot == 'dot':
                dot_diameter = 1
                center_y = int((det['bottom'] - det['top']) / 2)
                center_x = int((det['right'] - det['left']) / 2)
                det['top'] = det['top'] + center_y
                det['bottom'] = det['top'] + dot_diameter
                det['left'] = det['left'] + center_x
                det['right'] = det['left'] + dot_diameter

                draw_bounding_box_on_image(pilimg, det["bottom"], det["left"], det["top"], det["right"],
                                           color=color_cont[idx],
                                           thickness=thickness,
                                           display_str_list=det_str, use_normalized_coordinates=False)

            else:
                raise

        if 0: # no one useing that image
            enhancer = ImageEnhance.Brightness(pilimg)
            enhancer.enhance(brightness_fact).save(os.path.join(path_save, file_tag + '_cls_'+ actual_class_name + '_tiles_overlay.png'))
        else:
            pilimg.save(os.path.join(path_save, file_tag + '_Act_cls_' + actual_class_name + '_tiles_over_blind.png'))

############
    softmax_score_good_cls_target = softmax_score_good_cls
    if 0:
        if actual_class_name == 'good':
            color_base = mcolors.rgb_to_hsv(ImageColor.getrgb('green'))
        else:
            color_base = mcolors.rgb_to_hsv(ImageColor.getrgb('red'))
    else:
        color_base = mcolors.rgb_to_hsv(ImageColor.getrgb('green'))
        # softmax_score_good_cls_target = 1 - softmax_score_good_cls
    # 1-soft : since the basic assumption is that classification ok =>green if actual=good, hence the softmax divert that value

    color1 = [mcolors.hsv_to_rgb(color_base + np.array([1-soft, 0, 0])).astype('int') for soft in softmax_score_good_cls_target]
    if softmax_map:
        for idx, loc_tile in enumerate(tiles_id_):
            tile_x = int(loc_tile % pred.tile_map.shape[1])
            tile_y = int(loc_tile / pred.tile_map.shape[1])
            # print(tile_x, tile_y)
            # print(tile_x + tile_y * pred.tile_map.shape[0])
            # print(tile_x + tile_y * pred.tile_map.shape[1])
            det["bottom"] = int(tile_y * ovlap_tile_size_y + tile_size)
            det["left"] = int(tile_x * ovlap_tile_size_x)
            det["top"] = int(tile_y * ovlap_tile_size_y)
            det["right"] = int(tile_x * ovlap_tile_size_x + tile_size)
            color_tup = (color1[idx][0], color1[idx][1], color1[idx][2])

            draw_bounding_box_on_image(pilimg_soft, det["bottom"], det["left"], det["top"], det["right"],
                                       color=color_tup,
                                       thickness=thickness,
                                       display_str_list=det_str, use_normalized_coordinates=False)
            fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 70)
            d = ImageDraw.Draw(pilimg_soft)
            d.text((det["left"] + 0.25*det["right"]-0.25*det["left"], det["top"] - 0.5*(det["top"]-det["bottom"])),
                   str(softmax_score_good_cls_target[idx].__format__('.2f')), font=fnt, fill=(255, 255, 0))

        enhancer = ImageEnhance.Brightness(pilimg_soft)
        enhancer.enhance(brightness_fact).save(
            os.path.join(path_save, file_tag + '_Act_cls_' + actual_class_name + '_' + file_suffix + '.png'))


def create_tiles_and_meta_data(image, outline, tile_size, args, cutout_id, label, media_id=None, remove_tiles_on_edges=True, calculate_metadata_only=False):
    df_test = pd.DataFrame(columns=['file_name', 'cutout_uuid', 'class', 'label', 'percent_40', 'percent_60', 'percent_80',
                                    'in_outline_tiles_id', 'train_or_test', 'full_file_name', 'media_uuid',
                                    'n_tiles', 'N_rows', 'M_cols', 'tile_ind'])

    pred = blindSvmPredictors(cutout=image, outline=outline, tile_size=tile_size)
    image_hsv_acm_list = []

    dim_tiles = pred.tile_map.shape
    tot_n_tiles = dim_tiles[0] * dim_tiles[1]
    in_outline_n_tiles = np.where(pred.tile_map)[0].shape
    print("In blind tiles ratio {}".format(in_outline_n_tiles[0] / tot_n_tiles))
    # the tiles within the blind
    if pred.cache['tiles'].shape[3] != tile_size and pred.cache['tiles'].shape[2] != tile_size:
        print('Error not the tile dim specified')

    if remove_tiles_on_edges:
        dim_all_tiles = pred.tile_map.shape
        tile_map = list()
        # ref_edges = 'center'
        ref_edges = 'topleft'
        tile_map.append(tile_mapping(dim_all_tiles, cutout_shape=image.shape[:2], outline=outline, ref_edges=ref_edges))
        ref_edges = 'topright'
        tile_map.append(tile_mapping(dim_all_tiles, cutout_shape=image.shape[:2], outline=outline, ref_edges=ref_edges))
        ref_edges = 'bottomleft'
        tile_map.append(tile_mapping(dim_all_tiles, cutout_shape=image.shape[:2], outline=outline, ref_edges=ref_edges))
        ref_edges = 'bottomright'
        tile_map.append(tile_mapping(dim_all_tiles, cutout_shape=image.shape[:2], outline=outline, ref_edges=ref_edges))

        # and oiperation between 4 permutations of intersection between tile 4 edges and the outline
        temp =np.ones_like(pred.tile_map).astype('bool')
        for tmap in tile_map:
            temp = temp & tmap
        pred.cache['tile_map'] = temp

        all_tiles_id = np.where(pred.tile_map.ravel())[0]
        remove_tiles_on_edges_ratio = all_tiles_id.shape[0] / in_outline_n_tiles[0]
        print("Removed tiles on edges ratio {} total {}".format(remove_tiles_on_edges_ratio, all_tiles_id.shape[0]))

    tiles = pred.masked_tile_array
    if tiles.size == 0:
        print("Warning contour gave no Tiles!!!!")
        return

    # Tiles are ordered row-wise by pred.tile_map telling which is relevent in outline by TRue/False
    # saving all the tiles
    assert (np.where(pred.tile_map.reshape((-1)))[0].shape[0] == tiles.shape[0])
    in_outline_tiles_id = np.where(pred.tile_map.reshape((-1)))[0]
    for tile_no in range(tiles.shape[0]):
        fname_save = str(media_id) + '_' + cutout_id + '_cls_' + clsss_quality_label_2_str[label] + '_tile_' + str(tile_no)

        # tiles are normalized by 255 to [0 1] tiles are saved scaled to [0 255].
        # CNN dataloder (DataSet) expecting [0 255] and scaling occurs when __getitem()__ applied to [0 1]
        pilimg = Image.fromarray((tiles[tile_no, :, :, :] * 255.0).astype(np.uint8))
        full_fmame = os.path.join(args.result_dir, fname_save + '.png')
        file_name = fname_save
        if not calculate_metadata_only:
            pilimg.save(full_fmame)

        tile_img = np.asarray(pilimg)
        percent_40 = np.percentile(tile_img.ravel(), 40)
        percent_60 = np.percentile(tile_img.ravel(), 60)
        percent_80 = np.percentile(tile_img.ravel(), 80)

        image_hsv = mcolors.rgb_to_hsv(tile_img)
        image_hsv_acm_list.append(image_hsv)


        df_test.loc[len(df_test)] = [file_name, cutout_id, clsss_quality_label_2_str[label], label,
                                     percent_40, percent_60, percent_80, in_outline_tiles_id[tile_no], 'test', full_fmame,
                                     media_id, all_tiles_id.size, dim_tiles[0], dim_tiles[1], all_tiles_id[tile_no]]

    image_hsv_acm = np.concatenate(image_hsv_acm_list)
    sv_norm = image_hsv_acm[:, :, 1] * image_hsv_acm[:, :, 2]
    sv_norm = sv_norm.mean(axis=0).mean(axis=0)
    norm_hue = image_hsv_acm[:, :, 0] * image_hsv_acm[:, :, 1] * image_hsv_acm[:, :, 2] / sv_norm
    norm_hue = norm_hue.mean(axis=0).mean(axis=0)
    df_test["all_image_weighted_hue"] = norm_hue
    df_test["all_image_mean_sat"] = image_hsv_acm[:, :, 1].mean()

    df_test.to_csv(os.path.join(args.result_dir, cutout_id + '.csv'), columns=['file_name', 'cutout_uuid', 'class', 'label', 'percent_40', 'percent_60', 'percent_80',
                        'in_outline_tiles_id', 'train_or_test', 'media_uuid', 'all_image_weighted_hue', 'n_tiles', 'N_rows', 'M_cols', 'tile_ind'], index=False)

    return df_test, pred

def plot_outline_with_comments(image, outline, path, cutout_id, acc, label):
    # plot_image_and_outline(image, [outline], 10)
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.plot([p[0] for p in outline], [p[1] for p in outline])
    ax.title.set_text('contour_blind_' + str(acc.__format__('.2f')) + '_' + str(cutout_id) + '_cls_' + str(label))
    plt.savefig(os.path.join(path, 'outline' + str(acc.__format__('.2f')) + '_' + str(cutout_id) + '_cls_' + str(label) + '.png'))
"""
Getting that error most of the time
Batch size of dataloader was changed to 71 Dataset filtered 0
cutout_id 555c8d61-1621-5212-9ecd-9cc0fc8a2360: acc 0.6619718309859155
XIO:  fatal IO error 25 (Inappropriate ioctl for device) on X server "localhost:12.0"
      after 6513 requests (6513 known processed) with 126 events remaining.

"""

def plot_image_and_outline(img, outlines, fname_id=''):
    fig, ax = plt.subplots()
    ax.imshow(img)
    for outline in outlines:
        ax.plot([p[0] for p in outline], [p[1] for p in outline])
    plt.pause(0.001)
    plt.ion()
    ax.title.set_text('Contour_over_the_blind' + str(fname_id))
    plt.savefig('CContour_over_the_blind' + str(fname_id) + '.png')
    plt.show()

def main(args: list = None):

    parser = prepare_clargs_parser()
    add_clargs(parser, CLARG_SETS.COMMON)
    add_clargs(parser, CLARG_SETS.ARCHITECTURE)

    parser.add_argument('run_config_name', type=str, metavar='RUN_CONFIGURATION',
                                help="name of an existing run configuration")

    parser.add_argument("--outline-pickle-path", type=str, required=True, default=None, metavar='PATH',
                                        help="outline-pickle-path")

    parser.add_argument("--metadata-json-path", type=str, required=False, default=None, metavar='PATH',
                                        help="training metadata necessary[currently not mandatory] for inference mainly with HCF,")

    parser.add_argument("--result-pickle-file", type=str, required=False, default='inference-list.pkl', metavar='PATH',
                                        help="outline-pickle-file")

    parser.add_argument("--add-hue-offset", action='store_true',
                        help=' add hue offset for examined image to observe accuracy')

    parser.add_argument("--calculate-metadata-only", action='store_true',
                        help=' no saving images')

    parser.add_argument("--image-fusion-voting-th", type=float, required=False, default=0, metavar='FLOAT',
                        help='final image criterion')


    parser.add_argument("--image-fusion-voting-method", type=str, required=False, default='vote_4_good_class',
                        choices=['concensus_and_vote', 'vote_4_good_class', 'avg_pool'], metavar='STRING',
                                        help="")

    parser.add_argument("--statistics-collect", action='store_true',
                        help='statistics collect')

    parser.add_argument("--outline-file-format", type=str, required=False, default='media_id_cutout_id',
                        choices=['media_id_cutout_id', 'misc', 'other', 'fname_n_cutout_id'], metavar='STRING',
                                        help="")

    parser.add_argument("--process-folder-non-recursively", action='store_true', help="If true.")

    parser.add_argument("--plot-dot-overlay-instead-of-box", action='store_true', help="If true.")


    args = parser.parse_args(args)

    if args.image_fusion_voting_th<0 or args.image_fusion_voting_th>1:
        raise

    if args.dataset_split_csv is not None:
        df_annotation = load_csv_xls_2_df(args.dataset_split_csv)
        print('***********   Loading annotations !!!!!!!!  based analysis *****************')
    else:
        print('*********** No annotations *****************')

    remove_tiles_on_edges = True
    if remove_tiles_on_edges:
        print("Tiles on edges are removed !!!! ******************************************")

    # model_name = "mobilenet_v2_256_win" #"mobilenet_v2" #"resnet" #"squeezenet"
    num_classes = [3 if args.classify_image_all_tiles else 2][0]
    model_name = args.run_config_name
    feature_extract = True
    device = torch.device("cuda:" + str(args.gpu_id) + "" if torch.cuda.is_available() else "cpu")

    model_metadata_dictionary = dict()
    if 'metadata_json_path' in args:
        if args.metadata_json_path:
            with open(args.metadata_json_path, 'r') as f:
                model_metadata_dictionary = json.load(f)

    len_handcrafted_features = 0
    if args.handcrafted_features is not None:
        len_handcrafted_features = len(args.handcrafted_features)
        if len_handcrafted_features == 0: # HCF is not in args
            # load the HCF type not by args but from JSON
            if "hcf" not in model_metadata_dictionary: # also not in JSON
                raise ValueError("hand crafted feature was defines but w/o a type!! ")
        if 'metadata_json_path' not in args:
            raise ValueError("hand crafted feature metadata in the JSON mean/sdt os missing  ")
    else:# HCF is not defined by args but can be in JSON
        if "hcf" in model_metadata_dictionary:
            len_handcrafted_features = len(model_metadata_dictionary['hcf'].keys())
            args.handcrafted_features = list()
            for hcf_type in model_metadata_dictionary['hcf'].keys():
                args.handcrafted_features.append(hcf_type)

    pooling_method = ['avg_pooling' if args.fine_tune_pretrained_model_plan == 'freeze_pretrained_add_nn_avg_pooling' else
                      'lp_mean_pooling' if args.fine_tune_pretrained_model_plan == 'freeze_pretrained_add_nn_lp' else
                       'gated_attention' if args.fine_tune_pretrained_model_plan == 'freeze_pretrained_add_gated_atten' else None][0]

    model, input_size = initialize_model(model_name, num_classes, feature_extract,
                                         pretrained_type={'source': 'imagenet', 'path': None},
                                         n_layers_finetune=0,
                                         dropout=0.0, dropblock_group='4', replace_relu=False,
                                         len_handcrafted_features=len_handcrafted_features,
                                         pooling_method=pooling_method,
                                         device=device, pooling_at_classifier=args.pooling_at_classifier,
                                         fc_sequential_type=args.fc_sequential_type,
                                         positional_embeddings=args.positional_embeddings,
                                         debug=True)

    # validate args are not conflicting : only method named "_custom_forward_impl()" supports HCF by convention
    if not hasattr(model, '_custom_forward_impl'):
        if len_handcrafted_features>0:
            raise ValueError("You asked for hand crafted feature but model doesn't support !!!!!")
    else:
        if "hcf" not in model_metadata_dictionary:
            Warning("Loaded model with HCF but JSON dos not contains the HCF information :mean/std !!!!!")
        if len_handcrafted_features==0:
            Warning("You loaded pretrained model with HCF but argument does not contain which one =>hence it is taken from the JSON!!!!!")


    args.input_size = input_size
    args.loss_smooth = False

    # checkpoint = load_model(args.model_path, device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint)

    if torch.cuda.is_available():
        model = model.to(device)

    args.gamma_aug_corr = 1.0 #0.5
    args.batch_size = 64*4
    # args.num_workers = 12
    tile_size = 256 #256
    args.tile_size = tile_size
    print("Tile size {}".format(tile_size))
    if tile_size != 256:
        print("R U sure tile size {}".format(tile_size))

    brightness_fact = 1.0
    args.brightness_fact = brightness_fact
    args.tile_size = tile_size
    test_data_filter = dict()
    if brightness_fact!=1.0:
        print("Stored image are bightned by".format(brightness_fact))


    if 0:
        test_data_filter = {'lessthan': 25, 'greater': 150}
    else:
        test_data_filter = {}

    unique_run_name = args.model_path.split('___')[-1]

    args.test_data_filter = test_data_filter
    df_result_over = pd.DataFrame(columns=['file_name', 'cutout_id', 'good_cls_ratio', 'actual_class', 'FN', 'FP', 'quality_svm', 'predict_all_img_label', 'avg_pool'])

    if test_data_filter:
        print("!!!!!!!!!!  filtering the testset")

    try:
        with open(args.outline_pickle_path, 'rb') as f:
            cutout_2_outline_n_meta = pickle.load(f)
            print('direct')
    except:
        import sys
        sys.path.append('/home/hanoch/GIT')
        sys.path.append('/home/hanoch/GIT/Finders')
        sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        from Finders.finders.blind import default as blindfinder
        with open(args.outline_pickle_path, 'rb') as f:
            cutout_2_outline_n_meta = pickle.load(f)
            print('via_Finders')

    # os.environ['DISPLAY'] = str('localhost:10.0')
    print_arguments(args)
    outline_file_format = args.outline_file_format

    print(outline_file_format)
    resulat_acc = dict()

#path an d files handling
    # filenames = [os.path.join(args.database_root, x) for x in os.listdir(args.database_root)
    #                  if x.endswith('png')]
    if args.process_folder_non_recursively:
        filenames = [os.path.join(args.database_root, x) for x in os.listdir(args.database_root)
                         if x.endswith('png')]
        print('Process folder NON - recursively')
    else:
        filenames = glob.glob(args.database_root + '/**/*.png', recursive=True)
        print('Process folder recursively')

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    args.result_dir_softmap = os.path.join(args.result_dir, 'softmap')
    if not os.path.exists(args.result_dir_softmap):
        os.makedirs(args.result_dir_softmap)
    args.result_dir_outline = os.path.join(args.result_dir, 'outline')
    if not os.path.exists(args.result_dir_outline):
        os.makedirs(args.result_dir_outline)
    args.result_dir_csv = os.path.join(args.result_dir, 'csv')
    if not os.path.exists(args.result_dir_csv):
        os.makedirs(args.result_dir_csv)

    print("!!!! process {} cutouts *****".format(len(filenames)))

    filename_attr = ''
    hue_offset = 0.05
    if args.add_hue_offset:
        filename_attr = 'hue_' + str(hue_offset) + '_'
        print("Hue offset {} was added" .format(hue_offset))

    missing_label = False
    if args.dataset_split_csv is None:
        label = 4 #was 1 # Path should be unknown4  # unknown
        # dummy label
        missing_label = True

# modes of operation
    resulat_stat_acc = dict()
    if args.statistics_collect:
        quality_svm = -1
        fusion_tiles = FusionTiles(voting_method='concensus_and_vote')
        # fusion_tiles = FusionTiles(voting_method='vote_4_good_class')
        # SVM info in comparison
        if 0:
            from svm import blindSvm

            path_to_init_file = '/home/hanoch/GIT/blind_quality_svm/bin/trn/svm_benchmark/FACSIMILE_PRODUCTION_V0_1_5/fixed_C_200_facsimile___prod_init_hk.json'
            svm = blindSvm.blindSvm(
                blind_svm_params=path_to_init_file)  # Following the default path as an example, this line looks for all .json files in the trn submodule and picks the first one
            print(svm.blind_svm_params.model_handlers.keys())
            svm_scores = dict()

    predict_all_img_label = -1
    all_img_acc = -1
    unsupervised_label = ''
    if args.positional_embeddings:
        args.get_image_name_item = False
    else:
        args.get_image_name_item = True

    if args.confidence_threshold is None and args.classify_image_all_tiles:
        args.confidence_threshold = 1.0/3.0
    avg_pool = -1

    for idx, file in enumerate(tqdm.tqdm(filenames)):
        cutout_uuid = None
        media_id = None

        if file.split('.')[-1] == 'png':
            img = Image.open(file).convert("RGB")
            image = np.asarray(img)

            quality_svm_reported = quality_svm = None
            if outline_file_format == 'media_id_cutout_id':
                cutout_id = file.split('.')[-2].split('_')[-1]
                outline = cutout_2_outline_n_meta.get(cutout_id, None)
                if outline == None:
                    print("Outline for cutout_id {} was not found {}!!!".format(cutout_id, file))
                    continue

                if isinstance(outline, dict):
                    outline = outline['outline']

            elif outline_file_format == 'misc':
                cutout_id = file.split('/')[-1]
                outline_dict = cutout_2_outline_n_meta.get(cutout_id, None)
                if outline_dict == None:
                    print("Outline for cutout_id {} was not found!!!".format(cutout_id))
                    continue
                quality_svm_reported = outline_dict.get('quality', None)
                if 'contour' in outline_dict:
                    outline = outline_dict.get('contour', None)
                else:
                    outline = outline_dict.get('outline', None)

                # quality_svm_reported = clsss_quality_label_2_str[outline_dict['quality']]
            elif outline_file_format == 'fname_n_cutout_id':
                if 'blind_' in os.path.split(file)[-1]:
                    cutout_id = 'blind' + os.path.split(file)[-1].split('blind')[-1]
                    media_id = cutout_id # format is blind_xyz_cutout_uuid hdence keep the namre for filipping the order
                    cutout_uuid = os.path.split(file)[-1].split('_blind')[0]  # this is the reali cutout id
                elif 'cutout_' in os.path.split(file)[-1]:
                    cutout_id = os.path.split(file)[-1].split('cutout_')[-1].split('.png')[0]
                # cutout_uuid = os.path.split(file)[-1].split('_blind')[0] # this is the reali cutout id
                outline_dict = cutout_2_outline_n_meta.get(cutout_id, None)
                if outline_dict == None:
                    print("Outline for cutout_id {} was not found!!!".format(cutout_id))
                    continue
                quality_svm_reported = outline_dict.get('quality', None)
                label = outline_dict.get('image_quality_grade', None)
                if label:
                    label = ['unknown-tested' if label == -1 else label][0]
                    label = quality_to_labels_annotator_version[label] # labels given in string no grade
                else:
                    label = 4
                # supports old version
                if 'contour' in outline_dict:
                    outline = outline_dict.get('contour', None)
                    if outline is not None:
                        if isinstance(outline, dict):
                            # outline = outline['outline']
                            if 'contour' in outline.keys():
                                outline = outline_dict['contour']
                            elif 'outline' in outline.keys():
                                outline = outline['outline']
                else:
                    outline = outline_dict.get('outline', None)
                if outline == None:
                    print("Strange !!! the nested dict : Outline for cutout_id {} was not found!!!".format(cutout_id))
                    continue

            # If cutout isnot found then skip
            if media_id is None:
                if isinstance(cutout_2_outline_n_meta[cutout_id], dict):
                    media_id = cutout_2_outline_n_meta[cutout_id].get('media_id', None)

            if cutout_uuid is not None:
                cutout_id = cutout_uuid
            if media_id is None:
                media_id = cutout_id

            if args.add_hue_offset:
                image_hsv = mcolors.rgb_to_hsv(image)
                image_hsv[:, :, 0] = image_hsv[:, :, 0] + hue_offset
                image = mcolors.hsv_to_rgb(image_hsv) # comment

            # if isinstance(outline, dict):
            #     outline = outline['outline']
            # Use annotations
            if args.dataset_split_csv is not None:
                if np.array(['eileen' in x.lower() for x in df_annotation.keys().to_list()]).any():
                    cell = np.where(df_annotation['cutout_id_alias'] == cutout_id)[0]
                    label = quality_to_labels_basic_version[df_annotation['Eileen'].iloc[cell].item().lower()]
                    # label = df_annotation['Eileen'].apply(lambda x: class_labels[x.lower()])
                elif 'cutout_id' in df_annotation:
                    if ((df_annotation['cutout_id'] == cutout_id).any()):
                        cell = np.where(df_annotation['cutout_id'] == cutout_id)[0][0]
                        class_of_ex = df_annotation['image_quality_grade'].iloc[cell].lower()
                        if not class_of_ex in quality_to_labels_annotator_version:
                            print("CU {} Label is missing from csv {}".format(cutout_id, class_of_ex))
                            continue
                        label = quality_to_labels_annotator_version[class_of_ex] # the first entry of the tiles names having the same class label as the given cutout is taken as the class
                elif 'cutout_uuid' in df_annotation:
                    if ((df_annotation['cutout_uuid'] == cutout_id).any()):
                        cell = np.where(df_annotation['cutout_uuid'] == cutout_id)[0][0]
                        label = df_annotation['label'].iloc[cell] # the first entry of the tiles names having the same class label as the given cutout is taken as the class
                    else:
                        raise
                elif "marginal" in file.split('/'):
                    print('!!!! /*******  This is a patch remove')  #TODO remove
                    label = 1  # Marginal is missing from labels => bad class as well
                elif (df_annotation.cutout_id_alias == cutout_id).any():
                    cell = np.where(df_annotation.cutout_id_alias == cutout_id)[0].item()
                    label = df_annotation['label'].iloc[cell]
                else:  # no annotations in the file
                    print("cutout uuid was not found  in annotations !!!{}".format(cutout_id))
                    continue

            # tick = time.perf_counter()
            # start_time = time.time()

            df_test, blind_tile_obj = create_tiles_and_meta_data(image, outline, tile_size, args, cutout_id,
                                                                label, media_id, calculate_metadata_only=args.calculate_metadata_only)
            if args.calculate_metadata_only:
                continue
            # implant the global mean/std of hcf comes from the csv, hence csv is mandatory
            if "hcf" in model_metadata_dictionary:
                for hcf_type in model_metadata_dictionary['hcf'].keys():
                    # df_test[f"{hcf_type}_mean"] = ""
                    mu = model_metadata_dictionary['hcf'][hcf_type]['mean']
                    df_test[f"{hcf_type}_mean"] = mu
                    std = model_metadata_dictionary['hcf'][hcf_type]['std']
                    df_test[f"{hcf_type}_std"] = std


            # batch size need to be no greater than amount of tiles
            # print("Finished 0 in %.2fs" % (time.time() - start_time))

            batch_of_batches_size = int(len(df_test)/args.batch_size) +1

            all_targets = []
            all_predictions = []
            all_map_relevant_tiles_rowise = []
            all_atten_weithgs = list()
            actual_batch_size = 0

            kwargs = dict()
            kwargs['num_classes'] = num_classes
            kwargs['tta'] = args.tta

            for batch in range(batch_of_batches_size):

                df_test_sliced = df_test.iloc[pd.np.r_[batch*args.batch_size:min((batch + 1)*args.batch_size, len(df_test))]]
                test_dataloader, actual_mini_batch_size = prepare_test_dataloader_tile_based(args,
                                                        df_test_sliced, test_filter_percentile40=test_data_filter,
                                                        pre_process_gamma_aug_corr=args.gamma_aug_corr,
                                                        missing_label=missing_label,
                                                        positional_embeddings=args.positional_embeddings,
                                                        pooling_method=pooling_method, **kwargs)

                actual_batch_size += actual_mini_batch_size

                if actual_batch_size == 0:
                    # all are bad quality, since they were probably filtered by the  test_filter_percentile40
                    if clsss_quality_label_2_str[label] == 'bad':
                        acc = 1.0
                    else:
                        acc = 0.0
                        # plot_outline_with_comments(image, outline, args.result_dir, cutout_id, acc, clsss_quality_label_2_str[label])

                    print('All tiles were filtered out by the Dataset filter = > bad quality image')
                    continue
                # print("Finished 1 in %.2fs" % (time.time() - start_time))
                kwargs = dict()
                kwargs['extract_features'] = False

                return_dict = evaluate_model_on_dataset(model,
                                                    test_dataloader, device, loss_fn=None, max_number_of_batches=None,
                                                    do_softmax=True, missing_label=missing_label,
                                                    positional_embeddings=args.positional_embeddings, **kwargs)

                all_targets_batch = return_dict['all_targets']
                all_predictions_batch = return_dict['all_predictions']
                loss = return_dict['loss']
                all_features = return_dict['all_features']

                if args.get_image_name_item:
                    tiles_id = return_dict['all_tile_id']
                elif args.positional_embeddings:
                    tiles_id = df_test_sliced.tile_ind.to_numpy()

                if args.classify_image_all_tiles:
                    all_atten_weithgs_batch = return_dict['all_atten_weithgs']
                    all_atten_weithgs.append(all_atten_weithgs_batch)

                all_targets.append(all_targets_batch)
                all_predictions.append(all_predictions_batch)
                all_map_relevant_tiles_rowise.append(tiles_id)

                # print("Finished 2 in %.2fs" % (time.time() - start_time))
            all_targets = np.concatenate(all_targets)
            all_predictions = np.concatenate(all_predictions)
            all_map_relevant_tiles_rowise = np.concatenate(all_map_relevant_tiles_rowise)

            if args.classify_image_all_tiles:#when extracting full image blind processing tile NO are in (1,N) hence workarround
                all_atten_weithgs = np.concatenate(all_atten_weithgs)
                all_atten_weithgs = all_atten_weithgs.squeeze()
                all_map_relevant_tiles_rowise = all_map_relevant_tiles_rowise.squeeze()

            #testset no labels hence decision is open loop
            if missing_label:
                # if args.statistics_collect:
                #     raise ValueError('Cant calc statistics where labels are missing')
                if args.classify_image_all_tiles:
                    dets = all_predictions[:, class_labels['good']] > args.confidence_threshold
                    predict_all_img_label = softmax_map_to_final_labels[np.argmax(all_predictions)] # CNN out [0,1,2] goes to [class=1;bad, class=3;good, class=2;marginal]
                    unsupervised_label = 'g_atten__pred_' + clsss_quality_label_2_str[predict_all_img_label]
                    acc = -1
                    fn_tile = -1
                    fp_tile = -1
                    correct_det = all_targets == dets.astype('int')

                else:
                    dets = all_predictions[:, class_labels['good']] > args.confidence_threshold
                    good_cls_ratio = np.sum(dets.astype('int')) / dets.shape[0]
                    predict_all_img_label = -1
                    if args.image_fusion_voting_method == 'concensus_and_vote' or args.image_fusion_voting_method == 'vote_4_good_class':
                        #pseudo label for the entire image based on decision
                        # fusion made by voting method
                        predict_all_img_label, acc_all_img, th_of_class_img = voting_over_tiles(good_cls_ratio, args.image_fusion_voting_th,
                                                                                                    args.confidence_threshold,
                                                                                                    image_fusion_voting_method=args.image_fusion_voting_method)
                        th = args.image_fusion_voting_th
                        fusion_score = acc_all_img
                        #fusion made by average pooling if applicable it over run the voiting based fusion
                    elif args.image_fusion_voting_method == 'avg_pool':
                        avg_pool = all_predictions[:, class_labels['good']].mean()  # softmax out #1 is good class decision criterion remains the same
                        dets = all_predictions[:, class_labels['good']] > avg_pool
                        good_cls_ratio = np.sum(dets.astype('int')) / dets.shape[0]

                        if avg_pool >= args.confidence_threshold:  # softmax out #1 is good class decision criterion remains the same
                            predict_all_img_label = 3
                        else:
                            predict_all_img_label = 1
                        all_img_acc = predict_all_img_label
                        th = args.confidence_threshold
                        fusion_score = avg_pool

                    unsupervised_label = '_pred_' + clsss_quality_label_2_str[predict_all_img_label] + '_vote_th_' + str(th) + '_fusion_score_' + str(fusion_score) + '_'
                    print("File {} final fused cls {}".format(file, predict_all_img_label))

                    if predict_all_img_label == 3:
                        acc = good_cls_ratio
                        th_of_class = args.confidence_threshold
                        all_targets = 1 * np.ones_like(dets)  # as apeared in the softmax_output
                    else:
                        acc = 1 - good_cls_ratio
                        th_of_class = 1 - args.confidence_threshold
                        all_targets = np.zeros_like(dets)  # as apeared in the softmax_output

                    correct_det = all_targets == dets.astype('int')  # good det no matter what the class

                    fn_tile = -1  # dummy value
                    fp_tile = -1

                if args.statistics_collect:
                    if quality_svm_reported is None and 0: #compute svm qual explicit HK#TODO integrate svm qality here
                        pred = blindSvmPredictors(cutout=image, outline=outline)
                        predictors = pred.get_predictors(svm.blind_svm_params.predictors)
                        quality_svm = copy.deepcopy(svm.predict(predictors)[0])
                        # if quality_svm_reported != clsss_quality_label_2_str[quality_svm]:
                        #     print("Discrepancy between qaualities outline file :{} online {}".format(quality_svm_reported, quality_svm))

                nested_record_stat = {'label': label, 'actual_class': clsss_quality_label_2_str[label],
                                      'tile_good_class_pred': all_predictions[:, class_labels['good']],
                                      'pred_label': predict_all_img_label, 'pred_class': clsss_quality_label_2_str[predict_all_img_label],
                                      'acc': acc, 'quality_svm': quality_svm,
                                      'image_fusion_voting_th': args.image_fusion_voting_th,
                                      'confidence_threshold': args.confidence_threshold,
                                      'cutout_id': cutout_id, 'file_name': file,
                                      'nn_likelihood': all_predictions}

                resulat_stat_acc.update({cutout_id: nested_record_stat})

                # intermediate save , avoid all or nothing
                if (idx % 10 == 0):
                    with open(os.path.join(args.result_dir_csv, unique_run_name + '_stat_collect_' + '_tta_' + str(args.tta) + '_' + args.result_pickle_file), 'wb') as f:
                        pickle.dump(resulat_stat_acc, f)

            else: #if missing_label:
                # result analysis
                if args.statistics_collect:
                    if 0:#compute svm qual explicit HK#TODO integrate svm uality here
                        pred = blindSvmPredictors.blindSvmPredictors(cutout=image, outline=outline)
                        predictors = pred.get_predictors(svm.blind_svm_params.predictors)
                        quality_svm = svm.predict(predictors)[0]
                        svm_scores = copy.deepcopy(svm.score)

                        nested_record_stat = {'label': label, 'actual_class': clsss_quality_label_2_str[label],
                                              'tile_good_class_pred': all_predictions[:, class_labels['good']],
                                              'svm_scores': svm_scores, 'cutout_id': cutout_id, 'file_name': file}

                        resulat_stat_acc.update({cutout_id: nested_record_stat})

                        # intermediate save , avoid all or nothing
                        if (idx % 10 ==0):
                            with open(os.path.join(args.result_dir_csv, unique_run_name + '_stat_collect_' + '_tta_' + str(args.tta) + '_' + args.result_pickle_file),
                                      'wb') as f:
                                pickle.dump(resulat_stat_acc, f)


                    else:
                        if 0:# not supporting csv file as input
                            if not args.classify_image_all_tiles:
                                fusion_tiles.calc_fusion(clsss_quality_label_2_str[label], all_predictions[:, class_labels['good']])
                                if 0:
                                    pred = blindSvmPredictors.blindSvmPredictors(cutout=image, outline=outline)
                                    predictors = pred.get_predictors(svm.blind_svm_params.predictors)
                                    quality_svm = svm.predict(predictors)[0]

                all_targets_one_hot = label_binarize(all_targets, classes=[0, 1, 2]) # binarize the 3 classification for 1 vs. all : good class vs. all
                if hasattr(args, 'confidence_threshold'):
                    dets = all_predictions[:, class_labels['good']] > args.confidence_threshold
                    # correct_det = all_targets == dets.astype('int') # good det no matter what the class
                    correct_det = dets.astype('int') == all_targets_one_hot[:, class_labels['good']] # more generic for 3 classes
                else:
                    dets = all_predictions[:, class_labels['good']] >= 0.5
                    correct_det = all_targets_one_hot[:, class_labels['good']] == all_predictions.argmax(axis=1)

                if args.classify_image_all_tiles:
                    if dets:
                        predict_all_img_label = softmax_map_to_final_labels[1] # good class out on logit no #1
                    else:# label 0 or 2 out of indeces 0 or 1*2
                        predict_all_img_label = softmax_map_to_final_labels[2*np.argmax(np.concatenate((all_predictions[:, 0:class_labels['good']], all_predictions[:, class_labels['good']+1:])))]

                good_cls_ratio = np.sum(dets.astype('int'))/all_targets.shape[0]
                acc = np.sum(correct_det.astype('int'))/ all_targets.shape[0]
                print("cutout_id {}: acc {}".format(cutout_id, acc))

                if clsss_quality_label_2_str[label] == 'good':
                    fn_tile = 1 - acc # if all blind is in good q then the tiles below th are considered FN
                    fp_tile = None
                    th_of_class = args.confidence_threshold
                else:
                    fn_tile = None
                    fp_tile = 1 - acc
                    th_of_class = 1 - args.confidence_threshold # Threshold for class good => for class bad it is 1-th

                if args.image_fusion_voting_th and not args.classify_image_all_tiles:
                    good_cls_ratio = np.sum(dets.astype('int')) / dets.shape[0]
                    if args.image_fusion_voting_method == 'concensus_and_vote' or args.image_fusion_voting_method == 'vote_4_good_class':
                        predict_all_img_label, acc_all_img, th_of_class_img = voting_over_tiles(good_cls_ratio=good_cls_ratio,
                                                                                      image_fusion_voting_th=args.image_fusion_voting_th,
                                                                                      confidence_threshold=args.confidence_threshold,
                                                                                    image_fusion_voting_method=args.image_fusion_voting_method)
                        th = args.image_fusion_voting_th
                        fusion_score = acc_all_img

                    if args.image_fusion_voting_method == 'concensus_and_vote' or args.image_fusion_voting_method == 'vote_4_good_class':
                        #pseudo label for the entire image based on decision
                        # fusion made by voting method
                        predict_all_img_label, acc_all_img, th_of_class_img = voting_over_tiles(good_cls_ratio, args.image_fusion_voting_th,
                                                                                                    args.confidence_threshold,
                                                                                                    image_fusion_voting_method=args.image_fusion_voting_method)
                        th = args.image_fusion_voting_th
                        fusion_score = acc_all_img
                        #fusion made by average pooling if applicable it over run the voiting based fusion
                    elif args.image_fusion_voting_method == 'avg_pool':
                        avg_pool = all_predictions[:, class_labels['good']].mean()  # softmax out #1 is good class decision criterion remains the same
                        dets = all_predictions[:, class_labels['good']] > avg_pool
                        good_cls_ratio = np.sum(dets.astype('int')) / dets.shape[0]

                        if avg_pool >= args.confidence_threshold:  # softmax out #1 is good class decision criterion remains the same
                            predict_all_img_label = 3
                        else:
                            predict_all_img_label = 1
                        all_img_acc = predict_all_img_label
                        th = args.confidence_threshold
                        fusion_score = avg_pool





                all_img_acc = [1 if (predict_all_img_label==label) is True else 0][0]

            # tock = time.perf_counter()
            # print("Finished in %.2fs" % (tock - tick))
            # print("Finished 3 in %.2fs" % (time.time() - start_time))

            nested_record = {'good_cls_ratio': good_cls_ratio, 'actual_class': clsss_quality_label_2_str[label], 'FN': fn_tile, 'FP': fp_tile}
            resulat_acc.update({cutout_id: nested_record})
            df_result_over.loc[len(df_result_over)] = [file, cutout_id, good_cls_ratio, clsss_quality_label_2_str[label], fn_tile, fp_tile, quality_svm, predict_all_img_label, avg_pool]
            # Override for consistent labeling related to good class only
            th_of_class = args.confidence_threshold  # Threshold for class good => for class bad it is 1-th

            plot_name = cutout_id + '_' + str(actual_batch_size) + '_tiles_acc_' + str(acc.__format__('.2f')) + unsupervised_label + '_img_all_acc_' + str(all_img_acc.__format__('.1f')) + '_th_' + str(th_of_class.__format__('.2f')) + '_model_' + unique_run_name
            plot_pattern = ['dot' if args.plot_dot_overlay_instead_of_box else 'box'][0]
            if not args.calculate_metadata_only:
                if not args.classify_image_all_tiles:
                    plot_cls_tiles_box_over_blind(tile_size, blind_tile_obj, all_map_relevant_tiles_rowise, correct_det, img,
                                             args.result_dir_softmap, actual_class_name=clsss_quality_label_2_str[label], file_tag=plot_name,
                                             brightness_fact=brightness_fact, softmax_map=True,
                                             softmax_score_good_cls=all_predictions[:, 1], box_or_dot=plot_pattern)
                else:#plot attention per tile instead of softmax per tile
                    plot_cls_tiles_box_over_blind(tile_size, blind_tile_obj, all_map_relevant_tiles_rowise, correct_det, img,
                                             args.result_dir_softmap, actual_class_name=clsss_quality_label_2_str[label], file_tag=plot_name,
                                             brightness_fact=brightness_fact, softmax_map=True,
                                             softmax_score_good_cls=all_atten_weithgs, box_or_dot=plot_pattern,
                                             file_suffix='attention_weights_overlay')

            # clear dataloader and dataframe otherwise all the data are aggregated
            del test_dataloader
            df_test.drop(df_test.index, inplace=True)

            if (idx % 10 == 0):
                with open(os.path.join(args.result_dir_csv, unique_run_name + '_tta_' + str(args.tta) + '_' + args.result_pickle_file), 'wb') as f:
                    pickle.dump(resulat_acc, f)
                df_result_over.to_csv(os.path.join(args.result_dir_csv,
                                    'inference-results_partial_' + unique_run_name + '_th_tta_' + str(args.tta) + '_' + str(args.confidence_threshold) + '.csv'),
                                      index=False)
                # if args.statistics_collect:
                #     fusion_tiles.save_results(os.path.join(args.result_dir_csv, unique_run_name + '_th_tta_' + str(args.tta) + '_' + str(args.confidence_threshold) + '_statistics_fusion_tiles_tta' + str(args.tta) + '_' + args.result_pickle_file))
                    # stat_res = dict(tp_mat_acm=tp_mat_acm, tpr_cnt=tpr_cnt,
                    #                 tn_mat_acm=tn_mat_acm, tnr_cnt=tnr_cnt, fn_mat_acm=fn_mat_acm,
                    #                 fp_mat_acm=fp_mat_acm, thr=thr, voting_percentile=voting_percentile)
                    # with open(os.path.join(args.result_dir, unique_run_name + '_statistics_' + args.result_pickle_file), 'wb') as f:
                    #     pickle.dump(stat_res, f)

    with open(os.path.join(args.result_dir_csv, unique_run_name + '_tta_' + str(args.tta) + '_' + args.result_pickle_file), 'wb') as f:
        pickle.dump(resulat_acc, f)

    if args.statistics_collect:
        with open(os.path.join(args.result_dir_csv, unique_run_name + '_stat_collect_' + '_tta_' + str(args.tta) + '_' + args.result_pickle_file), 'wb') as f:
            pickle.dump(resulat_stat_acc, f)

    resulat_acc.update(vars(args))

    df_result = pd.DataFrame.from_dict(list(resulat_acc.items()))
    df_result = df_result.transpose()
    df_result.to_csv(os.path.join(args.result_dir_csv, 'inference-withsettings' + filename_attr + unique_run_name + '_th_tta_' + str(args.tta) + '_' + str(args.confidence_threshold) + '.csv'), index=False)
    df_result_over.to_csv(os.path.join(args.result_dir_csv, 'inference-results' + filename_attr + unique_run_name + '_th_tta_' + str(args.tta) + '_' + str(args.confidence_threshold) + '.csv'), index=False)

    # if args.statistics_collect:
    #     fusion_tiles.save_results(os.path.join(args.result_dir_csv, unique_run_name + '_th_tta_' + str(args.tta) + '_' + str(
    #                                 args.confidence_threshold) + '_statistics_' + args.result_pickle_file))

    arguments_str = '\n'.join(["{}: {}".format(key, resulat_acc[key]) for key in sorted(resulat_acc)])
    print(arguments_str + '\n')
    # for ROC need to aggregate info
    # roc_plot(all_targets, all_predictions[:, 1], 1, args.result_dir, unique_id=unique_run_name+str(args.confidence_threshold))

if __name__ == "__main__":
    main()


"""
************  This file accepts full cutout images ************* 

--model-path /hdd/hanoch/runmodels/img_quality/results/saved_state_mobilenet_v2_256_win___1591887990 --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/EileenQuality_needtoRunblindFinder_4_outline --result-dir /hdd/hanoch/runmodels/img_quality/results/inference_production/best --outline-pickle-path /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/EileenQuality_needtoRunblindFinder_4_outline/merged_outlines_json.pkl --result-pickle-path inference_best_qual.pkl
From command line:
moved to model 
saved_state_mobilenet_v2_256_win___1592479325
nohup python -u test_blind_quality.py --model-path /hdd/hanoch/runmodels/img_quality/results/saved_state_mobilenet_v2_256_win___1591887990 --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/holdout --result-dir /hdd/hanoch/runmodels/img_quality/results/inference_production --outline-pickle-path /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/holdout/quality_holdout_outlines_merged.pkl --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/train_test_val_quality_merged_tile_tile.csv --result-pickle-path inference.pkl & tail -f nohup.out


saved_state_mobilenet_v2_256_win_3_lyrsgamma_aug_corr___1592752657
--model-path /hdd/hanoch/runmodels/img_quality/results/saved_state_mobilenet_v2_256_win_3_lyrsgamma_aug_corr___1592752657 --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/holdout --result-dir /hdd/hanoch/runmodels/img_quality/results/inference_production --outline-pickle-path /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/holdout/quality_holdout_outlines_merged.pkl --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/train_test_val_quality_merged_tile_tile.csv --result-pickle-path inference.pkl 


nohup python -u test_blind_quality.py --model-path /hdd/hanoch/runmodels/img_quality/results/saved_state_mobilenet_v2_256_win_3_lyrsgamma_aug_corr___1592752657 --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/holdout --result-dir /hdd/hanoch/runmodels/img_quality/results/inference_production --outline-pickle-path /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/holdout/quality_holdout_outlines_merged.pkl --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/train_test_val_quality_merged_tile_tile.csv --result-pickle-file inference.pkl --gpu-id=1 --confidence-threshold 0.18 > ./test_blind_quality.log </dev/null 2>&1 & tail -f test_blind_quality.log

Eoleen
nohup python -u test_blind_quality.py --model-path /hdd/hanoch/runmodels/img_quality/results/saved_state_mobilenet_v2_256_win_3_lyrsgamma_aug_corr___1592752657 --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/EileenQuality_needtoRunblindFinder_4_outline --result-dir /hdd/hanoch/runmodels/img_quality/results/inference_production/best --outline-pickle-path /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/EileenQuality_needtoRunblindFinder_4_outline/merged_outline_Eileen_good_blind_links_upto_24june.pkl --result-pickle-file inference_best_qual.pkl --gpu-id=1 --confidence-threshold 0.18 > ./test_blind_quality_Eileen.log </dev/null 2>&1 & tail -f test_blind_quality_Eileen.log 

nohup python -u test_blind_quality.py --model-path /hdd/hanoch/runmodels/img_quality/results/saved_state_mobilenet_v2_256_win_n_lyrs___1593440243 --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/EileenQuality_needtoRunblindFinder_4_outline --result-dir /hdd/hanoch/runmodels/img_quality/results/inference_production/best --outline-pickle-path /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/EileenQuality_needtoRunblindFinder_4_outline/merged_outline_Eileen_good_blind_links_upto_24june.pkl --result-pickle-file inference_best_qual.pkl --gpu-id=1 --confidence-threshold 0.18 > ./test_blind_quality_Eileen.log </dev/null 2>&1 & tail -f test_blind_quality_Eileen.log
                                     --model-path /hdd/hanoch/runmodels/img_quality/results/saved_state_mobilenet_v2_256_win_n_lyrs___1593440243 --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/EileenQuality_needtoRunblindFinder_4_outline --result-dir /hdd/hanoch/runmodels/img_quality/results/inference_production/best --outline-pickle-path /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/EileenQuality_needtoRunblindFinder_4_outline/merged_14_july.pkl --result-pickle-file inference_best_qual.pkl --gpu-id=1 --confidence-threshold 0.18


nohup python -u test_blind_quality.py --model-path /hdd/hanoch/runmodels/img_quality/results/saved_state_mobilenet_v2_256_win_n_lyrs___1593440243 --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/EileenQuality_needtoRunblindFinder_4_outline --result-dir /hdd/hanoch/runmodels/img_quality/results/inference_production/best --outline-pickle-path /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/EileenQuality_needtoRunblindFinder_4_outline/merged_14_july.pkl --result-pickle-file inference_best_qual.pkl --gpu-id=1 --confidence-threshold 0.18 > ./test_blind_quality_Eileen.log </dev/null 2>&1 & tail -f test_blind_quality_Eileen.log
trainset:
nohup python -u test_blind_quality.py --model-path /hdd/hanoch/runmodels/img_quality/results/saved_state_mobilenet_v2_256_win_n_lyrs___1593440243 --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/train --result-dir /hdd/hanoch/runmodels/img_quality/results/inference_production/trainset --outline-pickle-path /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/train/quality_training_outlines_merged.pkl --result-pickle-file inference_best_qual.pkl --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/train_set_converted2test_quality_tile_tile_dropna.csv --gpu-id=1 --confidence-threshold 0.18 > ./trainset.log </dev/null 2>&1 & tail -f trainset.log

Eileen bad examples

nohup python -u test_blind_quality.py --model-path /hdd/hanoch/runmodels/img_quality/results/saved_state_mobilenet_v2_256_win_n_lyrs___1595244718 --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/EileenQuality_needtoRunblindFinder_4_outline/bad_examples --result-dir /hdd/hanoch/runmodels/img_quality/results/inference_production/best --outline-pickle-path /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/EileenQuality_needtoRunblindFinder_4_outline/Eileen_good_blind_links_5_july_with_bad.pkl --result-pickle-file inference_best_qual.pkl --gpu-id=1 --confidence-threshold 0.18 > ./test_bad_blind_quality_Eileen.log </dev/null 2>&1 & tail -f test_bad_blind_quality_Eileen.log
last:
--model-path /hdd/hanoch/runmodels/img_quality/results/saved_state_mobilenet_v2_256_win_n_lyrs___1593440243 --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/EileenQuality_needtoRunblindFinder_4_outline --result-dir /hdd/hanoch/runmodels/img_quality/results/inference_production/best --outline-pickle-path /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/EileenQuality_needtoRunblindFinder_4_outline/merged_14_july.pkl --result-pickle-file inference_best_qual.pkl --gpu-id=0 --confidence-threshold 0.18
Eileen
--model-path /hdd/hanoch/runmodels/img_quality/results/saved_state_mobilenet_v2_256_win_n_lyrs___1596521630 --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/EileenQuality_needtoRunblindFinder_4_outline --result-dir /hdd/hanoch/runmodels/img_quality/results/inference_production/best --outline-pickle-path /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/EileenQuality_needtoRunblindFinder_4_outline/Eileen_good_blind_links_5_july_with_bad.pkl --result-pickle-file inference_best_qual_eileen_good_bad.pkl --gpu-id=0 --statistics-collect --confidence-threshold 0.18 --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/EileenQuality_needtoRunblindFinder_4_outline/Eileen_good_bad_july_val.csv


Cage test:
nohup python -u test_blind_quality.py --model-path /hdd/hanoch/runmodels/img_quality/results/saved_state_mobilenet_v2_256_win_n_lyrs___1596521630 --database-root /hdd/annotator_uploads/ee4062421d7045f7567ded0b34998a9c/blindImages --result-dir /hdd/hanoch/runmodels/img_quality/results/inference_production/ee4062421d7045f7567ded0b34998a9c --outline-pickle-path /hdd/annotator_uploads/ee4062421d7045f7567ded0b34998a9c/quality.pkl --result-pickle-file inference_best_qual.pkl --gpu-id=0 --confidence-threshold 0.21

stat Eileen
nohup python -u test_blind_quality.py --model-path /hdd/hanoch/runmodels/img_quality/results/saved_state_mobilenet_v2_256_win_n_lyrs___1596521630 --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/EileenQuality_needtoRunblindFinder_4_outline/Eileen_good_bad_july_val.csv --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/EileenQuality_needtoRunblindFinder_4_outline --result-dir /hdd/hanoch/runmodels/img_quality/results/inference_production/best --outline-pickle-path /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/EileenQuality_needtoRunblindFinder_4_outline/all_merge_5_july_pen_cage_date_merge.pkl --result-pickle-file inference_best_qual_eileen_good_bad.pkl --gpu-id=0 --confidence-threshold 0.18 --statistics-collect & tail -f nohup.out

nohup python -u test_blind_quality.py --model-path /hdd/hanoch/runmodels/img_quality/results/saved_state_mobilenet_v2_256_win_n_lyrs___1596521630 --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/EileenQuality_needtoRunblindFinder_4_outline/Eileen_good_bad_july_val.csv --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/EileenQuality_needtoRunblindFinder_4_outline --result-dir /hdd/hanoch/runmodels/img_quality/results/inference_production/best --outline-pickle-path /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/EileenQuality_needtoRunblindFinder_4_outline/all_merge_5_july_pen_cage_date_merge.pkl --result-pickle-file inference_best_qual_eileen_good_bad.pkl --gpu-id=0 --confidence-threshold 0.14 --statistics-collect & tail -f nohup.out

Stat holdout
nohup python -u test_blind_quality.py --model-path /hdd/hanoch/runmodels/img_quality/results/saved_state_mobilenet_v2_256_win_n_lyrs___1596521630 --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/train_test_val_quality_tile_filtered_eileen_good_bad_val.csv --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/holdout --result-dir /hdd/hanoch/runmodels/img_quality/results/inference_production/holdout --outline-pickle-path /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/holdout/quality_holdout_outlines_merged.pkl --result-pickle-file production_holdout_results.pkl --gpu-id=0 --confidence-threshold 0.14 --statistics-collect & tail -f nohup.out

w/ marginal
nohup python -u test_blind_quality.py --model-path /hdd/hanoch/runmodels/img_quality/results/saved_state_mobilenet_v2_256_win_n_lyrs___1596521630 --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/train_test_val_quality_tile_filtered_eileen_good_bad_val.csv --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/holdout --result-dir /hdd/hanoch/runmodels/img_quality/results/inference_production/holdout --outline-pickle-path /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/holdout/quality_holdout_with_marginal_outlines_merged.pkl --result-pickle-file production_holdout_results.pkl --gpu-id=3 --confidence-threshold 0.21  --statistics-collect & tail -f nohup.out
nohup python -u test_blind_quality.py --model-path /hdd/hanoch/runmodels/img_quality/results/saved_state_mobilenet_v2_256_win_n_lyrs___1596521630 --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/train_test_val_quality_tile_filtered_eileen_good_bad_val.csv --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/holdout --result-dir /hdd/hanoch/runmodels/img_quality/results/inference_production/holdout --outline-pickle-path /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/holdout/quality_holdout_with_marginal_outlines_merged.pkl --result-pickle-file production_holdout_results.pkl --gpu-id=0 --confidence-threshold 0.21  --statistics-collect & tail -f nohup.out

eval
nohup python -u test_blind_quality.py --model-path /hdd/hanoch/runmodels/img_quality/results/saved_state_mobilenet_v2_256_win_n_lyrs___1596521630 --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/train_test_val_quality_tile_filtered_eileen_good_bad_val.csv --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/holdout --result-dir /hdd/hanoch/runmodels/img_quality/results/inference_production/trainset --outline-pickle-path /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/holdout/quality_holdout_outlines_merged.pkl --result-pickle-file production_holdout_results.pkl --gpu-id=0 --confidence-threshold 0.21  --image-fusion-voting-th 0.8 & tail -f nohup.out
eval with marginal
nohup python -u test_blind_quality.py --model-path /hdd/hanoch/runmodels/img_quality/results/saved_state_mobilenet_v2_256_win_n_lyrs___1596521630 --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/train_test_val_quality_tile_filtered_eileen_good_bad_val.csv --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/holdout --result-dir /hdd/hanoch/runmodels/img_quality/results/inference_production/holdout --outline-pickle-path /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/holdout/quality_holdout_with_marginal_outlines_merged.pkl --result-pickle-file production_holdout_results.pkl --gpu-id=0 --confidence-threshold 0.21  --image-fusion-voting-th 0.501 & tail -f nohup.out
nohup python -u test_blind_quality.py --model-path /hdd/hanoch/runmodels/img_quality/results/saved_state_mobilenet_v2_256_win_n_lyrs___1596521630 --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/train_test_val_quality_tile_filtered_eileen_good_bad_val.csv --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/holdout --result-dir /hdd/hanoch/runmodels/img_quality/results/inference_production/holdout --outline-pickle-path /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/holdout/quality_holdout_with_marginal_outlines_merged.pkl --result-pickle-file production_holdout_results.pkl --gpu-id=0 --confidence-threshold 0.21 --image-fusion-voting-th 0.501 > ./test_blind.py.log </dev/null 2>&1 & tail -f test_blind.py.log



Cage test:
nohup python -u test_blind_quality.py --model-path /hdd/hanoch/runmodels/img_quality/results/saved_state_mobilenet_v2_256_win_n_lyrs___1596521630 --database-root /hdd/annotator_uploads/ee4062421d7045f7567ded0b34998a9c/blindImages --result-dir /hdd/hanoch/runmodels/img_quality/results/inference_production/ee4062421d7045f7567ded0b34998a9c --outline-pickle-path /hdd/annotator_uploads/ee4062421d7045f7567ded0b34998a9c/quality.pkl --result-pickle-file inference_best_qual.pkl --gpu-id=0 --confidence-threshold 0.21  --image-fusion-voting-th 0.8 --model-path /hdd/hanoch/runmodels/img_quality/results/saved_state_mobilenet_v2_256_win_n_lyrs___1596521630 --database-root /hdd/annotator_uploads/ee4062421d7045f7567ded0b34998a9c/blindImages --result-dir /hdd/hanoch/runmodels/img_quality/results/inference_production/ee4062421d7045f7567ded0b34998a9c --outline-pickle-path /hdd/annotator_uploads/ee4062421d7045f7567ded0b34998a9c/quality.pkl --result-pickle-file inference_best_qual.pkl --gpu-id=3 --confidence-threshold 0.21  --image-fusion-voting-th 0.501  --outline-file-format misc & tail -f nohup.out

nohup python -u test_blind_quality.py --model-path /hdd/hanoch/runmodels/img_quality/results/saved_state_mobilenet_v2_256_win_n_lyrs___1596521630 --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/train_test_val_quality_tile_filtered_eileen_good_bad_val.csv --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/holdout --result-dir /hdd/hanoch/runmodels/img_quality/results/inference_production/holdout_soft_pred --outline-pickle-path /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/holdout/quality_holdout_with_marginal_outlines_merged.pkl --result-pickle-file production_holdout_results.pkl --gpu-id=0 --confidence-threshold 0.21  --statistics-collect & tail -f nohup.out

nohup python -u test_blind_quality.py --model-path /hdd/hanoch/runmodels/img_quality/results/saved_state_mobilenet_v2_256_win_n_lyrs___1596521630 --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/train_test_val_quality_tile_filtered_eileen_good_bad_val.csv --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/holdout --result-dir /hdd/hanoch/runmodels/img_quality/results/inference_production/holdout --outline-pickle-path /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/holdout/quality_holdout_with_marginal_outlines_merged.pkl --result-pickle-file production_holdout_results.pkl --gpu-id=0 --confidence-threshold 0.51 --image-fusion-voting-th 0.2  --image-fusion-voting-method vote_4_good_class > ./test_blind.py.log </dev/null 2>&1 & tail -f test_blind.py.log
# with optimized model

nohup python -u test_blind_quality.py --model-path /hdd/hanoch/runmodels/img_quality/results/saved_state_mobilenet_v2_256_win_n_lyrs___1594033374 --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/train_test_val_quality_tile_filtered_eileen_good_bad_val.csv --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/holdout --result-dir /hdd/hanoch/runmodels/img_quality/results/inference_production/holdout_soft_pred --outline-pickle-path /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/holdout/quality_holdout_with_marginal_outlines_merged.pkl --result-pickle-file production_holdout_results.pkl --gpu-id=0 --confidence-threshold 0.21  --statistics-collect & tail -f nohup.out
nohup python -u test_blind_quality.py --model-path /hdd/hanoch/runmodels/img_quality/results/saved_state_mobilenet_v2_256_win_n_lyrs___1593440243 --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/train_test_val_quality_tile_filtered_eileen_good_bad_val.csv --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/holdout --result-dir /hdd/hanoch/runmodels/img_quality/results/inference_production/holdout_soft_pred --outline-pickle-path /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/holdout/quality_holdout_with_marginal_outlines_merged.pkl --result-pickle-file production_holdout_results.pkl --gpu-id=0 --confidence-threshold 0.21  --statistics-collect & tail -f nohup.out


nohup python -u test_blind_quality.py --model-path /hdd/hanoch/runmodels/img_quality/results/saved_state_mobilenet_v2_256_win_n_lyrs___1594033374 --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/EileenQuality_needtoRunblindFinder_4_outline --result-dir /hdd/hanoch/runmodels/img_quality/results/inference_production/Eileen_5_july --outline-pickle-path /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/EileenQuality_needtoRunblindFinder_4_outline/all_merge_5_july_pen_cage_date_merge.pkl --result-pickle-file inference_best_qual_eileen_good_bad.pkl --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/EileenQuality_needtoRunblindFinder_4_outline/Eileen_good_bad_july_val.csv --confidence-threshold 0.21  --statistics-collect  --gpu-id=3 --process-folder-non-recursively & tail -f nohup.out 

nohup python -u test_blind_quality.py --model-path /hdd/hanoch/runmodels/img_quality/results/saved_state_mobilenet_v2_256_win_n_lyrs___1594033374 --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/train_test_val_quality_tile_filtered_eileen_good_bad_val.csv --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/holdout --result-dir /hdd/hanoch/runmodels/img_quality/results/inference_production/1594033374_model --outline-pickle-path /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/holdout/quality_holdout_with_marginal_outlines_merged.pkl --result-pickle-file production_holdout_results.pkl --gpu-id=3 --confidence-threshold 0.21  --statistics-collect & tail -f nohup.out

last:
--model-path /hdd/hanoch/runmodels/img_quality/results/saved_state_mobilenet_v2_256_win_n_lyrs___1594033374 --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/train_test_val_quality_tile_filtered_eileen_good_bad_val.csv --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/holdout --result-dir /hdd/hanoch/runmodels/img_quality/results/inference_production/holdout_soft_pred --outline-pickle-path /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/holdout/quality_holdout_with_marginal_outlines_merged.pkl --result-pickle-file production_holdout_results.pkl --gpu-id=3 --confidence-threshold 0.21  --statistics-collect

nohup python -u test_blind_quality.py --model-path /hdd/hanoch/runmodels/img_quality/results/saved_state_mobilenet_v2_256_win_n_lyrs___1593440243 --database-root /hdd/annotator_uploads/ee4062421d7045f7567ded0b34998a9c/blindImages --result-dir /hdd/hanoch/runmodels/img_quality/results/inference_production/ee4062421d7045f7567ded0b34998a9c --outline-pickle-path /hdd/annotator_uploads/ee4062421d7045f7567ded0b34998a9c/quality.pkl --result-pickle-file inference_best_qual.pkl --gpu-id=0 --confidence-threshold 0.31 --image-fusion-voting-th 0.7  --image-fusion-voting-method concensus_and_vote --outline-file-format misc & tail -f nohup.out 

run over trainset collect stat
nohup python -u test_blind_quality.py --model-path /hdd/hanoch/runmodels/img_quality/results/saved_state_mobilenet_v2_256_win_n_lyrs___1593440243 --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/train_test_val_quality_tile_filtered_eileen_good_bad_val.csv --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/train --result-dir /hdd/hanoch/runmodels/img_quality/results/inference_production/train --outline-pickle-path /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/train/quality_training_outlines_merged.pkl --result-pickle-file production_holdout_results.pkl --gpu-id=3 --confidence-threshold 0.21  --statistics-collect & tail -f nohup.out

nohup python -u test_blind_quality.py --model-path /hdd/hanoch/runmodels/img_quality/results/saved_state_mobilenet_v2_256_win_n_lyrs___1593440243 --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/train_test_val_quality_tile_filtered_eileen_good_bad_val.csv --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/holdout --result-dir /hdd/hanoch/runmodels/img_quality/results/inference_production/holdout_soft_pred --outline-pickle-path /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/holdout/quality_holdout_with_marginal_outlines_merged.pkl --result-pickle-file production_holdout_results.pkl --gpu-id=3 --confidence-threshold 0.21  --statistics-collect
#cages
nohup python -u test_blind_quality.py --model-path /hdd/hanoch/runmodels/img_quality/results/saved_state_mobilenet_v2_256_win_n_lyrs___1593440243 --database-root /hdd/hanoch/runmodels/img_quality/results/inference_production/cage_images_with_cutout_id --result-dir /hdd/hanoch/runmodels/img_quality/results/inference_production/cage_images_with_cutout_id/temp --outline-pickle-path /hdd/hanoch/runmodels/img_quality/results/inference_production/cage_images_with_cutout_id/pkl/merged_cages_outline.pkl --result-pickle-file inference_best_qual.pkl  --gpu-id=2 --confidence-threshold 0.31 --image-fusion-voting-th 0.7  --image-fusion-voting-method concensus_and_vote --outline-file-format misc  --process-folder-non-recursively > ./script.py.log </dev/null 2>&1 & tail -f script.py.log
Again:1600697434
nohup python -u test_blind_quality.py --model-path /hdd/hanoch/runmodels/img_quality/results/saved_state_mobilenet_v2_256_win_n_lyrs___1600697434 --database-root /hdd/annotator_uploads --result-dir /hdd/hanoch/runmodels/img_quality/results/inference_production/cage_images_with_cutout_id --outline-pickle-path /hdd/hanoch/runmodels/img_quality/results/inference_production/cage_images_with_cutout_id/pkl/merged_cages_outline.pkl --result-pickle-file inference_best_qual.pkl  --gpu-id=3 --confidence-threshold 0.31 --image-fusion-voting-th 0.7  --image-fusion-voting-method concensus_and_vote --outline-file-format misc  --statistics-collect > ./script.py.log </dev/null 2>&1 & tail -f script.py.log


nohup python -u test_blind_quality.py --model-path /hdd/hanoch/runmodels/img_quality/results/saved_state_mobilenet_v2_256_win_n_lyrs___1593440243 --database-root /hdd/annotator_uploads --result-dir /hdd/hanoch/runmodels/img_quality/results/inference_production/cage_images_with_cutout_id/temp --outline-pickle-path /hdd/hanoch/runmodels/img_quality/results/inference_production/cage_images_with_cutout_id/pkl/merged_cages_outline.pkl --result-pickle-file inference_best_qual.pkl  --gpu-id=2 --confidence-threshold 0.31 --image-fusion-voting-th 0.7  --image-fusion-voting-method concensus_and_vote --outline-file-format misc > ./script.py.log </dev/null 2>&1 & tail -f script.py.log

nohup python -u test_blind_quality.py --model-path /hdd/hanoch/runmodels/img_quality/results/saved_state_mobilenet_v2_256_win_n_lyrs___1593440243 --database-root /hdd/annotator_uploads --result-dir /hdd/hanoch/runmodels/img_quality/results/inference_production/cage_images_with_cutout_id/temp --outline-pickle-path /hdd/hanoch/runmodels/img_quality/results/inference_production/cage_images_with_cutout_id/pkl/merged_cages_outline.pkl --result-pickle-file inference_best_qual.pkl  --gpu-id=3 --confidence-threshold 0.31 --image-fusion-voting-th 0.7  --image-fusion-voting-method concensus_and_vote --outline-file-format misc --statistics-collect > ./script.py.log </dev/null 2>&1 & tail -f script.py.log
Eileen - tile_64
nohup python -u test_blind_quality.py --model-path /hdd/hanoch/runmodels/img_quality/results/saved_state_mobilenet_v2_256_win_n_lyrs___1593440243 --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/EileenQuality_needtoRunblindFinder_4_outline/bad_examples --result-dir /hdd/hanoch/runmodels/img_quality/results/inference_production/best/tile_64_eileen_bad --outline-pickle-path /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/EileenQuality_needtoRunblindFinder_4_outline/merged_bad_ex_9_2020_filtered_cutout.pkl --result-pickle-file inference_best_qual.pkl --gpu-id=1 --confidence-threshold 0.18 --process-folder-non-recursively > ./test_bad_blind_quality_Eileen.log </dev/null 2>&1 & tail -f test_bad_blind_quality_Eileen.log
Eileen bad 256
nohup python -u test_blind_quality.py --model-path /hdd/hanoch/runmodels/img_quality/results/saved_state_mobilenet_v2_256_win_n_lyrs___1593440243 --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/EileenQuality_needtoRunblindFinder_4_outline/bad_examples --result-dir /hdd/hanoch/runmodels/img_quality/results/inference_production/best/tile_256_eileen_bad --outline-pickle-path /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/EileenQuality_needtoRunblindFinder_4_outline/merged_bad_ex_9_2020_filtered_cutout.pkl --result-pickle-file inference_best_qual.pkl --gpu-id=1 --confidence-threshold 0.18 --process-folder-non-recursively > ./test_bad_blind_quality_Eileen.log </dev/null 2>&1 & tail -f test_bad_blind_quality_Eileen.log

Tile 64 Eileen
nohup python -u test_blind_quality.py --model-path /hdd/hanoch/runmodels/img_quality/results/tile_64/saved_state_mobilenet_v2_64_win_n_lyrs___1603311582 --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data_64 --result-dir /hdd/hanoch/runmodels/img_quality/results/inference_production/best/tile_64_eileen_bad --outline-pickle-path /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/EileenQuality_needtoRunblindFinder_4_outline/merged_bad_ex_9_2020_filtered_cutout.pkl --result-pickle-file inference_best_qual.pkl --gpu-id=1 --confidence-threshold 0.18 > ./test_bad_blind_quality_Eileen.log </dev/null 2>&1 & tail -f test_bad_blind_quality_Eileen.log

Tile 64 over best model ES=1, test set
nohup python -u test_blind_quality.py --model-path /hdd/hanoch/runmodels/img_quality/results/tile_64/saved_state_mobilenet_v2_64_win_n_lyrs___1603311582 --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/holdout --result-dir /hdd/hanoch/runmodels/img_quality/results/inference_production/best/tile_64_test --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data_64/post_train_eileenVal_test_tile_64.csv  --outline-pickle-path /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/holdout/quality_holdout_with_marginal_outlines_merged.pkl --gpu-id=2 --confidence-threshold 0.26 --statistics-collect --plot-dot-overlay-instead-of-box & tail -f nohup.out 

single channel 256 over cage
nohup python -u test_blind_quality.py --model-path /hdd/hanoch/runmodels/img_quality/results/saved_state_mobilenet_v2_256_win_n_lyrs___1603714813 --database-root /hdd/hanoch/data/cages_holdout/annotated_images_eileen --result-dir /hdd/hanoch/runmodels/img_quality/results/inference_production/single_channel/cage_images_with_cutout_id --outline-pickle-path /hdd/hanoch/runmodels/img_quality/results/inference_production/cage_images_with_cutout_id/pkl/merged_cages_outline.pkl --result-pickle-file inference_best_qual.pkl  --gpu-id=3 --confidence-threshold 0.2 --image-fusion-voting-th 0.7  --image-fusion-voting-method concensus_and_vote --outline-file-format misc  --statistics-collect --single-channel-input-grey > ./scriptSingleChan.py.log </dev/null 2>&1 & tail -f scriptSingleChan.py.log

Weighed hue over cages
nohup python -u test_blind_quality.py --model-path /hdd/hanoch/runmodels/img_quality/results/hue_weighted_norm/reg/saved_state_mobilenet_v2_256_win_n_lyrs___1604507681  --database-root /hdd/hanoch/data/cages_holdout/annotated_images_eileen --result-dir /hdd/hanoch/runmodels/img_quality/results/inference_production/hue_weighted --outline-pickle-path /hdd/hanoch/runmodels/img_quality/results/inference_production/cage_images_with_cutout_id/pkl/merged_cages_outline.pkl --result-pickle-file inference_best_qual.pkl --gpu-id=3 --confidence-threshold 0.2 --image-fusion-voting-th 0.7  --image-fusion-voting-method concensus_and_vote --outline-file-format misc  --statistics-collect --hue-norm-preprocess > ./scripthueWeigthedChan.py.log </dev/null 2>&1 & tail -f scripthueWeigthedChan.py.log

Vanilla reg/ES over cages 

nohup python -u test_blind_quality.py --model-path /hdd/hanoch/runmodels/img_quality/results/vanilla/saved_state_mobilenet_v2_256_win_n_lyrs___1604591734  --database-root /hdd/hanoch/data/cages_holdout/annotated_images_eileen --result-dir /hdd/hanoch/runmodels/img_quality/results/inference_production/vanilla --outline-pickle-path /hdd/hanoch/runmodels/img_quality/results/inference_production/cage_images_with_cutout_id/pkl/merged_cages_outline.pkl --result-pickle-file inference_best_qual.pkl --gpu-id=3 --confidence-threshold 0.2 --image-fusion-voting-th 0.7  --image-fusion-voting-method concensus_and_vote --outline-file-format misc  --statistics-collect  > ./scripVanila.py.log </dev/null 2>&1 & tail -f scripVanila.py.log

nohup python -u test_blind_quality.py --model-path /hdd/hanoch/runmodels/img_quality/results/hue_weighted_norm/reg/saved_state_mobilenet_v2_256_win_n_lyrs___1604507681  --database-root /hdd/hanoch/data/cages_holdout/annotated_images_eileen --result-dir /hdd/hanoch/runmodels/img_quality/results/inference_production/hue_weighted --outline-pickle-path /hdd/hanoch/runmodels/img_quality/results/inference_production/cage_images_with_cutout_id/pkl/merged_cages_outline.pkl --result-pickle-file inference_best_qual.pkl --gpu-id=3 --confidence-threshold 0.25 --image-fusion-voting-th 0.7  --image-fusion-voting-method concensus_and_vote --outline-file-format misc  --statistics-collect --hue-norm-preprocess --process-folder-non-recursively > ./scripthueWeigthedChan.py.log </dev/null 2>&1 & tail -f scripthueWeigthedChan.py.log 

nohup python -u test_blind_quality.py --model-path /hdd/hanoch/runmodels/img_quality/results/hue_weighted_norm/reg/saved_state_mobilenet_v2_256_win_n_lyrs___1604507681  --database-root /hdd/hanoch/data/cages_holdout/annotated_images_eileen --result-dir /hdd/hanoch/runmodels/img_quality/results/inference_production/hue_weighted --outline-pickle-path /hdd/hanoch/runmodels/img_quality/results/inference_production/cage_images_with_cutout_id/pkl/merged_cages_outline.pkl --result-pickle-file inference_best_qual.pkl --gpu-id=3 --confidence-threshold 0.25 --image-fusion-voting-th 0.7  --image-fusion-voting-method concensus_and_vote --outline-file-format misc  --statistics-collect --hue-norm-preprocess --process-folder-non-recursively > ./scripthueWeigthedChan.py.log </dev/null 2>&1 & tail -f scripthueWeigthedChan.py.log 

cages eval marignals
nohup python -u test_blind_quality.py --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/clip_grad_new_norm/sgd_filt_list/saved_state_mobilenet_v2_256_win_n_lyrs___1606226252 --database-root /hdd/hanoch/data/cages_holdout/annotated_images_eileen --result-dir /hdd/hanoch/runmodels/img_quality/results/inference_production/cages_252 --outline-pickle-path /hdd/hanoch/runmodels/img_quality/results/inference_production/cage_images_with_cutout_id/pkl/merged_cages_outline.pkl --result-pickle-file inference_best_qual.pkl --gpu-id=3 --confidence-threshold 0.1 --image-fusion-voting-th 0.7  --image-fusion-voting-method concensus_and_vote --outline-file-format misc  --statistics-collect --process-folder-non-recursively > ./scripthueWeigthedChan.py.log </dev/null 2>&1 & tail -f scripthueWeigthedChan.py.log 
cages_ new annotation 298
nohup python -u test_blind_quality.py --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/clip_grad_new_norm/sgd_filt_list/saved_state_mobilenet_v2_256_win_n_lyrs___1606226252 --database-root /hdd/hanoch/data/cages_holdout/annotated_images_eileen_batch2 --result-dir /hdd/hanoch/data/cages_holdout/annotated_images_eileen_batch2/res --outline-pickle-path /hdd/hanoch/data/cages_holdout/merged_cages_outline.pkl --result-pickle-file inference_best_qual.pkl --gpu-id=3 --confidence-threshold 0.1 --image-fusion-voting-th 0.7  --image-fusion-voting-method concensus_and_vote --outline-file-format misc  --statistics-collect --process-folder-non-recursively > ./scripthueWeigthedChan.py.log </dev/null 2>&1 & tail -f scripthueWeigthedChan.py.log 

Cages Fitjar
nohup python -u src/CNN/test_blind_quality.py --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/clip_grad_new_norm/sgd_filt_list/saved_state_mobilenet_v2_256_win_n_lyrs___1606226252 --database-root /hdd/hanoch/data/cages_holdout/annotated_images_eileen_fitjar --result-dir /hdd/hanoch/data/cages_holdout/annotated_images_eileen_fitjar/res --outline-pickle-path /hdd/hanoch/data/cages_holdout/omri_outfulljoin7_2020_fname_keys.pkl --result-pickle-file inference_best_qual.pkl --gpu-id=3 --confidence-threshold 0.1 --image-fusion-voting-th 0.7  --image-fusion-voting-method concensus_and_vote --outline-file-format misc  --statistics-collect --process-folder-non-recursively > ./scripthueWeigthedChan.py.log </dev/null 2>&1 & tail -f scripthueWeigthedChan.py.log 

Flushed cage : /hdd/blind_raw_images/ee40624
nohup python -u src/CNN/test_blind_quality.py --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/clip_grad_new_norm/sgd_filt_list/saved_state_mobilenet_v2_256_win_n_lyrs___1606226252 --database-root /hdd/blind_raw_images/ee40624 --result-dir /hdd/blind_raw_images/ee40624/res --outline-pickle-path /hdd/blind_raw_images/ee40624/merged_outlines_json.pkl --result-pickle-file inference_best_qual.pkl --gpu-id=3 --confidence-threshold 0.1 --image-fusion-voting-th 0.25  --image-fusion-voting-method vote_4_good_class --outline-file-format misc --statistics-collect --process-folder-non-recursively > ./scripthueWeigthedChan.py.log </dev/null 2>&1 & tail -f scripthueWeigthedChan.py.log 

Run again Matt holdout
nohup python -u src/CNN/test_blind_quality.py --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/clip_grad_new_norm/sgd_filt_list/saved_state_mobilenet_v2_256_win_n_lyrs___1606226252 --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/file_quality_tile_eileen_good_bad_val_bad_9_20_avg_pool_filt_conf_filt_no_edges_trn_tst_fitjar.csv --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/holdout --result-dir /hdd/hanoch/runmodels/img_quality/results/inference_production/holdout_soft_pred_new_model --outline-pickle-path /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/holdout/quality_holdout_with_marginal_outlines_merged.pkl --result-pickle-file production_holdout_results.pkl --gpu-id=2 --image-fusion-voting-th 0.25  --image-fusion-voting-method vote_4_good_class --confidence-threshold 0.1  --statistics-collect > ./matt_holdout.py.log </dev/null 2>&1 & tail -f matt_holdout.py.log

Cages over model with Fitjar
--model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/clip_grad_new_norm/fitjar/saved_state_mobilenet_v2_256_win_n_lyrs___1609764290 
nohup python -u test_blind_quality.py --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/clip_grad_new_norm/fitjar/saved_state_mobilenet_v2_256_win_n_lyrs___1609764290  --database-root /hdd/hanoch/data/cages_holdout/annotated_images_eileen_batch2 --result-dir /hdd/hanoch/data/cages_holdout/annotated_images_eileen_batch2/res_1609764290 --outline-pickle-path /hdd/hanoch/data/cages_holdout/merged_cages_outline.pkl --result-pickle-file inference_best_qual.pkl --gpu-id=3 --confidence-threshold 0.7 --image-fusion-voting-th 0.85  --image-fusion-voting-method concensus_and_vote --outline-file-format misc  --statistics-collect --process-folder-non-recursively > ./scriptCagesEvalModle.py.log </dev/null 2>&1 & tail -f scriptCagesEvalModle.py.log 

nohup python -u ./src/CNN/test_blind_quality.py --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/clip_grad_new_norm/fitjar/saved_state_mobilenet_v2_256_win_n_lyrs___1609764290  --database-root /hdd/hanoch/data/cages_holdout/annotated_images_eileen_batch2 --result-dir /hdd/hanoch/data/cages_holdout/annotated_images_eileen/res_1609764290 --outline-pickle-path /hdd/hanoch/data/cages_holdout/merged_cages_outline.pkl --result-pickle-file inference_best_qual.pkl --gpu-id=2 --confidence-threshold 0.1 --image-fusion-voting-th 0.25  --image-fusion-voting-method concensus_and_vote --outline-file-format misc  --statistics-collect --process-folder-non-recursively > ./scriptCagesEvalModle2.py.log </dev/null 2>&1 & tail -f scriptCagesEvalModle2.py.log 

Flushed cage : with Fitjar model with right th to see softmap /hdd/blind_raw_images/ee40624
nohup python -u src/CNN/test_blind_quality.py --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/clip_grad_new_norm/fitjar/saved_state_mobilenet_v2_256_win_n_lyrs___1609764290 --database-root /hdd/blind_raw_images/ee40624 --result-dir /hdd/blind_raw_images/ee40624/res_1609764290 --outline-pickle-path /hdd/blind_raw_images/ee40624/merged_outlines_json.pkl --result-pickle-file inference_best_qual.pkl --gpu-id=3 --confidence-threshold 0.7 --image-fusion-voting-th 0.85  --image-fusion-voting-method vote_4_good_class --outline-file-format misc --statistics-collect --process-folder-non-recursively > ./scripthueWeigthedChan.py.log </dev/null 2>&1 & tail -f scripthueWeigthedChan.py.log 

Cages model with Fitjar :efficientNetB0

nohup python -u ./src/CNN/test_blind_quality.py efficientnet_b0_w256 --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/clip_grad_new_norm/fitjar/saved_state_efficientnet_b0_w256___1609842424  --database-root /hdd/hanoch/data/cages_holdout/annotated_images_eileen_batch2 --result-dir /hdd/hanoch/data/cages_holdout/annotated_images_eileen_batch2/res_EfficientNetB0_1609842424 --outline-pickle-path /hdd/hanoch/data/cages_holdout/merged_cages_outline.pkl --result-pickle-file inference_best_qual.pkl --gpu-id=3 --confidence-threshold 0.87 --image-fusion-voting-th 0.8  --image-fusion-voting-method vote_4_good_class --outline-file-format misc  --statistics-collect --process-folder-non-recursively > ./scriptCagesEvalModle.py.log </dev/null 2>&1 & tail -f scriptCagesEvalModle.py.log 

HCF+balanced batch
nohup python -u ./src/CNN/test_blind_quality.py mobilenet_v2_2FC_w256_nlyrs --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/hcf_and_balanced_sampling/saved_state_mobilenet_v2_2FC_w256_nlyrs___1610488946  --database-root /hdd/hanoch/data/cages_holdout/annotated_images_eileen_batch2 --result-dir /hdd/hanoch/data/cages_holdout/annotated_images_eileen_batch2/res_hcf_balancedd_batch_1610488946 --outline-pickle-path /hdd/hanoch/data/cages_holdout/merged_cages_outline.pkl --result-pickle-file inference_best_qual.pkl --metadata-json-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/hcf_and_balanced_sampling/mobilenet_v2_2FC_w256_nlyrs_1610488946.json --gpu-id=3 --confidence-threshold 0.87 --image-fusion-voting-th 0.8  --image-fusion-voting-method vote_4_good_class --outline-file-format misc  --statistics-collect --process-folder-non-recursively > ./scriptCagesEvalModle.py.log  </dev/null 2>&1 & tail -f scriptCagesEvalModle.py.log 

final model
nohup python -u ./src/CNN/test_blind_quality.py mobilenet_v2_256_win_n_lyrs --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/clip_grad_new_norm/fitjar/saved_state_mobilenet_v2_256_win_n_lyrs___1609764290  --database-root /hdd/hanoch/data/cages_holdout/annotated_images_eileen_batch2 --result-dir /hdd/hanoch/data/cages_holdout/annotated_images_eileen_batch2/res_1609764290_th2 --outline-pickle-path /hdd/hanoch/data/cages_holdout/merged_cages_outline.pkl --result-pickle-file inference_best_qual.pkl --gpu-id=2 --confidence-threshold 0.75 --image-fusion-voting-th 0.7  --image-fusion-voting-method concensus_and_vote --outline-file-format misc  --statistics-collect --process-folder-non-recursively > ./scriptCagesEvalModle2.py.log </dev/null 2>&1 & tail -f scriptCagesEvalModle2.py.log 

mobilenet_v2_256_win_n_lyrs --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/clip_grad_new_norm/fitjar/saved_state_mobilenet_v2_256_win_n_lyrs___1609764290 --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/holdout --result-dir /hdd/hanoch/runmodels/img_quality/results/inference_production/holdout_soft_pred_1609764290 --outline-pickle-path /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/holdout/quality_holdout_with_marginal_outlines_merged.pkl --result-pickle-file inference_best_qual.pkl --gpu-id=3 --confidence-threshold 0.75 --image-fusion-voting-th 0.7  --image-fusion-voting-method concensus_and_vote --statistics-collect --process-folder-non-recursively

Remve tile on edges

nohup python -u test_blind_quality.py mobilenet_v2_256_win_n_lyrs --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/clip_grad_new_norm/fitjar/saved_state_mobilenet_v2_256_win_n_lyrs___1609764290 --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/EileenQuality_needtoRunblindFinder_4_outline/all_cutouts --result-dir /hdd/hanoch/runmodels/img_quality/results/inference_production/best/tile_eileen_val_remove_edges --outline-pickle-path /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/EileenQuality_needtoRunblindFinder_4_outline/merged_bad_ex_9_2020_filtered_cutout.pkl --result-pickle-file inference_best_qual.pkl --gpu-id=0 --confidence-threshold 0.75 --image-fusion-voting-th 0.7  --image-fusion-voting-method concensus_and_vote --statistics-collect --process-folder-non-recursively > ./test_bad_blind_quality_Eileen.log </dev/null 2>&1 & tail -f test_bad_blind_quality_Eileen.log

#eval with annotator grades 
nohup python -u test_blind_quality.py mobilenet_v2_256_win_n_lyrs --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/clip_grad_new_norm/fitjar/saved_state_mobilenet_v2_256_win_n_lyrs___1609764290 --database-root /hdd/hanoch/data/database/png_blind_q/png_with_q_grade --result-dir /hdd/hanoch/runmodels/img_quality/results/inference_production/best/quality_grades_db_eval --outline-pickle-path /hdd/hanoch/data/database/blind_quality/cutouts_path_blind_q1_1900.pkl --result-pickle-file inference_best_qual.pkl --gpu-id=0 --confidence-threshold 0.75 --image-fusion-voting-th 0.85  --image-fusion-voting-method concensus_and_vote --statistics-collect --process-folder-non-recursively > ./test_annotator_quality.log </dev/null 2>&1 & tail -f test_annotator_quality.log

Marginal cages create tiles 
nohup python -u test_blind_quality.py mobilenet_v2_256_win_n_lyrs --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/clip_grad_new_norm/fitjar/saved_state_mobilenet_v2_256_win_n_lyrs___1609764290 --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/cages_marginal --outline-pickle-path /hdd/hanoch/data/cages_holdout/merged_cages_outline.pkl --result-dir /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/cages_marginal/tiles_fixed_cutout_id --result-pickle-file inference_best_qual.pkl --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/cages_marginal/merged_eileen_blind_tests_marginals_cutout_uuid.csv --gpu-id=0 --confidence-threshold 0.75 --image-fusion-voting-th 0.85  --image-fusion-voting-method concensus_and_vote --statistics-collect --process-folder-non-recursively --outline-file-format fname_n_cutout_id > ./test_annotator_quality.log </dev/null 2>&1 & tail -f test_annotator_quality.

#eval with annotator grades large DB 
nohup python -u ./src/CNN/test_blind_quality.py mobilenet_v2_256_win_n_lyrs --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/clip_grad_new_norm/fitjar/saved_state_mobilenet_v2_256_win_n_lyrs___1609764290 --database-root /hdd/hanoch/data/database/png_blind_q_2nd_batch_2529 --result-dir /hdd/hanoch/runmodels/img_quality/results/inference_production/best/quality_grades_db_eval_2529 --outline-pickle-path /hdd/hanoch/data/database/png_blind_q_2nd_batch_2529/query_with_quality1_1900.pkl --result-pickle-file inference_best_qual.pkl --gpu-id=0 --confidence-threshold 0.85 --image-fusion-voting-th 0.85  --image-fusion-voting-method concensus_and_vote --statistics-collect --process-folder-non-recursively --outline-file-format fname_n_cutout_id > ./test_annotator_quality.log </dev/null 2>&1 & tail -f test_annotator_quality.log

#eval Gated attention 
nohup python -u ./src/CNN/test_blind_quality.py mobilenet_v2_2FC_w256_fusion_3cls --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/embedding_fusion_nonlinear_fix_bug/saved_state_mobilenet_v2_2FC_w256_fusion_3cls___1616951762 --database-root /hdd/hanoch/debug/gated_fusion_conf_mat/act_0_det_1 --dataset-split-csv /hdd/hanoch/debug/gated_fusion_conf_mat/act_0_det_1/annotations.csv --outline-pickle-path /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/EileenQuality_needtoRunblindFinder_4_outline/merged_bad_ex_9_2020_filtered_cutout.pkl --result-dir /hdd/hanoch/debug/gated_fusion_conf_mat/act_0_det_1/tiles --confidence-threshold 0.85 --gpu-id 0 --fine-tune-pretrained-model-plan freeze_pretrained_add_gated_atten --classify-image-all-tiles --statistics-collect --process-folder-non-recursively

#eval with FP DB 
nohup python -u ./src/CNN/test_blind_quality.py mobilenet_v2_256_win_n_lyrs --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/clip_grad_new_norm/fitjar/saved_state_mobilenet_v2_256_win_n_lyrs___1609764290 --database-root /hdd/hanoch/debug/gated_fusion_conf_mat/act_0_det_1 --result-dir /hdd/hanoch/debug/gated_fusion_conf_mat/act_0_det_1/tiles --outline-pickle-path /hdd/hanoch/data/database/png_blind_q_2nd_batch_2529/query_with_quality1_1900.pkl --result-pickle-file inference_best_qual.pkl --gpu-id=0 --confidence-threshold 0.85 --image-fusion-voting-th 0.85  --image-fusion-voting-method concensus_and_vote --statistics-collect --process-folder-non-recursively --outline-file-format fname_n_cutout_id > ./test_annotator_quality.log </dev/null 2>&1 & tail -f test_annotator_quality.log

#Gated attention eval with annotator grades 
nohup python -u ./src/CNN/test_blind_quality.py mobilenet_v2_2FC_w256_fusion_3cls --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/embedding_fusion_nonlinear_fix_bug/saved_state_mobilenet_v2_2FC_w256_fusion_3cls___1616951762 --database-root /hdd/hanoch/data/database/png_blind_q/png_with_q_grade --outline-pickle-path  /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/merged_outlines.pkl --result-dir /hdd/hanoch/data/database/png_blind_q/png_with_q_grade/gated_attention --gpu-id 3 --fine-tune-pretrained-model-plan freeze_pretrained_add_gated_atten --classify-image-all-tiles --statistics-collect --process-folder-non-recursively 
nohup python -u  ./modules/test_blind_quality.py mobilenet_v2_2FC_w256_fusion_3cls --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/gated_atten_1FC/saved_state_mobilenet_v2_2FC_w256_fusion_3cls___1620073199 --database-root /hdd/hanoch/data/database/png_blind_q/png_with_q_grade --outline-pickle-path  /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/merged_outlines.pkl --result-dir /hdd/hanoch/data/database/png_blind_q/png_with_q_grade/gated_attention/1FC_all_dropout --gpu-id 3 --fine-tune-pretrained-model-plan freeze_pretrained_add_gated_atten --classify-image-all-tiles --fc-sequential-type 1FC_all_dropout --statistics-collect --process-folder-non-recursively 
3with dropout
nohup python -u  ./modules/test_blind_quality.py mobilenet_v2_2FC_w256_fusion_3cls --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/gated_atten_1FC/saved_state_mobilenet_v2_2FC_w256_fusion_3cls___1620134168 --database-root /hdd/hanoch/data/database/png_blind_q/png_with_q_grade --outline-pickle-path  /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/merged_outlines.pkl --result-dir /hdd/hanoch/data/database/png_blind_q/png_with_q_grade/gated_attention/1FC_all_dropout_1620134168 --gpu-id 3 --fine-tune-pretrained-model-plan freeze_pretrained_add_gated_atten --classify-image-all-tiles --fc-sequential-type 1FC_all_dropout --statistics-collect --process-folder-non-recursively 


tta over tile based model 
python -u ./src/CNN/test_blind_quality.py mobilenet_v2_256_win_n_lyrs --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/clip_grad_new_norm/fitjar/saved_state_mobilenet_v2_256_win_n_lyrs___1609764290 --database-root /hdd/hanoch/data/database/png_blind_q/png_with_q_grade --result-dir /hdd/hanoch/data/database/png_blind_q/png_with_q_grade/tta --outline-pickle-path /hdd/hanoch/data/database/blind_quality/cutouts_path_blind_q1_1900.pkl --result-pickle-file inference_best_qual.pkl --gpu-id=0 --confidence-threshold 0.75 --image-fusion-voting-th 0.85  --image-fusion-voting-method concensus_and_vote --statistics-collect --process-folder-non-recursively --tta 2 >> ./tta_inf_production.py.log </dev/null 2>&1

add tile location out of NxM metadata only
nohup python -u test_blind_quality.py mobilenet_v2_256_win_n_lyrs --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/clip_grad_new_norm/fitjar/saved_state_mobilenet_v2_256_win_n_lyrs___1609764290 --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/EileenQuality_needtoRunblindFinder_4_outline/all_cutouts --result-dir /hdd/hanoch/runmodels/img_quality/results/inference_production/best/tile_eileen_val_remove_edges --outline-pickle-path /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/merged_outlines.pkl --result-pickle-file inference_best_qual.pkl --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/train_test_val_quality_tile_filtered_eileen_good_bad_val.csv --gpu-id=0 --confidence-threshold 0.75 --image-fusion-voting-th 0.7  --image-fusion-voting-method concensus_and_vote --statistics-collect --process-folder-non-recursively --calculate-metadata-only> ./test_bad_blind_quality_Eileen.log </dev/null 2>&1 & tail -f test_bad_blind_quality_Eileen.log
nohup python -u test_blind_quality.py mobilenet_v2_256_win_n_lyrs --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/clip_grad_new_norm/fitjar/saved_state_mobilenet_v2_256_win_n_lyrs___1609764290 --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/EileenQuality_needtoRunblindFinder_4_outline/all_cutouts --result-dir /hdd/hanoch/runmodels/img_quality/results/inference_production/best/tile_eileen_val_remove_edges --outline-pickle-path /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/merged_outlines.pkl --result-pickle-file inference_best_qual.pkl --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/train_test_val_quality_tile_filtered_eileen_good_bad_val.csv --gpu-id=0 --confidence-threshold 0.75 --image-fusion-voting-th 0.7  --image-fusion-voting-method concensus_and_vote --statistics-collect --process-folder-non-recursively --calculate-metadata-only> ./test_bad_blind_quality_Eileen.log </dev/null 2>&1 & tail -f test_bad_blind_quality_Eileen.log
nohup python -u test_blind_quality.py mobilenet_v2_256_win_n_lyrs --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/clip_grad_new_norm/sgd_filt_list/saved_state_mobilenet_v2_256_win_n_lyrs___1606226252 --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/fitjar --result-dir /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/fitjar/res --outline-pickle-path /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/merged_outlines.pkl --result-pickle-file inference_best_qual.pkl --gpu-id=3 --confidence-threshold 0.1 --image-fusion-voting-th 0.7  --image-fusion-voting-method concensus_and_vote --outline-file-format fname_n_cutout_id --statistics-collect --process-folder-non-recursively --calculate-metadata-only > ./scripthueWeigthedChan.py.log </dev/null 2>&1 & tail -f scripthueWeigthedChan.py.log


nohup python -u test_blind_quality.py mobilenet_v2_256_win_n_lyrs --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/clip_grad_new_norm/fitjar/saved_state_mobilenet_v2_256_win_n_lyrs___1609764290 --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/cages_marginal --outline-pickle-path /hdd/hanoch/data/cages_holdout/merged_cages_outline.pkl --result-dir /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/cages_marginal/res --result-pickle-file inference_best_qual.pkl --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/cages_marginal/merged_eileen_blind_tests_marginals_cutout_uuid.csv --gpu-id=0 --confidence-threshold 0.75 --image-fusion-voting-th 0.85  --image-fusion-voting-method concensus_and_vote --statistics-collect --process-folder-non-recursively --outline-file-format fname_n_cutout_id --calculate-metadata-only > ./test_annotator_quality.log </dev/null 2>&1 & tail -f test_annotator_quality.

create Tiles with tile position
nohup python -u test_blind_quality.py mobilenet_v2_256_win_n_lyrs --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/clip_grad_new_norm/sgd_filt_list/saved_state_mobilenet_v2_256_win_n_lyrs___1606226252 --database-root /hdd/hanoch/data/cages_holdout/annotated_images_eileen_batch2 --result-dir /hdd/hanoch/data/cages_holdout/annotated_images_eileen_batch2/res --outline-pickle-path /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/merged_outlines.pkl --result-pickle-file inference_best_qual.pkl --gpu-id=3 --confidence-threshold 0.85 --image-fusion-voting-th 0.85  --image-fusion-voting-method concensus_and_vote --outline-file-format misc  --statistics-collect --process-folder-non-recursively > ./scripthueWeigthedChan.py.log </dev/null 2>&1 & tail -f scripthueWeigthedChan.py.log 

                                     mobilenet_v2_256_win_n_lyrs --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/clip_grad_new_norm/sgd_filt_list/saved_state_mobilenet_v2_256_win_n_lyrs___1606226252 --database-root /hdd/hanoch/data/cages_holdout/annotated_images_eileen_batch2 --result-dir /hdd/hanoch/data/cages_holdout/annotated_images_eileen_batch2/res --outline-pickle-path /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/merged_outlines.pkl --result-pickle-file inference_best_qual.pkl --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/4_5cages_298ex$ cat list_4_5cages_blind_test_results_eileen_2ndBatch.xlsx --gpu-id=3 --confidence-threshold 0.85 --image-fusion-voting-th 0.85  --image-fusion-voting-method concensus_and_vote --outline-file-format misc  --statistics-collect --process-folder-non-recursively
Eval with 592 images annotator
3with dropout
nohup python -u  ./dev/test_blind_quality.py mobilenet_v2_2FC_w256_fusion_3cls --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/additive_raster_bitmap_8_8_mlp_1D/saved_state_mobilenet_v2_2FC_w256_fusion_3cls___1623655509 --database-root /hdd/hanoch/data/database/png_blind_q/png_with_q_grade --outline-pickle-path  /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/merged_outlines.pkl --result-dir /hdd/hanoch/data/database/png_blind_q/png_with_q_grade/gated_attention/1623655509 --gpu-id 3 --fine-tune-pretrained-model-plan freeze_pretrained_add_gated_atten --pooling-at-classifier  --classify-image-all-tiles --positional-embeddings additive_raster_bitmap_8_8_mlp_1D --statistics-collect --process-folder-non-recursively

back to gated only after ful untrained layers
nohup python -u  ./dev/test_blind_quality.py mobilenet_v2_2FC_w256_fusion_3cls --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/gated_atten_resotre_fix_gradtrue/saved_state_mobilenet_v2_2FC_w256_fusion_3cls___1624222383 --database-root /hdd/hanoch/data/database/png_blind_q/png_with_q_grade --outline-pickle-path  /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/merged_outlines.pkl --dataset-split-csv /hdd/hanoch/data/database/png_blind_q/png_with_q_grade/cutouts_path_blind_q1_1900meta_2nd_review.csv --result-dir /hdd/hanoch/data/database/png_blind_q/png_with_q_grade/gated_attention/1624222383 --gpu-id 3 --fine-tune-pretrained-model-plan freeze_pretrained_add_gated_atten --classify-image-all-tiles --statistics-collect --process-folder-non-recursively --confidence-threshold 0.6   

create Tiles with new 7817 quality grade
nohup python -u test_blind_quality.py mobilenet_v2_256_win_n_lyrs --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/clip_grad_new_norm/sgd_filt_list/saved_state_mobilenet_v2_256_win_n_lyrs___1606226252 --database-root /hdd/hanoch/data/database/blind_quality/quality_based_all_annotations_24_6/png --result-dir /hdd/hanoch/data/database/blind_quality/quality_based_all_annotations_24_6/png/tiles --outline-pickle-path /hdd/hanoch/data/database/blind_quality/quality_based_all_annotations_24_6/quality_based_all_annotations_24_6_CU.pkl --result-pickle-file inference_best_qual.pkl --gpu-id=3 --confidence-threshold 0.85 --image-fusion-voting-th 0.85  --image-fusion-voting-method concensus_and_vote --outline-file-format fname_n_cutout_id  --statistics-collect --process-folder-non-recursively > ./scripthueWeigthedChan.py.log </dev/null 2>&1 & tail -f scripthueWeigthedChan.py.log 

Cages 298 - holdout set
nohup python -u ./src/CNN/test_blind_quality.py mobilenet_v2_256_win_n_lyrs --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/clip_grad_new_norm/fitjar/saved_state_mobilenet_v2_256_win_n_lyrs___1609764290 --database-root /hdd/hanoch/data/cages_holdout/annotated_images_eileen_batch2 --result-dir /hdd/hanoch/data/cages_holdout/annotated_images_eileen_batch2/res_with_svm --outline-pickle-path /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/merged_outlines.pkl --result-pickle-file inference_best_qual.pkl --gpu-id=2 --confidence-threshold 0.85 --image-fusion-voting-th 0.85 --image-fusion-voting-method concensus_and_vote --statistics-collect --process-folder-non-recursively --outline-file-format misc >> ./annotated_images_eileen_batch2.py.log </dev/null 2>&1 & tail -f ./annotated_images_eileen_batch2.py.log 

Cages 298 new data collection 7818
nohup python -u ./dev/test_blind_quality.py mobilenet_v2_256_win_n_lyrs --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/new_db_7818_bin_class_penn_test_restore/saved_state_mobilenet_v2_256_win_n_lyrs___1625540111 --database-root /hdd/hanoch/data/cages_holdout/annotated_images_eileen_batch2 --result-dir /hdd/hanoch/data/cages_holdout/annotated_images_eileen_batch2/new_db_1625540111 --outline-pickle-path /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/merged_outlines.pkl --result-pickle-file inference_best_qual.pkl --gpu-id=2 --confidence-threshold 0.85 --image-fusion-voting-th 0.85 --image-fusion-voting-method concensus_and_vote --statistics-collect --process-folder-non-recursively --outline-file-format misc >> ./annotated_images_eileen_batch2.py.log </dev/null 2>&1 & tail -f ./annotated_images_eileen_batch2.py.log 

saved_state_mobilenet_v2_256_win_n_lyrs___1625642655
nohup python -u ./dev/test_blind_quality.py mobilenet_v2_256_win_n_lyrs --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/new_db_7818_bin_class_penn_test_restore/saved_state_mobilenet_v2_256_win_n_lyrs___1625642655 --database-root /hdd/hanoch/data/cages_holdout/annotated_images_eileen_batch2 --result-dir /hdd/hanoch/data/cages_holdout/annotated_images_eileen_batch2/new_db_1625642655 --outline-pickle-path /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/merged_outlines.pkl --result-pickle-file inference_best_qual.pkl --gpu-id=1 --confidence-threshold 0.85 --image-fusion-voting-th 0.85 --image-fusion-voting-method concensus_and_vote --statistics-collect --process-folder-non-recursively --outline-file-format misc >> ./annotated_images_eileen_batch2.py.log </dev/null 2>&1 & tail -f ./annotated_images_eileen_batch2.py.log

/hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/new_db_7818_bin_class_penn_regularization/saved_state_mobilenet_v2_256_win_n_lyrs___1625844658
nohup python -u ./dev/test_blind_quality.py mobilenet_v2_256_win_n_lyrs --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/new_db_7818_bin_class_penn_regularization/saved_state_mobilenet_v2_256_win_n_lyrs___1625844658 --database-root /hdd/hanoch/data/cages_holdout/annotated_images_eileen_batch2 --result-dir /hdd/hanoch/data/cages_holdout/annotated_images_eileen_batch2/new_db_1625844658 --outline-pickle-path /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/merged_outlines.pkl --result-pickle-file inference_best_qual.pkl --gpu-id=3 --confidence-threshold 0.85 --image-fusion-voting-th 0.85 --image-fusion-voting-method concensus_and_vote --statistics-collect --process-folder-non-recursively --outline-file-format misc >> ./annotated_images_eileen_batch2.py.log </dev/null 2>&1 & tail -f ./annotated_images_eileen_batch2.py.log

mobilenet_v2_256_win_n_lyrs --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/new_db_7818_bin_class_penn_regularization/saved_state_mobilenet_v2_256_win_n_lyrs___1625844658 --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/holdout --result-dir /hdd/hanoch/data/cages_holdout/annotated_images_eileen_batch2/new_db_1625844658 --outline-pickle-path /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/merged_outlines.pkl --result-pickle-file inference_best_qual.pkl --gpu-id=3 --confidence-threshold 0.85 --image-fusion-voting-th 0.85 --image-fusion-voting-method concensus_and_vote --statistics-collect --process-folder-non-recursively --outline-file-format media_id_cutout_id 
test ./holdout/60407163-19df-5e6a-9b63-8b55315a8e19_743ba24b-6d77-5d4e-8363-3d62022d5ee7.png 

Test KF + softmap
nohup python -u ./dev/test_blind_quality.py mobilenet_v2_256_win_n_lyrs --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/new_db_7818_bin_class_penn_regularization/saved_state_mobilenet_v2_256_win_n_lyrs___1625844658 --database-root /hdd/annotator_uploads/70e91f29c8f3f1147475c252c4e369c6/blindImages --outline-pickle-path /hdd/annotator_uploads/70e91f29c8f3f1147475c252c4e369c6/blindImages/merged_outlines_json.pkl --result-dir /hdd/hanoch/data/cages_holdout/k_25_6/mobilenet_v2_256_win_n_lyrs___1625844658 --result-pickle-file inference_best_qual.pkl --gpu-id 2  --outline-file-format fname_n_cutout_id --image-fusion-voting-method avg_pool --confidence-threshold 0.85 --statistics-collect --process-folder-non-recursively

Test KF + softmap + old model
nohup python -u ./dev/test_blind_quality.py mobilenet_v2_256_win_n_lyrs --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/clip_grad_new_norm/fitjar/saved_state_mobilenet_v2_256_win_n_lyrs___1609764290 --database-root /hdd/annotator_uploads/70e91f29c8f3f1147475c252c4e369c6/blindImages --outline-pickle-path /hdd/annotator_uploads/70e91f29c8f3f1147475c252c4e369c6/blindImages/merged_outlines_json.pkl --result-dir /hdd/hanoch/data/cages_holdout/k_25_6/mobilenet_v2_256_win_n_lyrs___1609764290 --result-pickle-file inference_best_qual.pkl --gpu-id 2  --outline-file-format fname_n_cutout_id --image-fusion-voting-method avg_pool --confidence-threshold 0.85 --statistics-collect --process-folder-non-recursively


Cage298
nohup python -u ./dev/test_blind_quality.py mobilenet_v2_256_win_n_lyrs --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/new_db_7818_bin_class_penn_regularization/saved_state_mobilenet_v2_256_win_n_lyrs___1627835268 --database-root /hdd/hanoch/data/cages_holdout/annotated_images_eileen_batch2 --result-dir /hdd/hanoch/data/cages_holdout/annotated_images_eileen_batch2/new_db_1627835268 --outline-pickle-path /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/merged_outlines.pkl --result-pickle-file inference_best_qual.pkl --gpu-id=2 --confidence-threshold 0.85 --image-fusion-voting-th 0.85 --image-fusion-voting-method concensus_and_vote --statistics-collect --process-folder-non-recursively --outline-file-format misc >> ./annotated_images_eileen_batch2.py.log </dev/null 2>&1 & tail -f ./annotated_images_eileen_batch2.py.log

Fusion bin class over validation set
nohup python -u ./dev/test_blind_quality.py mobilenet_v2_256_win_n_lyrs --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/new_db_7818_bin_class_penn_regularization/saved_state_mobilenet_v2_256_win_n_lyrs___1625844658 --database-root /hdd/hanoch/data/database/blind_quality/quality_based_all_annotations_24_6/png/Eidesvik_val_set --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/stratified_train_val_test_test_set_penn_dependant_label_concensus_data_list_new_7818.csv --result-dir /hdd/hanoch/data/validation_newset_7818/new_db_1627835268 --outline-pickle-path /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/merged_outlines.pkl --result-pickle-file inference_best_qual.pkl --gpu-id=2 --confidence-threshold 0.85 --image-fusion-voting-th 0.85 --image-fusion-voting-method avg_pool --statistics-collect --process-folder-non-recursively --outline-file-format fname_n_cutout_id >> ./annotated_images_eileen_batch2.py.log </dev/null 2>&1 & tail -f ./annotated_images_eileen_batch2.py.log 
 
"""


