from __future__ import print_function, division

import os
from PIL import Image
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from modules.blind_quality.svm_utils.blindSvmPredictors import blindSvmPredictors as blindSvmPredictors
from collections import namedtuple
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import PIL.ImageColor as ImageColor
from PIL import ImageEnhance
import matplotlib.colors as mcolors
import tqdm
import glob
import time

from dev.dataset import class_labels, DatasetOnLine
from dev.inference import prepare_clargs_parser
from torchvision import transforms
from dev.models import initialize_model
from dev.evaluation import to_device
from dev.configuration import print_arguments
from dev.evaluation import evaluate_model_on_dataset
from dev.test_blind_quality import voting_over_tiles

Point = namedtuple('Point', ['x', 'y'])
#clsss_quality = {1: 'bad', 2: 'marginal', 3: 'good', 4: 'unknown-tested'}
from dev.test_blind_quality import clsss_quality_label_2_str as clsss_quality

def inference_flat(tiles, transformations, device, model, to_scale):

    batch_images_tens = []
    batch_images = tiles   # over single CPU is it efficient to convert to tensor and qpply transformations() over all ?

    for image in batch_images:
        if to_scale:
            image = transformations((image * 255.0).astype(np.uint8))
        else:
            image = transformations(image)

        batch_images_tens.append(image)

    tock4 = time.perf_counter()

    batch_images_tens = torch.stack(batch_images_tens, dim=0)
    # batch_images_tens = torch.cat([batch_images_tens, torch.unsqueeze(image, 0)], 0)
    if 1:
        batch_images_tens = batch_images_tens.pin_memory()
    # batch_labels = Variable(batch_labels).cuda(async=True)
    inputs = to_device(batch_images_tens, device)
    # predictions = model.forward(inputs)
    # predictions = torch.nn.functional.softmax(predictions, dim=1)

    tock5 = time.perf_counter()

    outputs = model(inputs)
    predictions = torch.nn.functional.softmax(outputs, dim=1)
    all_predictions = predictions.detach().cpu().numpy()
    return all_predictions, tock5, tock4




# def voting_over_tiles(good_cls_ratio, image_fusion_voting_th, confidence_threshold):
#     if good_cls_ratio >= image_fusion_voting_th and good_cls_ratio > 0.5:  # (good_cls_ratio>0.5) n_good_tiles should be > bad_tiles 'concensus_and_vote'
#         th_of_class = confidence_threshold
#         acc = good_cls_ratio
#         label = 3
#     else:  # bad quality
#         th_of_class = 1 - confidence_threshold  # Threshold for class good => for class bad it is 1-th
#         acc = 1 - good_cls_ratio
#         label = 1
#     return label, acc, th_of_class

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
                             softmax_score_good_cls=None):
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
    if actual_class_name == 'good':
        color1 = [ImageColor.getrgb('green') if x == 1 else ImageColor.getrgb('yellow') for x in good_det.astype('int')]
    else:
        color1 = [ImageColor.getrgb('red') if x == 1 else ImageColor.getrgb('orange') for x in good_det.astype('int')]


    thickness = 10
    det_str = ''
    det = dict()
    ovlap_tile_size_y = (pred.scaled_image_data.shape[0] - tile_size) / (pred.tile_map.shape[0] - 1)
    ovlap_tile_size_x = (pred.scaled_image_data.shape[1] - tile_size) / (pred.tile_map.shape[1] - 1)

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

        draw_bounding_box_on_image(pilimg, det["bottom"], det["left"], det["top"], det["right"],
                                   color=color1[idx],
                                   thickness=thickness,
                                   display_str_list=det_str, use_normalized_coordinates=False)
    if 1:
        enhancer = ImageEnhance.Brightness(pilimg)
        enhancer.enhance(brightness_fact).save(os.path.join(path_save, file_tag + '_cls_'+ actual_class_name + '_tiles_overlay.png'))
    else:
        pilimg.save(os.path.join(path_save, file_tag + '_tiles_over_blind.png'))
    if actual_class_name == 'good':
        color_base = mcolors.rgb_to_hsv(ImageColor.getrgb('green'))
        softmax_score_good_cls_target = softmax_score_good_cls
    else:
        color_base = mcolors.rgb_to_hsv(ImageColor.getrgb('red'))
        softmax_score_good_cls_target = 1 - softmax_score_good_cls
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
            os.path.join(path_save, file_tag + '_cls_' + actual_class_name + '_softmax_overlay.png'))


def fast_create_tiles_and_meta_data(image, outline, tile_size, args, cutout_id, label, media_id=None, online_fast=False):
    start_time = time.perf_counter()
    pred = blindSvmPredictors(cutout=image, outline=outline, tile_size=tile_size)
    tock = time.perf_counter()
    print("Finished tiles calc in %.4e" % (tock - start_time))

    if online_fast:
        return -1, -1, pred.masked_tile_array

    df_test = pd.DataFrame(columns=['file_name', 'cutout_uuid', 'class', 'label',
                                    'in_outline_tiles_id', 'train_or_test', 'full_file_name', 'media_uuid'])
    dim_tiles = pred.tile_map.shape
    tot_n_tiles = dim_tiles[0] * dim_tiles[1]
    in_outline_n_tiles = np.where(pred.tile_map)[0].shape
    print("In blind tiles ratio {}".format(in_outline_n_tiles[0] / tot_n_tiles))
    # the tiles within the blind
    if pred.cache['tiles'].shape[3] != tile_size and pred.cache['tiles'].shape[2] != tile_size:
        print('Error not the tile dim specified')

    tiles = pred.masked_tile_array
    # media_id = 0

    # Tiles are ordered row-wise by pred.tile_map telling which is relevent in outline by TRue/False
    # saving all the tiles
    assert (np.where(pred.tile_map.reshape((-1)))[0].shape[0] == tiles.shape[0])
    in_outline_tiles_id = np.where(pred.tile_map.reshape((-1)))[0]
    for tile_no in range(tiles.shape[0]):
        fname_save = str(media_id) + '_' + cutout_id + '_cls_' + clsss_quality[label] + '_tile_' + str(tile_no)

        # tiles are normalized by 255 to [0 1] tiles are saved scaled to [0 255].
        # CNN dataloder (DataSet) expecting [0 255] and scaling occurs when __getitem()__ applied to [0 1]
        pilimg = Image.fromarray((tiles[tile_no, :, :, :] * 255.0).astype(np.uint8))
        full_fmame = os.path.join(args.result_dir, fname_save + '.png')
        file_name = fname_save

        pilimg.save(full_fmame)

        tile_img = np.asarray(pilimg)
        # percent_40 = np.percentile(tile_img.ravel(), 40)
        # percent_60 = np.percentile(tile_img.ravel(), 60)
        # percent_80 = np.percentile(tile_img.ravel(), 80)

        df_test.loc[len(df_test)] = [file_name, cutout_id, clsss_quality[label], label,
                                     in_outline_tiles_id[tile_no], 'test', full_fmame, media_id]

    # df_test.to_csv(os.path.join(args.result_dir, cutout_id + '.csv'), columns=['file_name', 'cutout_uuid', 'class', 'label', 'percent_40', 'percent_60', 'percent_80',
    #              'in_outline_tiles_id', 'train_or_test', 'media_uuid'], index=False)

    return df_test, pred, pred.masked_tile_array

def plot_outline_with_comments(image, outline, path, cutout_id, acc, label):
    # plot_image_and_outline(image, [outline], 10)
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.plot([p[0] for p in outline], [p[1] for p in outline])
    ax.title.set_text('contour_blind_' + str(acc.__format__('.2f')) + '_' + str(cutout_id) + '_cls_' + str(label))
    plt.savefig(os.path.join(path, 'contour_blind_' + str(acc.__format__('.2f')) + '_' + str(cutout_id) + '_cls_' + str(label) + '.png'))
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

    parser.add_argument('run_config_name', type=str, metavar='RUN_CONFIGURATION',
                                help="name of an existing run configuration")

    parser.add_argument("--outline-pickle-path", type=str, required=True, default=None, metavar='PATH',
                                        help="outline-pickle-path")

    parser.add_argument("--result-pickle-file", type=str, required=False, default='inference-list.pkl', metavar='PATH',
                                        help="outline-pickle-file")

    parser.add_argument("--add-hue-offset", action='store_true',
                        help=' add hue offset for examined image to observe accuracy')

    parser.add_argument("--image-fusion-voting-th", type=float, required=False, default=0, metavar='FLOAT',
                        help='final image criterion')

    parser.add_argument("--statistics-collect", action='store_true',
                        help='statistics collect')

    parser.add_argument("--outline-file-format", type=str, required=False, default='media_id_cutout_id', choices=['media_id_cutout_id', 'misc', 'other'], metavar='STRING',
                                        help="")

    # parser.add_argument("--media-id-info-pickle-file", type=str, required=False, default='None', metavar='PATH',
    #                                     help="media id -pickle-file path structure : the value of the cutout dict key")


    args = parser.parse_args(args)

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    if args.image_fusion_voting_th<0 or args.image_fusion_voting_th>1:
        raise

    if args.dataset_split_csv is not None:
        df_annotation = pd.read_csv(args.dataset_split_csv)
        print('***********   Loading annotations !!!!!!!!  based analysis *****************')
    else:
        print('*********** No annotations *****************')


    model_name = args.run_config_name
    num_classes = 2
    feature_extract = True
    device = torch.device("cuda:" + str(args.gpu_id) + "" if torch.cuda.is_available() else "cpu")

    model, input_size = initialize_model(model_name, num_classes, feature_extract,
                                         pretrained_type={'source': 'imagenet', 'path': None})
    args.input_size = input_size
    args.loss_smooth = False

    # checkpoint = load_model(args.model_path, device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint)
    if torch.cuda.is_available():
        model = model.to(device)
    # summary of the model
    if 0: # it occupies more memory over the GPU probably due to the injected input
        from torchsummary import summary
        summary(model.to(device), (3, 256, 256))

    plot_outline = False
    args.gamma_aug_corr = 1.0 #0.5
    args.batch_size = 64*4
    # args.num_workers = 12
    tile_size = 256
    brightness_fact = 1.5
    args.brightness_fact = brightness_fact
    args.tile_size = tile_size
    test_data_filter = dict()
    if brightness_fact!=1.0:
        print("Stored image are bright-ned by".format(brightness_fact))


    if 0:
        test_data_filter = {'lessthan': 25, 'greater': 150}
    else:
        test_data_filter = {}

    unique_run_name = args.model_path.split('___')[-1]

    args.test_data_filter = test_data_filter
    df_result_over = pd.DataFrame(columns=['file_name', 'cutout_id', 'acc', 'actual_class', 'FN', 'FP', 'quality_svm', 'predict_all_img_label'])

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

    args.variance_iterate_over_file = 1
    # os.environ['DISPLAY'] = str('localhost:10.0')
    print_arguments(args)
    outline_file_format = args.outline_file_format

    print(outline_file_format)
    resulat_acc = dict()

    # filenames = [os.path.join(args.database_root, x) for x in os.listdir(args.database_root)
    #                  if x.endswith('png')]
    filenames = glob.glob(args.database_root + '/**/*.png', recursive=True)

    print("!!!! process {} cutouts *****".format(len(filenames)))

    filename_attr = ''
    hue_offset = 0.05
    if args.add_hue_offset:
        filename_attr = 'hue_' + str(hue_offset) + '_'
        print("Hue offset {} was added" .format(hue_offset))

    missing_label = False
    if args.dataset_split_csv is None:
        label = 4  # unknown
        # dummy label
        missing_label = True

    predict_all_img_label = -1
    all_img_acc = -1

    # online_fast = True
    sampler = None
    printouts = False

    tiles_calc_time = []
    dataset_class_prepare_time = []
    dataloader_time = []
    model_run_time = []

    transform_time = []
    cpu_gpu_time = []
    inference_time = []

    to_scale = False  # to_scale=False remain in int8
    pin_memory = True
    print("pin_memory {}".format(pin_memory))
    direct_infernce = True
# Let pytorch engine do the mem optimization
    torch.backends.cudnn.benchmark = True
    print("torch.backends.cudnn.benchmark{}".format(torch.backends.cudnn.benchmark))
    # torch.set_deterministic(True)

    if direct_infernce:
        transformations = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.ToTensor(),
             transforms.Normalize([0.13804892, 0.21836744, 0.20237076], [0.08498618, 0.07658653, 0.07137364])])

    model.eval()
    torch.no_grad()
# TODO: add edge removal from utillities.tiles_utill import tile_mapping
    for idx, file in enumerate(tqdm.tqdm(filenames)):
        for iterate_over_file in range(args.variance_iterate_over_file):
            print("Iteration: NO {}" .format(iterate_over_file))
            if file.split('.')[-1] == 'png':
                img = Image.open(file).convert("RGB")
                image = np.asarray(img)

                quality_svm = None
                if outline_file_format is 'media_id_cutout_id':
                    cutout_id = file.split('.')[-2].split('_')[-1]
                    outline = cutout_2_outline_n_meta.get(cutout_id, None)
                    if outline == None:
                        print("Outline for cutout_id {} was not found {}!!!".format(cutout_id, file))
                        continue

                    if isinstance(outline, dict):
                        outline = outline['outline']

                else:
                    cutout_id = file.split('/')[-1]
                    outline_dict = cutout_2_outline_n_meta.get(cutout_id, None)
                    if outline_dict == None:
                        print("Outline for cutout_id {} was not found!!!".format(cutout_id))
                        continue
                    quality_svm = clsss_quality[outline_dict['quality']]
                    outline = outline_dict['contour']
                    if isinstance(outline, dict):
                        outline = outline['outline']
                # If cutout isnot found then skip

                media_id = None
                if isinstance(cutout_2_outline_n_meta[cutout_id], dict):
                    media_id = cutout_2_outline_n_meta[cutout_id].get('media_id', None)

                if media_id is None:
                    media_id = cutout_id

                if args.add_hue_offset:
                    image_hsv = mcolors.rgb_to_hsv(image)
                    image_hsv[:, :, 0] = image_hsv[:, :, 0] + hue_offset
                    image = mcolors.hsv_to_rgb(image_hsv) # comment

                # if isinstance(outline, dict):
                #     outline = outline['outline']
                if plot_outline:
                    plot_image_and_outline(image, [outline], 10)
                    fig, ax = plt.subplots()
                    ax.imshow(image)
                    ax.plot([p[0] for p in outline], [p[1] for p in outline])
                    plt.savefig(os.path.join(args.result_dir, 'CContour_over_the_blind' + str(cutout_id) + '.png'))
                    # plt.savefig('CContour_over_the_blind' + str(11) + '.png')

                # Use annotations
                if args.dataset_split_csv is not None:
                    find_id = np.where(df_annotation['cutout_uuid'] == cutout_id) == np.empty([])
                    if (not find_id.any() and find_id != np.array([])):
                        cell = np.where(df_annotation['cutout_uuid'] == cutout_id)[0]
                        label = df_annotation['label'].iloc[cell[0]] # the first entry of the tiles names having the same class label as the given cutout is taken as the class
                    elif "marginal" in file.split('/'):
                        print('!!!! /*******  This is a patch remove')  #TODO remove
                        label = 1  # Marginal is missing from labels => bad class as well
                    else:  # no annotations in the file
                        print("cutout uuid was not found  in annotations !!!{}".format(cutout_id))
                        continue

                # tick = time.perf_counter()

                # df_test, blind_tile_obj, tiles = fast_create_tiles_and_meta_data(image, outline, tile_size, args, cutout_id, label, media_id, online_fast=online_fast)
                tock0 = time.perf_counter()
                pred = blindSvmPredictors(cutout=image, outline=outline, tile_size=tile_size, to_scale=to_scale)

                tiles = pred.masked_tile_array
                # batch size need to be no greater than amount of tiles
                tock1 = time.perf_counter()
                tiles_calc_time += [tock1 - tock0]

                if printouts:
                    print("Finished tiles calc in %.4e" % (tock1 - tock0))

                if direct_infernce: # the final and fastest
                    tock3 = time.perf_counter()
                    all_predictions, tock5, tock4 = inference_flat(tiles, transformations, device, model, to_scale)
                    tock6 = time.perf_counter()

                    all_targets = np.array(label).repeat(tiles.shape[0])

                    model_run_time += [tock6 - tock3]
                    transform_time += [tock5 - tock4]
                    cpu_gpu_time += [tock4 - tock3]
                    inference_time += [tock6 - tock5]
                    print("Only model inference  %.4e ntiles  %d" % (inference_time[-1], tiles.shape[0]))
                    print("MAx allocated mem[MB] {} allocated mem[MB] {}".format(torch.cuda.max_memory_allocated(device)/1024**2, torch.cuda.memory_allocated(device)/1024**2))
                    print("MAx reserved mem[MB] {} reserved mem[MB] {}".format(torch.cuda.max_memory_reserved(device)/1024**2, torch.cuda.memory_reserved(device)/1024**2))
                    # torch.cuda.memory_snapshot()

                    if 0:
                        print("torch.cuda.memory_allocated: %fGB" % (torch.cuda.memory_allocated() / 1024 / 1024 / 1024))
                        print(torch.cuda.get_device_properties('cuda:0'))
                        max_allocated = torch.cuda.max_memory_allocated()
                        print("Max memory allocated to the GPU {} [MBytes]".format(max_allocated/(1024**2)))
                        print("Memory reserved to the GPU {} [MBytes]".format(torch.cuda.memory_reserved()/(1024**2)))
                        print("Max memory reserved to the GPU {} [MBytes]".format(torch.cuda.max_memory_reserved()/(1024**2)))

# torch.cuda.max_memory_reserved() Returns the maximum GPU memory managed by the caching allocator in bytes for a given device


                    if printouts:
                        print("********** Finished model inference ******** %.4e" % (inference_time[0]))
                        print("Finished tensor process %.4e" % (transform_time[0]))
                        print("Finished transform process %.4e" % (cpu_gpu_time[0]))
                        print("N tiles {}".format(tiles.shape[0]))


                else:
                    tock2 = time.perf_counter()
                    test_dataset = DatasetOnLine(data=tiles, target=np.array(label).repeat(tiles.shape[0]),
                                                 center_crop_size=args.tile_size, to_scale=to_scale)

                    dataset_class_prepare_time += [tock2 - tock1]

                    if printouts:
                        print("Finished Dataset prepare in %.4e" % (tock2 - tock1))

                    tock3 = time.perf_counter()
                    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=tiles.shape[0],
                                                                  shuffle=False,
                                                                  pin_memory=pin_memory, num_workers=args.num_workers,
                                                                  sampler=sampler)

                    tock4 = time.perf_counter()
                    dataloader_time += [tock4 - tock3]

                    if printouts:
                        print("Finished Dataloader prepare in %.4e" % (tock4 - tock3))
                        print("Finished prepare data in %.4e" % (tock4 - tock0))


                    if tiles.shape[0] == 0:
                        # all are bad quality, since they were probably filtered by the  test_filter_percentile40
                        if clsss_quality[label] == 'bad':
                            acc = 1.0
                        else:
                            acc = 0.0
                            # plot_outline_with_comments(image, outline, args.result_dir, cutout_id, acc, clsss_quality[label])

                        print('All tiles were filtered out by the Dataset filter = > bad quality image')
                        continue

                    tock5 = time.perf_counter()

                    all_targets, all_predictions, loss, all_features, tiles_id = evaluate_model_on_dataset(model,
                                                        test_dataloader, device, loss_fn=None, max_number_of_batches=None,
                                                        do_softmax=True, missing_label=missing_label)

                    tock6 = time.perf_counter()
                    model_run_time += [tock6 - tock5]

                    if printouts:
                        print("Finished model inference %.4e" % (tock6 - tock5))
                        print("N tiles {}".format(tiles.shape[0]))

    #TODO check bit extact to final decision of test_blind_quality.py
                #testset no labels hence decision is open loop
                if missing_label:

                    dets = all_predictions[:, class_labels['good']] > args.confidence_threshold
                    good_cls_ratio = np.sum(dets.astype('int')) / dets.shape[0]
                    #pseudo label for the entire image based on decision

                    predict_all_img_label, acc_all_img, th_of_class_img = voting_over_tiles(good_cls_ratio, args.image_fusion_voting_th,
                                                                                                args.confidence_threshold)

                    if predict_all_img_label == 3:
                        acc = good_cls_ratio
                        th_of_class = args.confidence_threshold

                    else:
                        acc = 1 - good_cls_ratio
                        th_of_class = 1 - args.confidence_threshold

                    good_det = dets.astype('int')
                    fn_tile = -1  # dummy value
                    fp_tile = -1

                else:
                    # result analysis

                    if hasattr(args, 'confidence_threshold'):
                        dets = all_predictions[:, class_labels['good']] > args.confidence_threshold
                        correct_det = all_targets == dets.astype('int') # good det no matter what the class
                    else:
                        dets = all_predictions[:, class_labels['good']] >= 0.5
                        correct_det = all_targets == all_predictions.argmax(axis=1)

                    good_cls_ratio = np.sum(dets.astype('int'))/all_targets.shape[0]
                    acc = np.sum(correct_det.astype('int'))/ all_targets.shape[0]

                    if clsss_quality[label] == 'good':
                        # acc = acc_good_cls
                        fn_tile = 1 - acc # if all blind is in good q then the tiles below th are considered FN
                        fp_tile = None
                        th_of_class = args.confidence_threshold
                    else:
                        # acc = 1 - acc_good_cls # since acc related
                        fn_tile = None
                        fp_tile = 1 - acc
                        th_of_class = 1 - args.confidence_threshold # Threshold for class good => for class bad it is 1-th

                    print("cutout_id {}: acc {}".format(cutout_id, acc))
                    if args.image_fusion_voting_th:
                        predict_all_img_label, acc_all_img, th_of_class_img = voting_over_tiles(good_cls_ratio=good_cls_ratio,
                                                                                      image_fusion_voting_th=args.image_fusion_voting_th,
                                                                                      confidence_threshold=args.confidence_threshold)
                        all_img_acc = [1 if (predict_all_img_label==label) is True else 0][0]

                # tock = time.perf_counter()
                # print("Finished 3 in %.2fs" % (tock - tick))
                # print("Finished 3 in %.4e" % (time.time() - start_time))

        # nested_record = {'acc': acc, 'actual_class': clsss_quality[label], 'FN': fn_tile, 'FP': fp_tile}
        tiles_calc_time_mu = np.mean(np.array(tiles_calc_time))
        tiles_calc_time_std = np.std(np.array(tiles_calc_time))
        dataset_class_prepare_time_mu = np.mean(np.array(dataset_class_prepare_time))
        dataset_class_prepare_time_std = np.std(np.array(dataset_class_prepare_time))
        dataloader_time_mu = np.mean(np.array(dataloader_time))
        dataloader_time_std = np.std(np.array(dataloader_time))
        model_run_time_mu = np.mean(np.array(model_run_time))
        model_run_time_std = np.std(np.array(model_run_time))
        transform_time_mu = np.mean(np.array(transform_time))
        cpu_gpu_time_mu = np.mean(np.array(cpu_gpu_time))
        inference_time_mu = np.mean(np.array(inference_time))


        nested_record = {'n_tiles': tiles.shape[0], 'tiles_calc_time_mu': tiles_calc_time_mu, 'tiles_calc_time_std': tiles_calc_time_std,
                         'dataset_class_prepare_time_mu' : dataset_class_prepare_time_mu,
                         'dataset_class_prepare_time_std': dataset_class_prepare_time_std,
                         'dataloader_time_mu': dataloader_time_mu, 'dataloader_time_std': dataloader_time_std,
                         'model_run_time_mu': model_run_time_mu, 'model_run_time_std': model_run_time_std,
                         'transform_time_mu':transform_time_mu, 'cpu_gpu_time_mu':cpu_gpu_time_mu,
                         'inference_time_mu':inference_time_mu}

        resulat_acc.update({cutout_id: nested_record})
        df_result_over.loc[len(df_result_over)] = [file, cutout_id, acc, clsss_quality[label], fn_tile, fp_tile, quality_svm, predict_all_img_label]

        # plot_name = cutout_id + '_tiles_acc_' + str(acc.__format__('.2f')) + '_img_all_acc_' + str(all_img_acc.__format__('.1f')) + '_th_' + str(th_of_class.__format__('.2f'))
        # plot_cls_tiles_box_over_blind(tile_size, blind_tile_obj, tiles_id, good_det, img,
        #                          args.result_dir, actual_class_name=clsss_quality[label], file_tag=plot_name,
        #                          brightness_fact=brightness_fact, softmax_map=True,
        #                          softmax_score_good_cls=all_predictions[:, 1])


        # if acc<0.75:
        #     plot_outline_with_comments(image, outline, args.result_dir, cutout_id, acc, label)

        if not direct_infernce:
        # clear dataloader and dataframe otherwise all the data are aggregated
            del test_dataloader
        # df_test.drop(df_test.index, inplace=True)

        if (idx % 10 == 0):
            with open(os.path.join(args.result_dir, unique_run_name + '_cycle_count_' + args.result_pickle_file), 'wb') as f:
                pickle.dump(resulat_acc, f)
            df_result_over.to_csv(os.path.join(args.result_dir,
                                'inference_cycle_count_partial' + unique_run_name + '_th_' + str(args.confidence_threshold) + '.csv'),
                                  index=False)
            if args.statistics_collect:
                fusion_tiles.save_results(os.path.join(args.result_dir, unique_run_name + '_th_' + str(args.confidence_threshold) + '_statistics_' + args.result_pickle_file))
                # stat_res = dict(tp_mat_acm=tp_mat_acm, tpr_cnt=tpr_cnt,
                #                 tn_mat_acm=tn_mat_acm, tnr_cnt=tnr_cnt, fn_mat_acm=fn_mat_acm,
                #                 fp_mat_acm=fp_mat_acm, thr=thr, voting_percentile=voting_percentile)
                # with open(os.path.join(args.result_dir, unique_run_name + '_statistics_' + args.result_pickle_file), 'wb') as f:
                #     pickle.dump(stat_res, f)

    with open(os.path.join(args.result_dir, unique_run_name + '_cycle_count_' + args.result_pickle_file), 'wb') as f:
        pickle.dump(resulat_acc, f)

    if args.statistics_collect:
        with open(os.path.join(args.result_dir, unique_run_name + '_stat_collect_' + args.result_pickle_file), 'wb') as f:
            pickle.dump(resulat_stat_acc, f)

    df_result_over.to_csv(os.path.join(args.result_dir, 'inference_cycle_count' + filename_attr + unique_run_name + '_th_' + str(args.confidence_threshold) + '.csv'), index=False)

    resulat_acc.update(vars(args))
    df_result = pd.DataFrame.from_dict(list(resulat_acc.items()))
    df_result = df_result.transpose()
    df_result.to_csv(os.path.join(args.result_dir, 'inference-withsettings' + filename_attr + unique_run_name + '_th_' + str(args.confidence_threshold) + '.csv'), index=False)

    if args.statistics_collect:
        fusion_tiles.save_results(os.path.join(args.result_dir, unique_run_name + '_th_' + str(
                                    args.confidence_threshold) + '_statistics_' + args.result_pickle_file))

    arguments_str = '\n'.join(["{}: {}".format(key, resulat_acc[key]) for key in sorted(resulat_acc)])
    print(arguments_str + '\n')
    # for ROC need to aggregate info
    # roc_plot(all_targets, all_predictions[:, 1], 1, args.result_dir, unique_id=unique_run_name+str(args.confidence_threshold))

if __name__ == "__main__":
    main()

"""
************  This file accepts full cutout images ************* 
--num-workers 1 : to mimic single CPU at the target
nohup python -u inference_production.py --model-path /hdd/hanoch/runmodels/img_quality/results/saved_state_mobilenet_v2_256_win_n_lyrs___1593440243 --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/train_test_val_quality_tile_filtered_eileen_good_bad_val.csv --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/holdout --result-dir /hdd/hanoch/runmodels/img_quality/results/inference_production/temp --outline-pickle-path /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/holdout/quality_holdout_with_marginal_outlines_merged.pkl --result-pickle-file production_holdout_results.pkl --gpu-id=0 --confidence-threshold 0.1 --image-fusion-voting-th 0.25 --num-workers 1 & tail -f nohup.out
                                        --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/clip_grad_new_norm/sgd_filt_list/saved_state_mobilenet_v2_256_win_n_lyrs___1606226252 --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/train_test_val_quality_tile_filtered_eileen_good_bad_val.csv --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/holdout --result-dir /hdd/hanoch/runmodels/img_quality/results/inference_production/temp --outline-pickle-path /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/holdout/quality_holdout_with_marginal_outlines_merged.pkl --result-pickle-file production_holdout_results.pkl --gpu-id=0 --confidence-threshold 0.1 --image-fusion-voting-th 0.25 --num-workers 1 & tail -f nohup.out
mobilenet_v2_256_win_n_lyrs --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/clip_grad_new_norm/fitjar/saved_state_mobilenet_v2_256_win_n_lyrs___1609764290 --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/train_test_val_quality_tile_filtered_eileen_good_bad_val.csv --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/holdout --result-dir /hdd/hanoch/runmodels/img_quality/results/inference_production/temp --outline-pickle-path /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/holdout/quality_holdout_with_marginal_outlines_merged.pkl --result-pickle-file production_holdout_results.pkl --gpu-id=0 --confidence-threshold 0.1 --image-fusion-voting-th 0.25 --num-workers 1
"""
#TODO add JSON model metadata reading and override the data normalization
