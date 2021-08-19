from modules.blind_quality.svm_utils import blindSvm #HK TODO after refactoring all repo use SVM eval from blind_quality_svm repo or go to mess with bringing it here as a source or a module
from modules.blind_quality.svm_utils.blindSvmPredictors import blindSvmPredictors as blindSvmPredictors
#from finders.common.mask_utils import Point
from PIL import Image
from io import BytesIO
import tqdm
import numpy as np
import os
import subprocess
import pickle
import pandas as pd
import tensorflow as tf
from itertools import groupby

#from skimage import measure
# creates error : the ollowing import
from dev.test_blind_quality import plot_cls_tiles_box_over_blind, clsss_quality_label_2_str, Point
# sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
# from Finders.finders_tools.finders.tools import tfrecord_utils, generalized_iou, url_utils
# the following code was borrowed from test_blind_quality.py please fix python dependancy
clsss_quality_label_2_str = {1: 'bad', 2: 'marginal', 3: 'good', 4: 'unknown-tested'}
from modules import tile_mapping

def append_url(prefix, suffix):
    return prefix._replace(path=os.path.join(prefix.path, suffix))

def main():
    # load data and train classifier once
    path_to_init_file = '/home/hanoch/GIT/blind_quality_svm/bin/trn/svm_benchmark/FACSIMILE_PRODUCTION_V0_1_5/fixed_C_200_facsimile___prod_init_hk.json'
    path_data = '/hdd/hanoch/data/objects-data-bbox-20191106-simple-sharded-part/cutout_data'
    # path_tile_data = '/hdd/hanoch/data/objects-data-bbox-20191106-simple-sharded-part/tile_data_64'
    path_tile_data = '/hdd/hanoch/data/objects-data-bbox-20191106-simple-sharded-part/tile_data_debug'
    # pickle_file_name = 'quality_training_outlines_temp.pkl'

    remove_tiles_on_edges = True
    if remove_tiles_on_edges:
        print("Tiles on edges are removed !!!! ******************************************")

    train_test_flag = 'holdout'
    tile_size = 256
    print("Tile size {}".format(tile_size))

    clsss_quality = {1 :'bad', 2:'marginal', 3: 'good'}
    label_to_be_excluded = 2# in the future if label 2 (marginal is of interest change to dummy labels list [1,3])
    svm = blindSvm.blindSvm(blind_svm_params=path_to_init_file)    #Following the default path as an example, this line looks for all .json files in the trn submodule and picks the first one

    if train_test_flag is 'train':
        print('process the train set')
        cutout_ids = svm.blind_svm_params.all_parameters['blind_keys_training']
        labels = svm.blind_svm_params.all_parameters['judgments_training']
        path_tiles = os.path.join(path_tile_data, 'train')
        train_or_test = 'train'
        if 0:
            pickle_file_name = 'quality_training_outlines.pkl'
        else:
            # pickle_file_name = 'train_cutout_from_blindfinder.pkl'
            pickle_file_name = 'quality_training_outlines_merged.pkl'
            print('2nd pickle from blindfinder')

        pickle_path = os.path.join(path_data, 'train')

    elif train_test_flag is 'holdout':
        print('process the holdout set !!!!')
        cutout_ids = svm.blind_svm_params.all_parameters['blind_keys_holdout']
        labels = svm.blind_svm_params.all_parameters['judgments_holdout']
        if 0:
            path_tiles = os.path.join(path_tile_data, 'test')
            pickle_file_name = 'quality_holdout_outlines.pkl'
            pickle_file_name = 'quality_holdout_outlines_merged.pkl'
            pickle_path = os.path.join(path_data, 'holdout')
        else: #for marginal class under holdout
            label_to_be_excluded = [0, 1] # take out only the marginals
            path_tiles = os.path.join(path_tile_data, 'test')
            pickle_file_name = 'holdout_marginal_cutout_from_blindfinder.pkl'
            # pickle_path = os.path.join(path_data, 'holdout')
            pickle_path = os.path.join(path_data, 'holdout/marginal')
        train_or_test = 'test'

    # aggregate labels and file names
    df_train_test = pd.DataFrame(columns=['file_name', 'cutout_uuid', 'media_uuid', 'class', 'label', 'percent_40', 'percent_60', 'percent_80', 'train_or_test'])
    # cutout extraction

    try:
        with open(os.path.join(pickle_path, pickle_file_name), 'rb') as f:
            cutout_2_outline_n_meta = pickle.load(f)
            print('direct')
    except:
        import sys
        sys.path.append('/home/hanoch/GIT/Finders')
        sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        from Finders.finders.blind import default as blindfinder
        with open(os.path.join(pickle_path, pickle_file_name), 'rb') as f:
            cutout_2_outline_n_meta = pickle.load(f)
            print('via_Finders')

    if 0:
        print('False labels and cutout')
        cutout_ids = list(cutout_2_outline_n_meta)
        labels = svm.blind_svm_params.all_parameters['judgments_training'][:len(cutout_ids)]

    if not os.path.exists(path_tiles):
        os.makedirs(path_tiles)

    for label_ in range(3):
        if not os.path.exists(os.path.join(path_tiles, str(clsss_quality[label_+1]))):
            os.makedirs(os.path.join(path_tiles, str(clsss_quality[label_+1])))

    # print(svm.blind_svm_params.model_handlers.keys())
    for train_cut_uuid, label in tqdm.tqdm(zip(cutout_ids, labels)):
        # skip marginal label only good or bad
        if label in label_to_be_excluded:
            continue
        # file_full_path = os.system('find -iname ' + '"*' + train_cut_uuid + '*"')
        file_full_path = subprocess.getoutput('find ' + path_data + ' -iname ' + '"*' + train_cut_uuid + '.png"')
        if '\n' in file_full_path:
            file_full_path = file_full_path.split('\n')[0]
        if file_full_path is '':
            print("Cutout file hasn;t found {}".format(train_cut_uuid))
            continue
        file_name = os.path.basename(file_full_path)
        print(file_name)
        media_id = file_name.split('_')[0]
        cutout_id = file_name.split('_')[1].split('.')[0]  # should be equal to train_cut_uuid

        img = Image.open(file_full_path).convert("RGB")
        # image = img.load()
        image = np.asarray(img)
        # make sure image array is [0 255 since pred scales it dwon assuming it is ]
        # no info in the pickle about the outline
        if cutout_2_outline_n_meta.get(train_cut_uuid, None) is None:
            print("No blind Q key cutout found in pickles from tfrecords i.e. no outline {}".format(train_cut_uuid))
            continue
        outline = cutout_2_outline_n_meta[train_cut_uuid]['outline']

# crop-polygon is created implicitly and filters out the tiles accordingly

        pred = blindSvmPredictors.blindSvmPredictors(cutout=image, outline=outline, tile_size=tile_size)

        dim_all_tiles = pred.tile_map.shape
        tot_n_tiles = dim_all_tiles[0] * dim_all_tiles[1]
        in_outline_n_tiles = np.where(pred.tile_map)[0].shape
        print("In blind tiles ratio {}".format(in_outline_n_tiles[0]/tot_n_tiles))
        # the tiles within the blind
        if pred.cache['tiles'].shape[3] != tile_size and pred.cache['tiles'].shape[2] != tile_size:
            print('Error not the tile dim specified')

        if remove_tiles_on_edges:
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
            temp = np.ones_like(pred.tile_map).astype('bool')
            for tmap in tile_map:
                temp = temp & tmap
            pred.cache['tile_map'] = temp

            correct_det = label
            all_tiles_id = np.where(pred.tile_map.ravel())[0] # translate the active tiles in the outline to row-wise tile id
            correct_det = label*np.ones_like(all_tiles_id)
            brightness_fact = 1.0
            remove_tiles_on_edges_ratio = all_tiles_id.shape[0] / in_outline_n_tiles[0]
            print("Removed tiles on edges ratio {}".format(remove_tiles_on_edges_ratio))
            # fig, ax = plt.subplots()
            # ax.imshow(image)
            # ax.plot([p[0] for p in outline], [p[1] for p in outline])
            # ax.title.set_text(
            #     'contour_blind_'  + str(cutout_id) + '_cls_' + str(label))
            # plt.savefig(os.path.join(path_tile_data, 'contour_blind_' + '_' + str(
            #     cutout_id) + '_cls_' + str(label) + '.png'))


            plot_cls_tiles_box_over_blind(tile_size, pred, all_tiles_id, correct_det, img,
                                     path_tile_data, actual_class_name=clsss_quality_label_2_str[label], file_tag=cutout_id,
                                     brightness_fact=brightness_fact, softmax_map=True,
                                     softmax_score_good_cls=np.ones_like(all_tiles_id))

        tiles = pred.masked_tile_array
        if tiles.size == 0:
            print("Warning contour gave no Tiles!!!!")
            return -1
        # saving all the tiles
        for tile_no in range(tiles.shape[0]):
            fname_save = media_id + '_' + train_cut_uuid + '_cls_' + clsss_quality[label] + '_tile_' + str(tile_no)

            pilimg = Image.fromarray((tiles[tile_no, :, :, :] * 255).astype(np.uint8))
            pilimg.save(os.path.join(path_tiles, str(clsss_quality[label]), fname_save + '.png'))

            tile_img = np.asarray(pilimg)
            percent_40 = np.percentile(tile_img.ravel(), 40)
            percent_60 = np.percentile(tile_img.ravel(), 60)
            percent_80 = np.percentile(tile_img.ravel(), 80)

            df_train_test.loc[len(df_train_test)] = [fname_save, train_cut_uuid, media_id, clsss_quality[label], label, percent_40, percent_60, percent_80, train_or_test]
        #add records to pandas
        # d = '8d79bf88-5702-55e3-bcbf-2da396a27631'
        # [idx for idx, ind in enumerate(svm.blind_svm_params.all_parameters['blind_keys_training']) if d in ind]
        # svm.blind_svm_params.all_parameters['classes']['3']
    df_train_test.to_csv(os.path.join(path_tile_data, train_test_flag + '_' + pickle_file_name.split('.')[0] + '_tile.csv'), index=False)

    path_out = r'/hdd/hanoch/data/quality_tiles'
    # path_input = '/home/user/blindVision/objects-data-bbox-20191106-simple-sharded'
    SAVE_CUTOUT = True
    # file = 'objects_train_00009.tfrecord'

    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    return
"""

"""
OUTLINE_PTS = 1024
#
MAX_EXAMPLES_PER_SHARD = 256


def int64_feature(values):
    """Returns a TF-Feature of int64s.
    Args:
      values: A scalar or list of values.
    Returns:
      a TF-Feature.
    """
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
    """Returns a TF-Feature of bytes.
    Args:
      values: A string.
    Returns:
      a TF-Feature.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def floats_list_feature(value):
    return tf.train.FeatureList(
        feature=[tf.train.Feature(float_list=tf.train.FloatList(value=v))
                 for v in value])


def float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def image_to_tfexample(image_uuid, blind_uuid, crop_xmin, crop_xmax, crop_ymin,
                       crop_ymax, object_xmin, object_xmax, object_ymin,
                       object_ymax, image_data, image_format, height, width,
                       class_id, class_text):
    return tf.train.Example(features=tf.train.Features(feature={
        'image/uuid': bytes_feature(image_uuid),
        'image/source_id': bytes_feature(image_uuid),
        'image/crop/uuid': bytes_feature(blind_uuid),
        'image/crop/pixel_bbox/xmin': float_feature(crop_xmin),
        'image/crop/pixel_bbox/xmax': float_feature(crop_xmax),
        'image/crop/pixel_bbox/ymin': float_feature(crop_ymin),
        'image/crop/pixel_bbox/ymax': float_feature(crop_ymax),
        'image/object/pixel_bbox/xmin': float_feature(object_xmin),
        'image/object/pixel_bbox/xmax': float_feature(object_xmax),
        'image/object/pixel_bbox/ymin': float_feature(object_ymin),
        'image/object/pixel_bbox/ymax': float_feature(object_ymax),
        'image/encoded': bytes_feature(image_data),
        'image/format': bytes_feature(image_format),
        'image/class/label': int64_feature(class_id),
        'image/class/text': bytes_feature(class_text),
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/channels': int64_feature(3),
        'image/colorspace': bytes_feature(b'rgb'),
    }))


def image_and_bb_to_tfexample(image_uuid, crop, outline, image_data, bboxes,
                              channels=3, image_format=b'png',
                              colorspace=b'RGB'):
    xmin = []
    xmax = []
    ymin = []
    ymax = []
    normalized_xmin = []
    normalized_xmax = []
    normalized_ymin = []
    normalized_ymax = []
    label = []
    class_name = []
    image_sha = hashlib.sha256(image_data).digest()
    for box in bboxes:
        xmin.append(box['left'])
        xmax.append(box['left'] + box['width'])
        ymin.append(box['top'])
        ymax.append(box['top'] + box['height'])
        normalized_xmin.append(max(0, box['left'] / crop['width']))
        normalized_xmax.append(
            min(1, (box['left'] + box['width']) / crop['width']))
        normalized_ymin.append(max(0, box['top'] / crop['height']))
        normalized_ymax.append(
            min(1, (box['top'] + box['height']) / crop['height']))
        label.append(box['label'])
        class_name.append(box['class'])
    ox = [float(x) for x in outline[0]]
    oy = [float(y) for y in outline[1]]
    return tf.train.Example(features=tf.train.Features(feature={
        'image/uuid': bytes_feature(image_uuid),
        'image/source_id': bytes_feature(image_uuid),
        'image/key/sha256': bytes_feature(image_sha),
        'image/encoded': bytes_feature(image_data),
        'image/format': bytes_feature(image_format),
        'image/height': int64_feature(crop['height']),
        'image/width': int64_feature(crop['width']),
        'image/colorspace': bytes_feature(colorspace),
        'image/channels': int64_feature(channels),
        'image/crop/uuid': bytes_feature(crop['uuid']),
        'image/crop/pixel_bbox/xmin': float_feature(crop['xmin']),
        'image/crop/pixel_bbox/xmax': float_feature(crop['xmax']),
        'image/crop/pixel_bbox/ymin': float_feature(crop['ymin']),
        'image/crop/pixel_bbox/ymax': float_feature(crop['ymax']),
        'image/outline/x': float_list_feature(ox),
        'image/outline/y': float_list_feature(oy),
        'image/object/pixel_bbox/xmin': float_feature(xmin),
        'image/object/pixel_bbox/xmax': float_feature(xmax),
        'image/object/pixel_bbox/ymin': float_feature(ymin),
        'image/object/pixel_bbox/ymax': float_feature(ymax),
        'image/object/bbox/xmin': float_feature(normalized_xmin),
        'image/object/bbox/xmax': float_feature(normalized_xmax),
        'image/object/bbox/ymin': float_feature(normalized_ymin),
        'image/object/bbox/ymax': float_feature(normalized_ymax),
        'image/object/bbox/label': int64_feature(label),
        'image/object/class/label': int64_feature(label),
        'image/object/class/text': bytes_list_feature(class_name),
    }))


def read_count_file(dataset_path, subset):
    pattern = "TOTAL_(\d+)_SHARDS_%s" % (subset)
    for fname in tf.gfile.ListDirectory(dataset_path):
        m = re.match(pattern, fname)
        if m is not None:
            shard_count = int(m.groups()[0])
            with tf.gfile.Open(os.path.join(dataset_path, fname)) as f:
                example_count = int(f.read())
            return shard_count, example_count
    return None, None


class EnumeratedGenerator(object):
    def __init__(self, length, generator):
        self.length = length
        self.generator = generator

    def __len__(self):
        return self.length

    def __iter__(self):
        return self.generator


def iterate_tfrecord_subset(dataset_path, subset, mode='objects', skip=None):
    shard_count, example_count = read_count_file(dataset_path, subset)

    def record_generator():
        shard=0
        full_input_path = os.path.join(
            dataset_path,
            "%s_%s_%05d.tfrecord" % (mode, subset, shard))
        while os.path.isfile(full_input_path):
            yield from iterate_tfrecord(full_input_path, kind=mode, skip=skip)
            shard=shard+1
            full_input_path = os.path.join(
                dataset_path,
                "%s_%s_%05d.tfrecord" % (mode, subset, shard))
    return EnumeratedGenerator(example_count, record_generator())


def iterate_tfrecord(full_input_path, kind='objects', skip=None):
    record_iterator = iter(tf.python_io.tf_record_iterator(path=full_input_path))
    example = tf.train.Example()
    if skip is not None:
        skip_first_counter = 0
        skip_gap_counter = skip[1]
    for string_record in record_iterator:
        if skip is not None:
            if skip_first_counter < skip[0]:
                skip_first_counter += 1
                continue
            if skip_gap_counter < skip[1]:
                skip_gap_counter += 1
                continue
            else:
                skip_gap_counter = 0

        example.ParseFromString(string_record)
        image_uuid = (example.features.feature['image/uuid'].bytes_list
                                                            .value[0])
        image_uuid = image_uuid.decode('ascii')
        image_data = (example.features.feature['image/encoded'].bytes_list
                                                               .value[0])
        image_format = (example.features.feature['image/format'].bytes_list
                                                                .value[0])
        image_colorspace = (
            example.features.feature['image/colorspace'].bytes_list.value[0])
        height = int(example.features.feature['image/height']
                                     .int64_list
                                     .value[0])
        width = int(example.features.feature['image/width']
                                    .int64_list
                                    .value[0])
        channels = int(example.features.feature['image/channels']
                                       .int64_list
                                       .value[0])

        if kind == 'objects':
            crop_uuid = (example.features.feature['image/crop/uuid'].bytes_list
                                                                    .value[0])
            crop_uuid = crop_uuid.decode('utf-8')
            crop_xmin = example.features.feature[
                'image/crop/pixel_bbox/xmin'].float_list.value[0]
            crop_xmax = example.features.feature[
                'image/crop/pixel_bbox/xmax'].float_list.value[0]
            crop_ymin = example.features.feature[
                'image/crop/pixel_bbox/ymin'].float_list.value[0]
            crop_ymax = example.features.feature[
                'image/crop/pixel_bbox/ymax'].float_list.value[0]
            crop = dict(uuid=crop_uuid,
                        left=crop_xmin,
                        top=crop_ymin,
                        right=crop_xmax,
                        bottom=crop_ymax)
            outline_x = example.features.feature[
                'image/outline/x'].float_list.value
            outline_y = example.features.feature[
                'image/outline/y'].float_list.value
            outline = [Point(x, y) for x, y in zip(outline_x, outline_y)
                       if x >= 0 and y >= 0]
        p_xmins = example.features.feature[
            'image/object/pixel_bbox/xmin'].float_list.value
        p_xmaxs = example.features.feature[
            'image/object/pixel_bbox/xmax'].float_list.value
        p_ymins = example.features.feature[
            'image/object/pixel_bbox/ymin'].float_list.value
        p_ymaxs = example.features.feature[
            'image/object/pixel_bbox/ymax'].float_list.value
        xmins = example.features.feature[
            'image/object/bbox/xmin'].float_list.value
        xmaxs = example.features.feature[
            'image/object/bbox/xmax'].float_list.value
        ymins = example.features.feature[
            'image/object/bbox/ymin'].float_list.value
        ymaxs = example.features.feature[
            'image/object/bbox/ymax'].float_list.value
        labels = example.features.feature[
            'image/object/bbox/label'].int64_list.value
        text = example.features.feature[
            'image/object/class/text'].bytes_list.value
        if kind == 'blind':
            mask = example.features.feature[
                'image/object/mask'].bytes_list.value
            outline_x = example.features.feature[
                'image/object/outline/x'].float_list.value
            outlines_x = map(lambda kg: map(lambda g: g[1], kg[1]),
                             groupby(enumerate(outline_x),
                                     key=lambda t: t[0] // OUTLINE_PTS))
            outline_y = example.features.feature[
                'image/object/outline/y'].float_list.value
            outlines_y = map(lambda kg: map(lambda g: g[1], kg[1]),
                             groupby(enumerate(outline_y),
                                     key=lambda t: t[0] // OUTLINE_PTS))
            outlines = [[
                Point(x, y) for x, y in zip(ox, oy) if x >= 0 and y >= 0]
                for ox, oy in zip(outlines_x, outlines_y)]
            outline = []
        objects = [
            dict(top=ymin, left=xmin, width=xmax - xmin,
                 height=ymax - ymin, label=label, text=text)
            for xmin, xmax, ymin, ymax, label, text in zip(
                p_xmins, p_xmaxs, p_ymins, p_ymaxs, labels, text)]
        if kind == 'blind':
            for o, m in zip(objects, mask):
                o['mask'] = m
            for o, oo in zip(objects, outlines):
                o['outline'] = oo
        normalized_objects = [
            dict(top=ymin, left=xmin, width=xmax - xmin,
                 height=ymax - ymin, label=label, text=text)
            for xmin, xmax, ymin, ymax, label, text in zip(
                xmins, xmaxs, ymins, ymaxs, labels, text)]
        if kind == 'blind':
            for o, m in zip(normalized_objects, mask):
                o['mask'] = m
            for o, oo in zip(normalized_objects, outlines):
                o['outline'] = [Point(pt.x / width, pt.y / height)
                                for pt in oo]
        objects.sort(key=lambda x: x['left'])
        normalized_objects.sort(key=lambda x: x['left'])
        unpacked = dict(
            image_uuid=image_uuid,
            image_data=image_data,
            image_colorspace=image_colorspace,
            image_format=image_format,
            image_height=height,
            image_width=width,
            image_channels=channels,
            objects=objects,
            normalized_objects=normalized_objects,
            outline=outline)
        if kind == 'objects':
            unpacked['crop'] = crop
        yield unpacked


def decode_png(pngdata):
    return np.array(Image.open(BytesIO(pngdata)))


def sharded_example_writer(destination, split, prefix,
                           num_examples=MAX_EXAMPLES_PER_SHARD):
    current_shard=0
    example_count=0
    while True:
        shard_name = "%s%s_%05d.tfrecord" % (prefix, split, current_shard)
        output_filename = append_url(destination, shard_name)
        tfrecord_writer = tf.python_io.TFRecordWriter(
                output_filename.geturl())
        for _ in range(num_examples):
            example = yield
            if example is None:
                break
            tfrecord_writer.write(example)
            example_count += 1
        tfrecord_writer.close()
        if example is None:
            break
        current_shard += 1
    shard_count_file = append_url(
        destination, "TOTAL_%d_SHARDS_%s" % (current_shard + 1, split))
    save_to_url(("%d\n" % example_count).encode("utf-8"), shard_count_file)

def clip_rectangles_to_outline(rectangles, outline):
    contour = np.array([(pt.x, pt.y) for pt in outline])
    for rect in rectangles:
        detection_center = [rect['left'] + rect['width'] / 2.0,
                            rect['top'] + rect['height'] / 2.0]
        rect['on_blind'] = bool(points_in_poly([detection_center],
                                              contour)[0])
    return list(filter(lambda r: r['on_blind'], rectangles))

if __name__ == '__main__':
    main()

"""
"""