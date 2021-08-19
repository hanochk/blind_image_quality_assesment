from src.svm.svm import blindSvm
from src.svm.svm import blindSvmPredictors
#from finders.common.mask_utils import Point
from PIL import Image
from io import BytesIO
from collections import namedtuple
import tqdm
import numpy as np
import os
import pandas as pd
import subprocess
import pickle
import pandas as pd
import json

# failed to add absolute path
# sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
# from Finders.finders_tools.finders.tools import tfrecord_utils, generalized_iou, url_utils
Point = namedtuple('Point', ['x', 'y'])


# import importlib.util
# spec = importlib.util.spec_from_file_location("tfrecord_utils", "/home/hanoch/GIT/Finders/finders_tools/finders/tools/tfrecord_utils.py")
# foo = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(foo)   # ugly solution :https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path

def decode_png(pngdata):
    return np.array(Image.open(BytesIO(pngdata)))

def extract_values(obj, key):
    """Pull all values of specified key from nested JSON."""
    arr = []

    def extract(obj, arr, key):
        """Recursively search for values of key in JSON tree."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    extract(v, arr, key)
                elif k == key:
                    arr.append(v)
        elif isinstance(obj, list):
            for item in obj:
                extract(item, arr, key)
        return arr

    results = extract(obj, arr, key)
    return results

def main():
    # load data and train classifier once
    path_to_init_file = '/home/hanoch/GIT/blind_quality_svm/bin/trn/svm_benchmark/FACSIMILE_PRODUCTION_V0_1_5/fixed_C_200_facsimile___prod_init_hk.json'
    path_png_data = '/hdd/hanoch/data/objects-data-bbox-20191106-simple-sharded-part/cutout_data'
    path_meta_json = '/hdd/S3download_blind_quality/dat/blind_db/wp2_0-2_0_1.json'  # big json
    # path_meta_json = '/hdd/S3download_blind_quality/dat/blind_db/wp2_5-2_0_1.json'
    raw_path_local = '/hdd/hanoch/data/objects-data-bbox-20191106-simple-sharded-part/raw_data/marginal'

    path_media_local = '/hdd/blind_raw_images'

    with open(path_meta_json) as f:
        data_json = json.load(f)

    cut_outs = extract_values(data_json, 'cutout_id')
    # s3_media_path
    frodo = list(data_json['data'])
    # for k in frodo:
    #     if (data_json['data'][k]['cutout_id'] == train_cut_uuid):
    #         print(k)
    train_test_flag = 'holdout'

    clsss_quality = {1 :'bad', 2:'marginal', 3: 'good'}

    svm = blindSvm.blindSvm(blind_svm_params=path_to_init_file)    #Following the default path as an example, this line looks for all .json files in the trn submodule and picks the first one

    if train_test_flag is 'train':
        print('process the train set')
        cutout_ids = svm.blind_svm_params.all_parameters['blind_keys_training']
        labels = svm.blind_svm_params.all_parameters['judgments_training']
        exist_cutout_path = os.path.join(path_png_data, 'train')
    elif train_test_flag is 'holdout':
        print('process the holdout set !!!!')
        cutout_ids = svm.blind_svm_params.all_parameters['blind_keys_holdout']
        labels = svm.blind_svm_params.all_parameters['judgments_holdout']
        exist_cutout_path = os.path.join(path_png_data, 'holdout')


    # aggregate labels and file names
    df_train_test = pd.DataFrame(columns=['train_cut_uuid', 'media_id', 'x', 'y', 'width', 'height', 'padded_x', 'padded_width',
                                          'padded_height', 'aspect_ratio', 'clsss', 'label'])
    # cutout extraction

    label_to_be_excluded = 2# in the future if label 2 (marginal is of interest change to dummy labels list [1,3])

    # print(svm.blind_svm_params.model_handlers.keys())
    for train_cut_uuid, label in tqdm.tqdm(zip(cutout_ids, labels)):
        # skip marginal label only good or bad
        if label == label_to_be_excluded:  # doesn't catch marginals
            continue
        file_found = 0
# if cutout isn;t found in my local data folder than get media_id (cutout_id) raw image
        for k in frodo:
            if (data_json['data'][k]['cutout_id'] == train_cut_uuid):
                file_found = 1
                s3_cutout_path = data_json['data'][k]['s3_media_path']
                media_id = data_json['data'][k]['media_id']
                x = data_json['data'][k]['x']
                y = data_json['data'][k]['y']
                width = data_json['data'][k]['width']
                height = data_json['data'][k]['height']
                padded_x = data_json['data'][k]['padded_x']
                padded_width = data_json['data'][k]['padded_width']
                padded_height = data_json['data'][k]['padded_height']
                aspect_ratio = data_json['data'][k]['aspect_ratio']

                df_train_test.loc[len(df_train_test)] = [train_cut_uuid, media_id, x, y, width, height, padded_x, padded_width, padded_height, aspect_ratio, clsss_quality[label], label]

                # Look at my local repo
                file_full_path = subprocess.getoutput('find ' + exist_cutout_path + ' -iname ' + '"*' + train_cut_uuid + '.png"')
                if file_full_path is not '':
                    print("Png file found in my local repo (from Tfrecords) {}".format(train_cut_uuid))
                    continue

                # search for the media_id
                file_full_path = subprocess.getoutput('find ' + path_media_local + ' -iname ' + '"*' + media_id + '"')
                if file_full_path is '':
                    os.system('aws s3 cp ' + '"' + data_json['data'][k]['s3_media_path'] + '"' + ' ' + raw_path_local)
                    print("cutout uuid get from aws {}".format(train_cut_uuid))
                    #continue to next record searching
                else:
                    os.system('cp ' + file_full_path  + ' ' + raw_path_local)
                    print("cutout uuid get from raw folder {}".format(train_cut_uuid))
                break
        if file_found == 0:
            print("blind_1 : cutout uuid {} was not found in json ".format(train_cut_uuid))

# crop-polygon is created implicitly and filters out the tiles accordingly

    df_train_test.to_csv(os.path.join(raw_path_local, train_test_flag + '_raw.csv'), index=False)

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

