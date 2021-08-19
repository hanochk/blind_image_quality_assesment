import os
import re
import numpy as np
import tensorflow as tf
from itertools import groupby
from collections import namedtuple
from six.moves.urllib.parse import urlparse, quote, unquote
from PIL import Image
from io import BytesIO


Point = namedtuple('Point', ['x', 'y'])
OUTLINE_PTS = 1024

MAX_EXAMPLES_PER_SHARD = 256


class EnumeratedGenerator(object):
    def __init__(self, length, generator):
        self.length = length
        self.generator = generator

    def __len__(self):
        return self.length

    def __iter__(self):
        return self.generator

def read_count_file(dataset_path, subset):
    pattern = "TOTAL_(\d+)_SHARDS_%s" % (subset)
    for fname in tf.io.gfile.listdir(dataset_path):
        m = re.match(pattern, fname)
        if m is not None:
            shard_count = int(m.groups()[0])
            with tf.io.gfile.GFile(os.path.join(dataset_path, fname)) as f:
                example_count = int(f.read())
            return shard_count, example_count
    return None, None

def iterate_tfrecord_subset(dataset_path, subset, mode='objects', skip=None):
    shard_count, example_count = read_count_file(dataset_path, subset)
    if skip is not None:
        example_count = int(np.ceil(float(example_count) / (skip[1]+1)))

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
    record_iterator = iter(tf.compat.v1.io.tf_record_iterator(path=full_input_path))
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
        cutout_uuid = None
        if kind == 'objects':
            # cutout_uuid = (example.features.feature['image/cutout_uuid'].bytes_list
            #                                                             .value[0])
            # cutout_uuid = cutout_uuid.decode('ascii')

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
            # cutout_uuid=cutout_uuid,
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


def parse_url(url):
    url = urlparse(url)
    if url.scheme == 'file':
        if len(url.path.lstrip('/')) == 0:
            raise Exception("Invalid file:// url, no path specified. Did you "
                            "mean file:///?")
        if len(url.netloc) > 0 and url.netloc != 'localhost':
            raise Exception("Invalid file:// url, netloc is neither empty nor "
                            "localhost. Did you mean file:///?")
    url = url._replace(path=unquote(url.path))
    return url

def decode_png(pngdata):
    return np.array(Image.open(BytesIO(pngdata)))
