import tensorflow as tf
#from finders.common.mask_utils import Point
from PIL import Image
from io import BytesIO
from collections import namedtuple
import tqdm
import numpy as np
import os
import pandas as pd
import pickle
from src.svm.svm import blindSvm
import logging

Point = namedtuple('Point', ['x', 'y'])



def decode_png(pngdata):
    return np.array(Image.open(BytesIO(pngdata)))


def main():
    # load data and train classifier once
    path_to_init_file = '/home/hanoch/GIT/blind_quality_svm/bin/trn/svm_benchmark/FACSIMILE_PRODUCTION_V0_1_5/fixed_C_200_facsimile___prod_init_hk.json'

    path_out = r'/hdd/hanoch/data/objects-data-bbox-20191106-simple-sharded-part'
    path_input = '/home/user/blindVision/objects-data-bbox-20191106-simple-sharded'
    # go over all TFrecords train/val/test
    sets_in_tfrecords = ['train', 'test', 'validate']  
    
    # process the blind quality set
    blind_q_train_test_flag = 'holdout'
    
    path_out = os.path.join(path_out, blind_q_train_test_flag)

# mapper of cutout to crop from database
    with open('/home/user/GIT/Finders/sandbox/system_observations/txt.tmp') as txt_file:
        txt = txt_file.read()
    splitted = txt.split('\n')
    splitted = splitted[3:-4]
    splitted = [x.replace(" ", "") for x in splitted]
    splitted = [x.split('|') for x in splitted]
    crop_2cutout_dict = {x[1]: x[0] for x in splitted}

    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # Following the default path as an example, this line looks for all .json files in the trn submodule and picks the first one
    svm = blindSvm.blindSvm(blind_svm_params=path_to_init_file)

    if blind_q_train_test_flag is 'holdout':
        print('process the holdout set')
        pickle_fname = 'quality_holdout_outlines.pkl'
        svm_dataset_keys = svm.blind_svm_params.all_parameters['blind_keys_holdout']
    elif blind_q_train_test_flag is 'train':
        print('process the train set !!!!')
        pickle_fname = 'quality_training_outlines.pkl'
        svm_dataset_keys = svm.blind_svm_params.all_parameters['blind_keys_training']

    cutout_record = {}
    nested_record = {'image_uuid': [], 'outline': [], 'media_id' : [], 'crop_uuid': []}
    # crop_media_dict = {'media_id' : [], 'cutout_uuid':[]}
    n_nxt_found = 0

    for root, _, filenames in os.walk(path_input):
        filenames.sort()
        for file in filenames:
            # remove tfrecords0 ones
            if any([not any(char.isdigit() for char in file.split('.')[-1]) and typ in file for typ in sets_in_tfrecords]):
                print(file)
                logging.info(file)
                svm_dec = []
                result = []
                for idx2, rec in tqdm.tqdm(enumerate(iterate_tfrecord(os.path.join(path_input, file)))):
                    image = decode_png(rec['image_data'])
                    # crop_uuid
                    # print("SVM category {}".format(category))
                    crop_uuid = rec['crop']['uuid']
                    outline = rec['outline']

                    cutout_uuid = crop_2cutout_dict.get(crop_uuid, None)

                    if cutout_uuid is None:
                        print("Cutout uuid of crop {} not found".format(crop_uuid))

                    media_id = rec['image_uuid']
# save to csv the entire dataset mapping
                    result.append([crop_uuid, cutout_uuid, media_id])
# only for blind quality
                    if cutout_uuid is not None and cutout_uuid in svm_dataset_keys:
                        fname_save = media_id + '_' + cutout_uuid
                        #save to png since dataset is in jpeg
                        pilimg = Image.fromarray(image)
                        pilimg.save(os.path.join(path_out, fname_save + '.png'))
    #save to pickle fro the sake of the blind quality
                        nested_record = {'outline': outline, 'media_id': media_id, 'crop_uuid': crop_uuid}
                        cutout_record.update({cutout_uuid: nested_record})
# intermediate save , avoid all or nothing
                        if n_nxt_found % 10:
                            with open(os.path.join(path_out, pickle_fname), 'wb') as f:
                                pickle.dump(cutout_record, f)

                        n_nxt_found += 1
                        #no need to go further and check all the tf records check the next
                        if n_nxt_found+1 == len(svm_dataset_keys):
                            break

        df_res = pd.DataFrame(result)
        df_res.columns = ['crop_uuid', 'cutout_uuid', 'media_id']
        df_res.to_csv(os.path.join(path_out, blind_q_train_test_flag + '_meta.csv'), index=False)

    with open(os.path.join(path_out, pickle_fname), 'wb') as f:
        pickle.dump(cutout_record, f)

    return

# with open(os.path.join(path_out, 'quality_training_outlines.pkl'), 'rb') as f:
#   ff = pickle.load(f)

OUTLINE_PTS = 1024
#
MAX_EXAMPLES_PER_SHARD = 256

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
