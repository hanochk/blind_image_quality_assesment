import os
import argparse
import pickle
import numpy as np
import tqdm
from dev.utillities.tensorflow.tfrecords_utils import iterate_tfrecord_subset, parse_url, decode_png
from modules.blind_quality.quality import blindQualityNode
from PIL import Image
# record_path = '/home/pini/blindVision/lice-data-bbox-amb-simple-20200824/lice_test_00000.tfrecord'


def main():
    parser = argparse.ArgumentParser()

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--save-mode", default=False, action="store_true")
    parser.add_argument("--subsets", default=['train', 'validate', 'test'], type=lambda x: x.split(','))
    # parser.add_argument("--graph-path", default='/home/user/GIT/Finders/sandbox')
    # parser.add_argument("--container", default=-1, type=int)
    # parser.add_argument("--skip", default=[0, 0], type=lambda x: list(map(int, x.split(','))))
    # parser.add_argument("--entire", default=0, type=int, help="0 over all, N - first N examples out of subset")
    # parser.add_argument("--dev", default=False, action="store_true")
    parser.add_argument("--export-path", default=None)
    # parser.add_argument("--force-raw", default=False, action="store_true")
    parser.add_argument('--gpu-id', type=int, default=0, metavar='INT',
                        help="cuda device id ")

    parser.add_argument('--filter-by-quality-th', type=int, default=-1, metavar='INT',
                        help="cuda device id ")

    parser.add_argument("--quality-data-pickle-path", required=False)

    opts = parser.parse_args()

    if opts.filter_by_quality_th>-1:
        with open(opts.quality_data_pickle_path, 'rb') as f:
            blind_quality_filter = pickle.load(f)
            image_export_path = os.path.join(opts.export_path, 'images')
            if not os.path.exists(image_export_path):
                os.makedirs(image_export_path)


    # import sys
    # sys.path.append('/home/hanoch/GIT')
    # sys.path.append('/home/hanoch/GIT/Finders')
    # sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    # from Finders.finders.blind import default as blindfinder
    # from Finders.finders_tools.finders.tools import tfrecord_utils, generalized_iou, url_utils
# blind quality

    blind_quality = blindQualityNode(gpu_id=opts.gpu_id,
                                   debug_is_active=True)

    tfdir = parse_url(opts.data_path)
    res_dict = dict()
    for subset in opts.subsets:  #  opts.subsets:
        # cm = np.zeros((names_to_labels.__len__() + 1, names_to_labels.__len__() + 1))
        base_path = opts.export_path + '/' + subset
        if not os.path.exists(base_path) and opts.save_mode:
            os.makedirs(base_path)
        fn = base_path + '.pkl'
        if os.path.exists(fn):
            os.remove(fn)
        with open(fn, 'wb') as pickle_file:
            for ind, record in enumerate(tqdm.tqdm(iterate_tfrecord_subset(dataset_path=tfdir.geturl(),
                                                                           subset=subset))):
                # if opts.entire > 0:
                #     eval_counter += 1
                #     if eval_counter == opts.entire:
                #         return
                image = decode_png(record['image_data'])
                outline = record['outline']

                if opts.filter_by_quality_th > -1:
                    if blind_quality_filter[record['crop']['uuid']]['blind_quality_category'] >= opts.filter_by_quality_th:
                        pilimg = Image.fromarray((image).astype(np.uint8))
                        pilimg.save(os.path.join(image_export_path, 'cutout_' + record['crop']['uuid'] + '.png'))
                    else:
                        continue

                blind_quality_category, soft_score = blind_quality.handle_image(image=image, contour=outline)

                imgDict = {'blind_quality_category': blind_quality_category,
                           'blind_quality_soft_score': soft_score,
                           'img_uuid': record['image_uuid'],
                           'crop_uuid': record['crop']['uuid']}
                res_dict.update({record['crop']['uuid']: imgDict})
                if (ind % 100 == 0):
                    pickle.dump(res_dict,
                                pickle_file,
                                protocol=pickle.HIGHEST_PROTOCOL)
                    with open(base_path + 'fin.pkl', 'wb') as f:
                        pickle.dump(res_dict, f)

        # pickle.dump(res_dict,
        #             pickle_file,
        #             protocol=pickle.HIGHEST_PROTOCOL)
        with open(base_path + 'fin.pkl', 'wb') as f:
            pickle.dump(res_dict, f)

if __name__ == "__main__":
    main()


# /home/user/blindVision/lice-data-bbox-amb-simple-20200824
"""
nohup python -u ./dev/evaluations/eval_lice_det_based_quality.py --data-path /home/user/blindVision/lice-data-bbox-amb-simple-20200824 --subsets test --export-path /hdd/hanoch/results/eval_with_lice_finder --save-mode & tail -f nohup.out
"""