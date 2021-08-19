from src.svm.svm import blindSvm
from src.svm.svm import blindSvmPredictors
import tensorflow as tf
#from finders.common.mask_utils import Point
from PIL import Image
from io import BytesIO
from collections import namedtuple
import tqdm
import numpy as np
import os
import pandas as pd
import subprocess
import sklearn.metrics
import pickle
import tqdm
import sys
import copy
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

def main():
    # load data and train classifier once
    l = '920c7f49-b90c-575f-a14a-682000e8f06e'
    d = 'a29547bb-0841-54a4-9ca7-799335db5da4'
    dd = '41184214-4732-5975-aa9a-0566304c4854'

    df = 'a29547bb-0841-54a4-9ca7-799335db5da4'

    path_to_init_file = '/home/hanoch/GIT/blind_quality_svm/bin/trn/svm_benchmark/FACSIMILE_PRODUCTION_V0_1_5/fixed_C_200_facsimile___prod_init_hk.json'
    svm = blindSvm.blindSvm(blind_svm_params=path_to_init_file)    #Following the default path as an example, this line looks for all .json files in the trn submodule and picks the first one
    print(svm.blind_svm_params.model_handlers.keys())

    path_outline_pkl = '/hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/merged_outlines.pkl'
    annotation_dataset_path = '/hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data'
    annotation_dataset_file = 'file_quality_tile_eileen_good_bad_val_bad_9_20_avg_pool_filt_conf_filt_no_edges_trn_tst_fitjar_hcf_marginal_ext.csv'
    path_images_database = '/hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data'
    path_save_results = '/hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/embedding_fusion_nonlinear_fix_bug'

    ref_set = 'val'
    print('process the {} set'. format(ref_set))

    dataframe = pd.read_csv(os.path.join(annotation_dataset_path, annotation_dataset_file), index_col=False)

    if ref_set == 'test':
        df = dataframe.loc[dataframe['train_or_test'] == ref_set]
    elif ref_set == 'train':
        df = dataframe.loc[dataframe['train_or_test'] == ref_set]
        df = df.loc[df['val'] == 0]
    elif ref_set == 'val': # val==1 is for sure validation else it may be train or test
        df = dataframe.loc[dataframe['val'] == 1]

    sys.path.append('/home/hanoch/GIT')
    sys.path.append('/home/hanoch/GIT/Finders')
    sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    from Finders.finders.blind import default as blindfinder
    with open(path_outline_pkl, 'rb') as f:
        cutout_2_outline_n_meta = pickle.load(f)
        print('direct')

    fname_empty = list()
    all_targets = list()
    all_predictions = list()
    result = dict()

    for cu in tqdm.tqdm(df.cutout_uuid.unique()):
        fname = cu
        print(cu)
        file_full_path = subprocess.getoutput('find ' + path_images_database + ' -iname ' + '"*' + fname + '*"')
        if file_full_path == '':
            #try 2nd option
            print('2nd option file search {}'.format(fname))
            fname1 = df[df['cutout_uuid'] == cu]['file_name'].iloc[0]
            fname2 = os.path.split(fname1)[-1].split('_blind_')[-1].split('.png')[0]
            file_full_path = subprocess.getoutput('find ' + path_images_database + ' -iname ' + '"*' + fname2 + '*"')
            if file_full_path == '':
                fname_empty.append(fname)
                print("!!!!!!!!! File is missing")
                print(fname2)
                raise
                continue
        cu_for_outline = cu
        if '\n' in file_full_path:
            file_full_path = [x if x.endswith('.png') else None for x in file_full_path.split('\n')][-1]
            # file_full_path = file_full_path.split('\n')[0]
        if 'blind' in file_full_path:
            cu_for_outline = 'blind_' + os.path.split(file_full_path)[-1].split('_blind_')[1]

        outline = cutout_2_outline_n_meta.get(cu_for_outline, None)
        if outline == None:
            print("Outline for cutout_id {} was not found {}!!!".format(cu, fname_empty))

            continue
        if 'outline' in outline:
            outline = outline['outline']
        if 'contour' in outline:
            outline = outline['contour']

        image = Image.open(file_full_path).convert('RGB') #np.array(Image.open(BytesIO(pngdata)))
        image = image.convert('RGB')
        image = np.asarray(image)

        all_targets.append(df[df['cutout_uuid'] == cu].label.iloc[0])
        pred = blindSvmPredictors.blindSvmPredictors(cutout=image, outline=outline)

        if 0:
            predictors = pred.get_predictors(svm.blind_svm_params)  # 0.1.2 oldish version
        else:
            predictors = pred.get_predictors(svm.blind_svm_params.predictors)

        category = svm.predict(predictors)[0]
        all_predictions.append(category)
        print("SVM category {}".format(category))
        result.update({cu: category})

    conf_mat = sklearn.metrics.confusion_matrix(y_true=all_targets, y_pred=all_predictions)
    print(conf_mat)
    print(fname_empty)
    print("conf mat in {1,3,2} class format : bad, good, marginal")
    all_predictions_2 = copy.deepcopy(all_predictions)
    all_targets_2 = copy.deepcopy(all_targets)

    all_predictions_2 = np.array(all_predictions_2)
    all_predictions_2[np.where(np.array(all_predictions) == 3)[0]] = 2
    all_predictions_2[np.where(np.array(all_predictions) == 2)[0]] = 3

    all_targets_2 = np.array(all_targets_2)
    all_targets_2[np.where(np.array(all_targets) == 3)[0]] = 2
    all_targets_2[np.where(np.array(all_targets) == 2)[0]] = 3

    conf_mat_1_3_2 = sklearn.metrics.confusion_matrix(y_true=all_targets_2, y_pred=all_predictions_2)
    print(conf_mat_1_3_2)


    # df_result = pd.DataFrame(result)
    # df_result = df_result.transpose()
    # df_result.to_csv(os.path.join(path_save_results, 'svm_results.csv'), index=False)
    result.update({'conf_mat': conf_mat})
    result.update({'conf_mat_1_3_2': conf_mat_1_3_2})
    with open(os.path.join(path_save_results, ref_set + 'svm_results.pkl'), 'wb') as f:
        pickle.dump(result, f)

    return
if __name__ == '__main__':
    main()

