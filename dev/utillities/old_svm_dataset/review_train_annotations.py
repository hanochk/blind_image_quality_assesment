from src.svm.svm import blindSvm
from modules.blind_quality.svm_utils.blindSvmPredictors import blindSvmPredictors as blindSvmPredictors
import os
import tqdm
import pickle
from collections import namedtuple
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

clsss_quality = {1: 'bad', 2: 'marginal', 3: 'good', 4: 'unknown-tested'}


def plot_outline_with_comments(image, outline, path, cutout_id, acc, label, title_image):
    # plot_image_and_outline(image, [outline], 10)
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.plot([p[0] for p in outline], [p[1] for p in outline])
    ax.title.set_text(title_image)
    plt.savefig(os.path.join(path, title_image + '.png'))
    plt.close()


Point = namedtuple('Point', ['x', 'y'])

path_to_init_file = '/home/hanoch/GIT/blind_quality_svm/bin/trn/svm_benchmark/FACSIMILE_PRODUCTION_V0_1_5/fixed_C_200_facsimile___prod_init_hk.json'
path_data = '/hdd/hanoch/data/objects-data-bbox-20191106-simple-sharded-part/cutout_data'
# path_tile_data = '/hdd/hanoch/data/objects-data-bbox-20191106-simple-sharded-part/tile_data'
# pickle_file_name = 'quality_training_outlines_temp.pkl'
result_dir = '/hdd/hanoch/runmodels/img_quality/results/inference_production/benchmark_svm'

train_test_flag = 'holdout'
tile_size = 256

clsss_quality = {1: 'bad', 2: 'marginal', 3: 'good'}

svm = blindSvm.blindSvm(
    blind_svm_params=path_to_init_file)  # Following the default path as an example, this line looks for all .json files in the trn submodule and picks the first one
print(svm.blind_svm_params.model_handlers.keys())

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
    features = svm.blind_svm_params.all_parameters['predictors_raw_holdout']
    labels = svm.blind_svm_params.all_parameters['judgments_holdout']
    # path_tiles = os.path.join(path_tile_data, 'test')
    # pickle_file_name = 'quality_holdout_outlines.pkl'
    pickle_file_name = 'quality_holdout_outlines_merged.pkl'
    pickle_path = os.path.join(path_data, 'holdout')
    train_or_test = 'test'
    outline_pickle_path = '/hdd/hanoch/data/objects-data-bbox-20191106-simple-sharded-part/cutout_data/holdout/quality_holdout_outlines_merged.pkl'
# aggregate labels and file names


# path_suspects_images = '/hdd/hanoch/runmodels/img_quality/results/inference_production/trainset/softmap'
path_suspects_images = '/hdd/hanoch/data/objects-data-bbox-20191106-simple-sharded-part/cutout_data/holdout'
filenames = [os.path.join(path_suspects_images, x) for x in os.listdir(path_suspects_images)
             if x.endswith('png')]

with open(outline_pickle_path, 'rb') as f:
    cutout_2_outline_n_meta = pickle.load(f)

for file in tqdm.tqdm(filenames):
    if file.split('.')[-1] == 'png':
        img = Image.open(file).convert("RGB")
        image = np.asarray(img)

    cutout_id = file.split('.')[-2].split('_')[-1]
    outline = cutout_2_outline_n_meta.get(cutout_id, None)
    if outline == None:
        print("Outline for cutout_id {} was not found {}!!!".format(cutout_id, file))
        continue
    if isinstance(outline, dict):
        outline = outline['outline']

    pred = blindSvmPredictors.blindSvmPredictors(cutout=image, outline=outline)
    predictors = pred.get_predictors(svm.blind_svm_params.predictors)
    quality_svm = svm.predict(predictors)[0]
    if 1:  # debug to MAtt annotations of the hold outs
        # actual = labels
        cutout_ids_svm = cutout_ids
        id = [i for i, x in enumerate(cutout_ids_svm) if x == cutout_id]
        # delta_features = (features[id[0]] - predictors).sum()
        max_delta_feast = np.max(np.abs((features[id[0]] - predictors)))
        feat_id = np.argmax(np.abs((features[id[0]] - predictors)))
        svm_predict_json_feat = svm.predict(np.array(features[id[0]]).reshape(1, -1))[0]

        if svm_predict_json_feat != quality_svm:
            feat_name = svm.blind_svm_params.all_parameters['predictors'][feat_id]
            if max_delta_feast < 1:
                print('')
            print("Huston we have a problem id: {} max |delta_feat| {} feat_id {} feat name {}".format(cutout_id, max_delta_feast, feat_id, feat_name))
            # quality_svm = actual[id[0]]
            plot_outline_with_comments(image, outline, result_dir, cutout_id, quality_svm, clsss_quality[quality_svm],
                                       'DD' + str(cutout_id) + '_Json_cls_' + str(svm_predict_json_feat) + '_Svm_' + str(quality_svm) + 'dFeat_' + str(max_delta_feast.__format__('.3f')) + 'id_' + str(feat_id))
        # else:
        #     print("Max |Delta_feat| {} feat_id {}".format(max_delta_feast, feat_id))
        #     plot_outline_with_comments(image, outline, result_dir, cutout_id, quality_svm, clsss_quality[quality_svm],
        #                                'contour_' + str(cutout_id) + '_Json_cls_' + str(
        #                                    svm_predict_json_feat) + '_Svm_' + str(quality_svm) + 'dFeat_' + str(max_delta_feast.__format__('.3f'))+ 'id_' + str(feat_id))

    # score = file.split('/')[-1].split('_')[2]
    # cutout_id = file.split('/')[-1].split('_')[0]
    # actual_class = file.split('/')[-1].split('_')[6]
    # if (float(score)<0.7):
    #     ind_in_annot = cutout_ids.index(cutout_id)
    #     print(labels[ind_in_annot])
    #     print(cutout_id, score)
    #     if actual_class != clsss_quality[labels[ind_in_annot]]:
    #         raise()