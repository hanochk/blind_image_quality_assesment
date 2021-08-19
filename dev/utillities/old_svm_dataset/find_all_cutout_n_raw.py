import pandas as pd
import os
import numpy as np
import tqdm
import subprocess
import datetime

if 1:
    from src.svm.svm import blindSvm
    from src.svm.svm import blindSvmPredictors

    path_to_init_file = '/home/hanoch/GIT/blind_quality_svm/bin/trn/svm_benchmark/FACSIMILE_PRODUCTION_V0_1_5/fixed_C_200_facsimile___prod_init_hk.json'

    path_cutout_jpg = '/home/hanoch/GIT/blind_quality_svm/bin/dat/svm_annotator_cutouts'
    path_out = r'/hdd/hanoch/data/objects-data-bbox-20191106-simple-sharded-part'
    path_media_local = '/hdd/blind_raw_images'
    raw_path = '/hdd/hanoch/data/objects-data-bbox-20191106-simple-sharded-part/raw_data'
    path_cutout = '/hdd/hanoch/data/objects-data-bbox-20191106-simple-sharded-part/cutout_data'
    clsss_quality = {1 :'bad', 2:'marginal', 3: 'good'}
    train_test_flag = 'holdout'
    print("=====================================")
    print(train_test_flag)

    df_train_test = pd.DataFrame(columns=['path_cutout_jpg', 'cutout_uuid', 'cutout_path', 'media_id', 'media_path', 'clsss', 'label'])
    svm = blindSvm.blindSvm(blind_svm_params=path_to_init_file)    #Following the default path as an example, this line looks for all .json files in the trn submodule and picks the first one


    if train_test_flag is 'training':
        print('process the train set')
        svm_dataset_keys = svm.blind_svm_params.all_parameters['blind_keys_training']
        labels = svm.blind_svm_params.all_parameters['judgments_training']
        # exist_cutout_path = os.path.join(path_png_data, 'train')
    elif train_test_flag is 'holdout':
        print('process the holdout set !!!!')
        svm_dataset_keys = svm.blind_svm_params.all_parameters['blind_keys_holdout']
        labels = svm.blind_svm_params.all_parameters['judgments_holdout']
        # exist_cutout_path = os.path.join(path_png_data, 'holdout')

    result = []

    for cutout_uuid, label in tqdm.tqdm(zip(svm_dataset_keys, labels)):
        # skip marginal label only good or bad
        if label == 2:
            continue
        file_found = 0

        file_full_path = subprocess.getoutput('find ' + path_cutout_jpg + ' -iname ' + '"*' + cutout_uuid + '.jpg"')

        if file_full_path is '':
            print("Png file hasn;t found {}".format(cutout_uuid))
            continue

        file_name = os.path.basename(file_full_path)
        media_id = file_name.split('_')[0]
        cutout_id = file_name.split('_')[1].split('.')[0]  # should be equal to train_cut_uuid

        # Hierarchial search
        # chack if cutout is alreay exist
        cutout_full_path = subprocess.getoutput('find ' + path_cutout + ' -iname ' + '"*' + cutout_id + '.png"')

        #search for that media_id in raw data media id based
        media_path = subprocess.getoutput('find ' + path_media_local + ' -iname ' + '"*' + media_id + '"')

        if media_path is '':
            # print("Raw1 file hasn;t found media_id {}".format(media_id))
            media_path = subprocess.getoutput('find ' + raw_path + ' -iname ' + '"*' + media_id + '*"')
            # if media_path is '':
                # print("Raw2 file hasn;t found local raw dataset {}".format(media_id))


        # search for the media_id in 2nd raw location

        df_train_test.loc[len(df_train_test)] = [path_cutout_jpg, cutout_uuid, cutout_full_path, media_id, media_path, clsss_quality[label], label]

    # only for blind quality
                #save to png since dataset is in jpeg
    x = datetime.datetime.now()
    df_train_test.to_csv(os.path.join(path_out, train_test_flag + str(x.today()) +'_media_cutout_data.csv'), index=False)

elif 0:



    raw_path_local = '/hdd/hanoch/data/objects-data-bbox-20191106-simple-sharded-part/raw_data'
    train_test_flag = 'holdout'
    path_raw_png = os.path.join(raw_path_local, train_test_flag)
    df = pd.read_csv(os.path.join(path_raw_png, 'holdout_raw.csv'))

    for root, _, filenames in os.walk(path_raw_png):
        filenames.sort()
        for idx, file in enumerate(filenames):
            if (any(np.where(df['media_id'] == file.split('.')[0])[0])):
                print(file)
            else:
                print("not found {}".format(file.split('.')[0]))

