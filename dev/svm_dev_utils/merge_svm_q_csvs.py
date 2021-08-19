import pandas as pd
import os
import numpy as np
# Creating train/test split out of scv set description in the following format :
# columns=['file_name', 'cutout_uuid', 'class', 'label', 'percent_40', 'percent_60', 'percent_80', 'in_outline_tiles_id', 'train_or_test', 'media_uuid']
if 1:
    # path_input = '/hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/test_eileen_best_qual/csv'
    path_input = '/hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/test_eileen_best_qual/bad_class/csv'
    # path_out = '/hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/test_eileen_best_qual/csv/inference_res'
    path_out = path_input
    train_test_split = 0.8
    df_tbl = pd.DataFrame()
    filenames = [os.path.join(path_input, x) for x in os.listdir(path_input)
                     if x.endswith('csv')]


    for file in filenames:
        if file.endswith(".csv"):
            print(file)
            df_tbl = df_tbl.append([pd.read_csv( file, index_col=False)])
    # df_tbl.columns = ['images_quality', 'crop_uuid']
    print(len(df_tbl))

    lenth = []
    for cutout in df_tbl.cutout_uuid.unique():
        print("Cutout_id {} N = {}".format(cutout, len(df_tbl[df_tbl.cutout_uuid == cutout])))
        lenth += [len(df_tbl[df_tbl.cutout_uuid == cutout])]
    df_tbl.to_csv(os.path.join(path_out, 'merged_csvs.csv'), index=False)

    df_tbl['train_or_test'] = 'train'
    ordered_ = np.array(lenth).argsort()
    cs = np.cumsum(np.sort(np.array(lenth)))
    ind_last_ele = len(cs[cs< (1-train_test_split)*len(df_tbl)]) -1
    cutout_ind_testset = ordered_[:ind_last_ele]
    cutout_id_testset = df_tbl.cutout_uuid.unique()[cutout_ind_testset]

    df_tbl.reset_index()
    for cutout_test in cutout_id_testset:
        print(len(df_tbl[df_tbl['cutout_uuid'] == cutout_test]))
        df_tbl.loc[(df_tbl['cutout_uuid'] == cutout_test), ['train_or_test']] = 'test'
        print(len(df_tbl[df_tbl['train_or_test'] == 'test']))

    df_tbl.to_csv(os.path.join(path_out, 'merged_train_test_split.csv'), index=False)

else:
    path_input = '/hdd/hanoch/results/blind_quality_0p1p5'

    df_tbl = pd.DataFrame()
    for root, _, filenames in os.walk(path_input):
        for file in filenames:
            if file.endswith(".csv") and 'test' in file:
                print(file)
                df_tbl = df_tbl.append([pd.read_csv(os.path.join(path_input, file), index_col=False)])
                print(len(df_tbl))
        df_tbl.columns = ['images_quality', 'crop_uuid']
        df_tbl.to_csv(os.path.join(path_input, 'blind_q_test.csv'), index=False)
