import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import tqdm

import pandas as pd

if 0: # merge with Eileen decision
    pkl_file = '1593440243_stat_collect_inference_best_qual.pkl'

    path = r'C:\Users\hanoch.kremer\OneDrive - ALLFLEX EUROPE\HanochWorkSpace\Projects\Results\quality_eval\voting_img_quality\MatHoldoutWithMarginal_fix_model_optimized_1593440243\cages_unsupervised_for_inspection'
    pkl_file = '1593440243_stat_collect_inference_best_qual.pkl'

    with open(os.path.join(path, pkl_file), 'rb') as f:
        results_meta = pickle.load(f)

    csv_file = 'post_proc_inference-results1593440243_th_0.31.csv'
    df_results = pd.read_csv(os.path.join(path, csv_file), index_col=False)

    eileen_annot = '_wlinkpost_proc_inference-results1593440243_th_0.31_blind (1)_Eileen.csv'
    df_eilen = pd.read_csv(os.path.join(path, eileen_annot), index_col=False)


    # avg pooling
    df_results['avg_pool'] = ""
    df_results['avg_pool_dec_0p32'] = ""
    df_results['avg_pool_dec_0p6'] = ""
    df_results['fname'] = df_results['file_name'].apply(lambda x: x.split('/')[-1])

    for cu_id in tqdm.tqdm(results_meta.keys()):
        res = results_meta[cu_id]
        label = res['label']
        prob_good_cls = res['tile_good_class_pred']
        # label of class 2 was reverted to 1 as bad
        avg_pool = prob_good_cls.mean()
        df_results['avg_pool'].loc[df_results.fname == cu_id] = avg_pool
        df_results['avg_pool_dec_0p32'].loc[df_results.fname == cu_id] = [3 if avg_pool > 0.32 else 1][0]
        df_results['avg_pool_dec_0p6'].loc[df_results.fname == cu_id] = [3 if avg_pool > 0.6 else 1][0]

    df = pd.merge(df_results, df_eilen, how='inner', on='cutout_id')
    df.to_csv(os.path.join(path, 'merged.csv'), index=False)


else:

    # path = r'C:\Users\hanoch.kremer\OneDrive - ALLFLEX EUROPE\HanochWorkSpace\Projects\Results\quality_eval\voting_img_quality\MatHoldoutWithMarginal_fix_model_optimized_1593440243\avg_pool\ee4062421d7045f7567ded0b34998a9c'

    # path = r'C:\Users\hanoch.kremer\OneDrive - ALLFLEX EUROPE\HanochWorkSpace\Projects\Results\quality_eval\voting_img_quality\MatHoldoutWithMarginal_fix_model_optimized_1593440243\cages_unsupervised_for_inspection'
    # pkl_file = '1593440243_stat_collect_inference_best_qual.pkl'
    # csv_file = 'inference-results1593440243_th_0.31.csv'
    id_fname_map_file = 'field_performance_cutouts.csv'
    save_filename_prefix = 'post_proc_cage_1536_list_'

    # path = r'C:\Users\hanoch.kremer\OneDrive - ALLFLEX EUROPE\HanochWorkSpace\Projects\Results\quality_eval\voting_img_quality\MatHoldoutWithMarginal_fix_model_optimized_1593440243\cages_unsupervised_for_inspection_pick_cnn_good_dec'
    # path = r'c:\Users\hanoch.kremer\OneDrive - ALLFLEX EUROPE\HanochWorkSpace\Projects\Results\quality_eval\voting_img_quality\MatHoldoutWithMarginal_fix_model_optimized_1593440243\cages_unsupervised_for_inspection_pick_cnn_good_dec\ReCalc_SVM'
    if 1:
        path = r'/hdd/hanoch/runmodels/img_quality/results/inference_production/cage_images_with_cutout_id/1600697434'
        pkl_file = '1600697434_stat_collect_inference_best_qual.pkl'
        csv_file = 'inference-results_partial1600697434_th_0.31.csv' #'inference-results1593440243_th_0.31.csv'
        id_fname_map_file = 'cutouts_path.csv'
        save_filename_prefix = 'post_proc_cage_all_list_'
    else:
        path = r'C:\temp\ReCalc_SVM'
        path = '/hdd/hanoch/data/ReCalc_SVM'
        pkl_file = '1593440243_stat_collect_inference_best_qual.pkl'
        csv_file = 'inference-results_partial1593440243_th_0.31.csv' #'inference-results1593440243_th_0.31.csv'
        id_fname_map_file = 'cutouts_path.csv'
        save_filename_prefix = 'post_proc_cage_all_list_'

    # eileen_annot = '_wlinkpost_proc_inference-results1593440243_th_0.31_blind (1)_Eileen.csv'
    # df_eilen = pd.read_csv(os.path.join(path, eileen_annot), index_col=False)

    df_results = pd.read_csv(os.path.join(path, csv_file), index_col=False)
    df_results['cutout_id_alias'] = df_results['cutout_id']
    assert len(df_results) != 0

    with open(os.path.join(path, pkl_file), 'rb') as f:
        results_meta = pickle.load(f)

    df_cutout_map = pd.read_csv(os.path.join(path, id_fname_map_file))
    df_cutout_map['fname'] = df_cutout_map['path'].apply(lambda x: x.split('/')[-1])

    if 0:
        lst = list()
        with open(os.path.join(path, 'cutouts_path_media_id_no_header.txt'), 'r') as cutouts_path_media_id:
            for line in tqdm.tqdm(cutouts_path_media_id):
                fields = line.split("|")
                try:
                    if len(fields) == 3:
                        dictionary = {"cutout_id": fields[0], "media_id": fields[1], "path": fields[2]}
                    elif len(fields) == 2:
                        dictionary = {"media_id": fields[0], "path": fields[1]}
                    else:
                        raise ValueError("unknown option")

                    lst.append(dictionary)
                except:
                    print(len(fields))
        df_cutout_map2 = pd.DataFrame(lst)
        df_cutout_map2.to_csv(os.path.join(path, 'cutouts_path_media_id_no_header.csv'), index=False)
    # else:
        # df_cutout_map2 = pd.read_csv(os.path.join(path, 'cutouts_path_media_id_no_header.csv'), index_col=False)
        # df_cutout_map2['fname'] = df_cutout_map2['path'].apply(lambda x: x.split('/')[-1])
        # df_cutout_map2['fname'] = df_cutout_map2['fname'].apply(lambda x: x.rstrip())

    if 0:
        # fields = ['cutout_id', 'media_id', 'path']
        # df_cutout_map2 = pd.read_csv(os.path.join(path, 'cutouts_path_media_id.csv'), usecols=fields, dtype={"path": "string"})
        # df_cutout_map2['fname'] = df_cutout_map2['path'].apply(lambda x: x.split('/')[-1])
        cutouts_path_media_id_map = np.genfromtxt(os.path.join(path, 'cutouts_path_media_id.txt'), dtype='str')
        df_cutouts_path_media_id_map = pd.DataFrame(cutouts_path_media_id_map)
        df_cutouts_path_media_id_map.columns = ['cutout_id', 'media_is', 'path']

    df_results['hyperlink'] = ""
    df_results['link'] = ""

    # avg pooling
    df_results['avg_pool'] = ""
    df_results['avg_pool_dec_0p7'] = ""
    df_results['avg_pool_dec_0p6'] = ""
    df_results['fname'] = df_results['file_name'].apply(lambda x: x.split('/')[-1])
    df_results['cage'] = df_results['file_name'].apply(lambda x: x.split('/')[-3])

    # df_cutout_map2['fname'] = df_cutout_map2['fname'].apply(lambda x: x.rstrip())
    if 0: # overrun
        df_cutout_map = df_cutout_map2

    hyperlink_str = 'https://annotator.scrdairy.com/annotator/#/viewer?cutout_id='
    for idx in tqdm.tqdm(range(len(df_results))):
        loc_ind = np.where(df_cutout_map['fname'] == df_results['cutout_id_alias'].iloc[idx])[0]
        if np.where(df_cutout_map['fname'] == df_results['cutout_id_alias'].iloc[idx])[0].size >0 :
            df_results.loc[idx, 'cutout_id'] = df_cutout_map.cutout_id.iloc[loc_ind].to_list()
            df_results.loc[idx, 'link'] = hyperlink_str + df_cutout_map.cutout_id.iloc[loc_ind].to_list()[0].strip()
            # df_results.loc[idx, 'hyperlink'] = "<a href="url">" + df_results.loc[idx, 'link'] + "</a>"
            #extract softmax and liklihood
            cu_id = df_results['fname'].iloc[idx]
            if results_meta.get(cu_id, None) is not None:
                res = results_meta[cu_id]
                label = res['label']
                prob_good_cls = res['tile_good_class_pred']
                # label of class 2 was reverted to 1 as bad
                avg_pool = prob_good_cls.mean()
                df_results['avg_pool'].loc[df_results.fname == cu_id] = avg_pool
                df_results['avg_pool_dec_0p7'].loc[df_results.fname == cu_id] = [3 if avg_pool > 0.7 else 1][0]
                df_results['avg_pool_dec_0p6'].loc[df_results.fname == cu_id] = [3 if avg_pool > 0.6 else 1][0]
                if df_results.loc[idx, 'predict_all_img_label'] != res['pred_label']:
                    print('From some reason labels are not equal !!!!!')
            else:
                print('ka')
    df_results.to_csv(os.path.join(path, save_filename_prefix + csv_file), index=False)
    if 0:
        for idx in range(len(df_cutout_map)):
            if df_cutout_map['path'].iloc[idx].split('/')[-1] in results_meta.keys():
                print(df_cutout_map['path'].iloc[idx])