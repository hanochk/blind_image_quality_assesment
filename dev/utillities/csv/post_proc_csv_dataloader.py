import os
import pandas as pd
import subprocess
from PIL import Image
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import tqdm
import cv2

# This endpoint script aims to post processing train/val/test csv file in order to add additional meta data

def add_num_edges_between_thresholds(data_df, **kwargs):

    canny_edges_sum_50_70_acm = list()
    data_df['canny_edges_sum_50_70'] = ""
    for idx in tqdm.tqdm(range(len(data_df))):
        full_file_name = data_df['full_file_name'].iloc[idx]

        img = mpimg.imread(full_file_name)
        gimg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gimg = gimg ** 0.5 # Gamma correction
        edges = cv2.Canny(np.uint8(gimg * 255), 50, 70)
        canny_edges_sum_50_70 = np.sum(edges) / 256 / 100
        data_df['canny_edges_sum_50_70'].iloc[idx] = canny_edges_sum_50_70
        canny_edges_sum_50_70_acm.append(canny_edges_sum_50_70)
        # if idx == 200:
        #     break
    mean_canny = data_df['canny_edges_sum_50_70'].mean() #np.mean(np.array(canny_edges_sum_50_70_acm))
    std_canny = data_df['canny_edges_sum_50_70'].std() #np.std(np.array(canny_edges_sum_50_70_acm))

    if kwargs['compute_mean_std']:
        data_df['canny_edges_sum_50_70_mean'] = mean_canny
        data_df['canny_edges_sum_50_70_std'] = std_canny
    elif kwargs['propagte_mean_std']:
        data_df['canny_edges_sum_50_70_mean'] = kwargs['mean_feat']
        data_df['canny_edges_sum_50_70_std'] = kwargs['std_feat']

    return data_df, mean_canny, std_canny

def add_df_all_image_weighted_hue(train_df, w_huw_calc=False):
    cutouts = train_df.cutout_uuid.unique()
    # for index, row in train_df.iterrows():
    train_df["all_image_weighted_hue"] = ""
    train_df["all_image_mean_sat"] = ""
    train_df["n_tiles"] = ""
    train_df["n_tiles_mean"] = ""
    train_df["n_tiles_std"] = ""
    size_acm = list()

    for cutout in tqdm(cutouts):
        df = train_df[train_df.cutout_uuid == cutout]
        if w_huw_calc:
            image_hsv_acm_list = []
            for index, row in df.iterrows():
                #    cutout_full_path = subprocess.getoutput('find ' + cutout_path + ' -iname ' + '"*' + cutout + '.png"')
                img = Image.open(row.full_file_name)
                img = img.convert('RGB')
                image_hsv = mcolors.rgb_to_hsv(img)
                image_hsv_acm_list.append(image_hsv)

            image_hsv_acm = np.concatenate(image_hsv_acm_list)
            sv_norm = image_hsv_acm[:, :, 1] * image_hsv_acm[:, :, 2]
            sv_norm = sv_norm.mean(axis=0).mean(axis=0)
            norm_hue = image_hsv_acm[:, :, 0] * image_hsv_acm[:, :, 1] * image_hsv_acm[:, :, 2] / sv_norm
            norm_hue = norm_hue.mean(axis=0).mean(axis=0)
            train_df["all_image_weighted_hue"][train_df.cutout_uuid == cutout] = norm_hue
            train_df["all_image_mean_sat"][train_df.cutout_uuid == cutout] = image_hsv_acm[:, :, 1].mean()
        train_df["n_tiles"][train_df.cutout_uuid == cutout] = np.where(train_df.cutout_uuid == cutout)[0].size
        size_acm.append(np.where(train_df.cutout_uuid == cutout)[0].size)

    train_df["n_tiles_mean"] = np.mean(np.array(size_acm))
    train_df["n_tiles_std"] = np.std(np.array(size_acm))

    return train_df

def calc_hs_hist_per_image(train_df, norm_hue_by_meta_data=False):
    cutouts = train_df.cutout_uuid.unique()
    # for index, row in train_df.iterrows():
    hue_avg_acm = []
    sat_avg_acm = []
    for cutout in tqdm.tqdm(cutouts):
        df = train_df[train_df.cutout_uuid == cutout]
        image_hsv_acm_list = []
        for index, row in df.iterrows(): # only over the relevant tiles within the outline
            #    cutout_full_path = subprocess.getoutput('find ' + cutout_path + ' -iname ' + '"*' + cutout + '.png"')
            img = Image.open(row.full_file_name)
            img = img.convert('RGB')
            image_hsv = mcolors.rgb_to_hsv(img)
            if 0:
                if 1:  # per image normalization of hue: the factor was calculated offline and is in the csv
                    # renorm by mean of hsv per image to a predefined mean which correspond to the train set stat
                    img_weighted_hue = train_df["all_image_weighted_hue"][train_df.cutout_uuid == cutout].iloc[0]
                    image_hsv_intern = mcolors.rgb_to_hsv(np.array(img) / 255)
                    import copy
                    h_norm = image_hsv_intern[:, :, 0] - img_weighted_hue + 0.5  # 0.5 works fine also
                    h_norm[h_norm < 0] = h_norm[h_norm < 0] + 1  # wrap arround % 1.0
                    image_hsv_new = np.dstack((h_norm, image_hsv_intern[:, :, 1], image_hsv_intern[:, :, 2]))
                    hsv2rgb = mcolors.hsv_to_rgb(image_hsv_new)
                    hsv2rgb = hsv2rgb * 255.0
                    if 0:
                        import pickle
                        path = '/hdd/hanoch/data/temp/hsv2rgb.pkl'
                        with open(path, 'wb') as f:
                            pickle.dump(hsv2rgb, f)
                        np.seterr(all='raise')

                    hsv2rgb[hsv2rgb > 255.0] = 255.0
                    img_norm = Image.fromarray(hsv2rgb.astype(np.uint8))
                    # TODO: add numpy.seterr(all='raise') to catch the   RuntimeWarning: invalid value encountered in greater
                    img_norm.save('/hdd/hanoch/data/temp/' + cutout + '.png')
                    img.save('/hdd/hanoch/data/temp/' + cutout + '_orig.png')

            image_hsv_acm_list.append(image_hsv)

        image_hsv_acm = np.concatenate(image_hsv_acm_list)
        if norm_hue_by_meta_data:
            image_hsv_acm[:, :, 0] = image_hsv_acm[:, :, 0] - train_df["all_image_weighted_hue"][train_df.cutout_uuid == cutout].iloc[0]

        hue_avg = image_hsv_acm[:, :, 0].mean(axis=0).mean(axis=0)
        sat_avg = image_hsv_acm[:, :, 1].mean(axis=0).mean(axis=0)

        hue_avg_acm.append(hue_avg)
        sat_avg_acm.append(sat_avg)
    return hue_avg_acm, sat_avg_acm

def main(args: list = None):

    path_csv = '/hdd/hanoch/data/objects-data-bbox-20191106-simple-sharded-part/tile_data'
    # csv_file = 'file_quality_tile_eileen_good_bad_val_bad_9_20_avg_pool_filt_conf_filt_no_edges_trn_tst_fitjar'
    if 1:
        csv_file = 'file_quality_tile_eileen_good_bad_val_bad_9_20_avg_pool_filt_conf_filt_no_edges_trn_tst_fitjar'
        database_root = '/hdd/hanoch/data/objects-data-bbox-20191106-simple-sharded-part/tile_data'
        process_one_csv_train_val_test = True  # used for train other csv types are only for test such as Eileen cages holdout
    else:
        csv_file = 'Cages_n5_no_marginals_tiles_eileen_blind_test_tiles'
        database_root = '/hdd/hanoch/data/cages_holdout/annotated_images_eileen/tile_data/test'
        process_one_csv_train_val_test = False  # used for train other csv types are only for test such as Eileen cages holdout

    # cutout_path = '/hdd/hanoch/data/objects-data-bbox-20191106-simple-sharded-part/cutout_data'

    # process_one_csv_train_val_test = False  # used for train other csv types are only for test such as Eileen cages holdout
    add_image_weighted_hue_ntiles = False
    add_num_edges = True
    post_proc_func = {1: add_df_all_image_weighted_hue,
                      2: add_num_edges_between_thresholds}
    meta_function_post_proc = [post_proc_func[1] if add_image_weighted_hue_ntiles is True else post_proc_func[2]][0]

    dataframe = pd.read_csv(os.path.join(path_csv, csv_file + '.csv'), index_col=False)

    if 1:
        csv_file_out = csv_file + '_post_proc'
        val_df = None
        train_df = dataframe.loc[dataframe['train_or_test'] == 'train']
        test_df = dataframe.loc[dataframe['train_or_test'] == 'test']
        if len(train_df) >0 :
            val_df = train_df[train_df['val'] == 1]
            train_df = train_df[train_df['val'] != 1]

        if process_one_csv_train_val_test:
            if len(train_df) >0 :
                train_df['full_file_name'] = train_df.apply(
                    lambda x: os.path.join(database_root, 'train', x['class'], x['file_name'] + '.png'), axis=1)
                # Sanity check
                assert train_df['full_file_name'].apply(
                    lambda x: os.path.isfile(x)).all(), "Some images referenced in the CSV file were not found"

            if val_df is not None :
                val_df['full_file_name'] = val_df.apply(
                    lambda x: os.path.join(database_root, 'val', x['class'], x['file_name'] + '.png'), axis=1)

                assert val_df['full_file_name'].apply(
                    lambda x: os.path.isfile(x)).all(), "Some images referenced in the CSV file were not found"

            if len(test_df) >0 :
                test_df['full_file_name'] = test_df.apply(
                    lambda x: os.path.join(database_root, 'test', x['class'], x['file_name'] + '.png'), axis=1)
                assert test_df['full_file_name'].apply(
                    lambda x: os.path.isfile(x)).all(), "Some images referenced in the CSV file were not found"
        else:
            train_df = test_df # switch to embedd mean and std of HCF which are only computed over training set
            del test_df
            if len(train_df) >0 :
                train_df['full_file_name'] = train_df.apply(
                    lambda x: os.path.join(database_root, x['file_name'] + '.png'), axis=1)

                assert train_df['full_file_name'].apply(
                    lambda x: os.path.isfile(x)).all(), "Some images referenced in the CSV file were not found"

        if len(train_df) > 0:
            kwargs = {'compute_mean_std': True}
            train_df, mean_feat_train, std_feat_train = meta_function_post_proc(train_df, **kwargs)
        if val_df is not None:
# copy HCF stat based on trin to val/test
            kwargs = {'compute_mean_std': False,
                      'propagte_mean_std': True,
                      'mean_feat': mean_feat_train,
                      'std_feat': std_feat_train}

            val_df, _, _ = meta_function_post_proc(val_df, **kwargs)
        if test_df is not None and len(test_df) > 0:
            kwargs = {'compute_mean_std': False,
                      'propagte_mean_std': True,
                      'mean_feat': mean_feat_train,
                      'std_feat': std_feat_train}
            test_df, _, _ = meta_function_post_proc(test_df, **kwargs)

        df_acm = pd.DataFrame()
        df_acm = df_acm.append((train_df))
        df_acm = df_acm.append((val_df))
        df_acm = df_acm.append((test_df))
        del df_acm['full_file_name']
        df_acm.to_csv(os.path.join(path_csv, csv_file_out + '.csv'), index=False)

    else:
        result_dir = '/hdd/hanoch/data/hue_sat_stat'

        database_cage = 'hue_weightedinference-results1604507681_th_0.25.csv'
        dataframe_cages = pd.read_csv(os.path.join(path, database_cage), index_col=False)

        train_df = dataframe.loc[dataframe['train_or_test'] == 'train']
        test_df = dataframe.loc[dataframe['train_or_test'] == 'test']
        val_df = train_df[train_df['val'] == 1]
        train_df = train_df[train_df['val'] != 1]

        dataframe_cages['full_file_name'] = dataframe_cages['file_name']
        dataframe_cages['cutout_uuid'] = dataframe_cages['cutout_id']

        assert dataframe_cages['full_file_name'].apply(
            lambda x: os.path.isfile(x)).all(), "Some images referenced in the CSV file were not found"

        train_df['full_file_name'] = train_df.apply(
            lambda x: os.path.join(database_root, 'train', x['class'], x['file_name'] + '.png'), axis=1)

        val_df['full_file_name'] = val_df.apply(
            lambda x: os.path.join(database_root, 'val', x['class'], x['file_name'] + '.png'), axis=1)

        test_df['full_file_name'] = test_df.apply(
            lambda x: os.path.join(database_root, 'test', x['class'], x['file_name'] + '.png'), axis=1)

        # Sanity check
        assert train_df['full_file_name'].apply(
            lambda x: os.path.isfile(x)).all(), "Some images referenced in the CSV file were not found"

        assert val_df['full_file_name'].apply(
            lambda x: os.path.isfile(x)).all(), "Some images referenced in the CSV file were not found"

        assert test_df['full_file_name'].apply(
            lambda x: os.path.isfile(x)).all(), "Some images referenced in the CSV file were not found"

        if 1: # calaulate weighted hue per image
            dataframe_cages = add_df_all_image_weighted_hue(dataframe_cages)
            dataframe_cages.to_csv(os.path.join(path, 'hue_weighted' + database_cage), index=False)

            train_df = add_df_all_image_weighted_hue(train_df)
            val_df = add_df_all_image_weighted_hue(val_df)
            test_df = add_df_all_image_weighted_hue(test_df)

            df_acm = pd.DataFrame()
            df_acm = df_acm.append((train_df))
            df_acm = df_acm.append((val_df))
            df_acm = df_acm.append((test_df))
            df_acm.to_csv(os.path.join(path, csv_file + '_weighted_hue_byimage'))
        else:
            norm_hue_by_meta_data = True
            tag = ''
            if norm_hue_by_meta_data:
                tag = 'norm_hue_by_meta_data'

            marker = ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']
            hue_dict = dict()

            hue_avg_cages, sat_avg_cages = calc_hs_hist_per_image(dataframe_cages, norm_hue_by_meta_data=norm_hue_by_meta_data)
            hue_dict = {'hue_avg_cages': hue_avg_cages, 'sat_avg_cages': sat_avg_cages}


            hue_avg_train, sat_avg_train = calc_hs_hist_per_image(train_df, norm_hue_by_meta_data=norm_hue_by_meta_data)
            hue_dict.update({'hue_avg_train':hue_avg_train, 'sat_avg_train':sat_avg_train})

            hue_avg_val, sat_avg_val = calc_hs_hist_per_image(val_df, norm_hue_by_meta_data=norm_hue_by_meta_data)

            hue_dict.update({'hue_avg_val':hue_avg_val, 'sat_avg_val':sat_avg_val})



            with open(os.path.join(result_dir, 'hue_sat_hist' + tag + '.pkl'), 'wb') as f:
                pickle.dump(hue_dict, f)

            # import matplotlib.cm as cm
            # colors = cm.rainbow(np.linspace(0, 1, len(ys)))

            fig, ax = plt.subplots()
            ax.scatter(hue_avg_train, sat_avg_train, marker=marker[0], s=2, c='green')
            ax.scatter(hue_avg_val, sat_avg_val, marker=marker[0], s=2, c='blue')
            ax.scatter(hue_avg_cages, sat_avg_cages, marker=marker[0], s=2, c='red')
            ax.set_ylabel('Saturation intensity : mean per image in outline')
            ax.set_xlabel('Hue intensity: mean per image in outline')
            ax.legend()
            # ax.set_xlim(0, 1.0)
            # ax.set_ylim(0, 1.0)
            ax.grid()
            plt.show()



        if 0:
            path = '/hdd/hanoch/data/cages_holdout'
            df = pd.read_csv(os.path.join(path, 'merged_eileen_blind_tests.csv'))

if __name__ == "__main__":
    main()
