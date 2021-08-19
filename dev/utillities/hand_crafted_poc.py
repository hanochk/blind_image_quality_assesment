import numpy as np
import matplotlib.pyplot as plt
import os
import json
from argparse import ArgumentParser
from src.CNN.utillities.math_utils import spectralOmni, plot_2d_stat, plot_3d_stat, mtf
from torchvision import transforms
from PIL import Image
import pandas as pd
import tqdm
import sklearn.metrics
from src.CNN.inference import roc_plot



def n_tiles_stat(df_tiles, result_dir, database_root):
    n_tiles_bad = list()
    n_tiles_good = list()

    for c_id in tqdm.tqdm(df_tiles.cutout_uuid.unique()):
        n_tiles = np.where(df_tiles.cutout_uuid == c_id)[0].size
        type = df_tiles['class'][df_tiles.cutout_uuid == c_id].iloc[0]
        if type == 'bad':
            n_tiles_bad.append(n_tiles)
        elif type == 'good':
            n_tiles_good.append(n_tiles)
        else:
            raise

    return n_tiles_bad, n_tiles_good


def vollath_f4(img):
    vol_f4 = (
        np.sum(img[:-1, :] * img[1:, :]) - np.sum(img[:-2, :] * img[2:, :]))
    return vol_f4

def vollath_f4_col(img):
    vol_f4 = (
        np.sum(img[:, :-1] * img[:, 1:]) - np.sum(img[:, :-2] * img[:, 2:]))
    return vol_f4

def vollath_fx_tiles_stat(df_tiles, result_dir, database_root, volath_type):
    vollath_bad = list()
    vollath_good = list()

    filenames = [os.path.join(database_root, x) for x in os.listdir(database_root)
                   if x.endswith('png')]

    for i, file in enumerate(tqdm.tqdm(filenames)):
        # print(i, file)
        fname_in_csv = os.path.split(file)[-1].split('.png')[0]

        if len(df_tiles['class'][df_tiles.file_name == fname_in_csv]) == 0:
            # print('Tile info in csv not found')
            continue
        type_cls = df_tiles['class'][df_tiles.file_name == fname_in_csv].item()

        # img = Image.open(file).convert("RGB")
        img_gray = Image.open(file).convert("L")
        image = np.asarray(img_gray)

        vollath_x = globals()[f'vollath_f{volath_type}'](image.astype('float'))
        # vollath_x = vollath_f4(image.astype('float'))
        # vollath_x = vollath_f4_col(image.astype('float'))

        if type_cls == 'bad':
            vollath_bad.append(vollath_x)
        elif type_cls == 'good':
            vollath_good.append(vollath_x)
        else:
            raise
    plt.hist(vollath_bad, bins=100, alpha=0.5, label='bad')
    plt.hist(vollath_good, bins=100, alpha=0.5, label='good')
    plt.legend(loc='upper right')
    plt.title('Vollath_f4 focus feature N = {}'.format(len(vollath_bad) + len(vollath_good)))
    plt.grid()
    plt.show()

    all_predictions = [np.array(vollath_bad), np.array(vollath_good)]
    all_predictions = np.concatenate(all_predictions)
    all_targets = [np.zeros_like(vollath_bad), np.ones_like(vollath_good)]
    all_targets = np.concatenate(all_targets)
    auc = sklearn.metrics.roc_auc_score(y_true=all_targets, y_score=all_predictions)
    print("auc{}".format(auc))
    roc_plot(labels=all_targets, predictions=all_predictions, positive_label=1, save_dir=result_dir)

    return vollath_bad, vollath_good

def spectrum_analyse(df_tiles, result_dir, database_root, omitt_dc=True):
    filenames = [os.path.join(database_root, x) for x in os.listdir(database_root)
                   if x.endswith('png')]

    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, len(filenames))]

    # spectOmni_acm = np.empty(512)
    # spectOmni_norm1_acm = np.empty(512)
    spectOmni_acm = list()
    spectOmni_norm1_acm_good_cls = list()
    spectOmni_norm1_acm_bad_cls = list()
    stat_2d = True
    plt.ion()
    for i, file in enumerate(tqdm.tqdm(filenames)):
        # print(i, file)
        fname_in_csv = os.path.split(file)[-1].split('.png')[0]

        if len(df_tiles['class'][df_tiles.file_name == fname_in_csv]) == 0:
            print('Tile info in csv not found')
            continue
        class_of_tile = df_tiles['class'][df_tiles.file_name == fname_in_csv].item()
        fname = class_of_tile

        # img = Image.open(file).convert("RGB")
        img_gray = Image.open(file).convert("L")
        image = np.asarray(img_gray)

        # spectOmni, spec2d = spectralOmni(image, 'hamming')
        # spectOmni_norm1 = spectOmni / sum(spectOmni)
        mean_radi_spec, error, npts = mtf(image)

        mean_radi_spec = mean_radi_spec/mean_radi_spec.sum()
        if omitt_dc:
            tag = '_no_dc_'
            mean_radi_spec = mean_radi_spec[1:]
            mean_radi_spec = mean_radi_spec/mean_radi_spec.sum()

        perc_stat = [np.percentile(mean_radi_spec, 25), np.percentile(mean_radi_spec, 50), np.percentile(mean_radi_spec, 75)]

        plt.figure()
        plt.semilogy(mean_radi_spec)
        # plt.semilogy(spectOmni_norm1)
        plt.title("_".join(fname_in_csv.split('_')[1:]) + tag)
        plt.ylim([1e-7, 1])
        plt.grid()
        plt.text(0.2, 0.2, s="{}".format([val.__format__('.2e') for val in perc_stat]))
        plt.savefig(os.path.join(result_dir,
                                 "_".join(fname_in_csv.split('_')[1:]) + tag + 'omni_spectrum.png'), format="png")

        plt.close()

        plt.figure()
        plt.semilogy(error)
        # plt.semilogy(spectOmni_norm1)
        plt.title("_".join(fname_in_csv.split('_')[1:]) + tag +  ' : std')
        plt.ylim([1e-8, 1])
        plt.grid()
        plt.savefig(os.path.join(result_dir,
                                 "_".join(fname_in_csv.split('_')[1:]) + tag + 'omni_spectrum_std.png'), format="png")
        plt.close()
        # plt.show()
        if stat_2d:
            if class_of_tile == 'good':
                spectOmni_norm1_acm_good_cls.append(mean_radi_spec[np.newaxis, :])
            elif class_of_tile == 'bad':
                spectOmni_norm1_acm_bad_cls.append(mean_radi_spec[np.newaxis, :])
            else:
                raise
        else:
            raise

    if stat_2d:
        plot_2d_stat(np.concatenate(spectOmni_norm1_acm_good_cls), fname=fname+'good')
        plot_2d_stat(np.concatenate(spectOmni_norm1_acm_bad_cls), fname=fname+'bad')
    else:
        plot_3d_stat(spectOmni_acm, fname=fname)



def zero_crossing(df_tiles, result_dir, database_root, omitt_dc=True):


    filenames = [os.path.join(database_root, x) for x in os.listdir(database_root)
                   if x.endswith('png')]

    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, len(filenames))]

    df_res = pd.DataFrame(columns=['file_name', 'class', 'row_zc_75', 'row_zc_95', 'row_zc_98', 'col_zc_75', 'col_zc_95', 'col_zc_98'])


    # spectOmni_acm = np.empty(512)
    # spectOmni_norm1_acm = np.empty(512)
    spectOmni_acm = list()
    spectOmni_norm1_acm_good_cls = list()
    spectOmni_norm1_acm_bad_cls = list()
    stat_2d = True
    plt.ion()
    for i, file in enumerate(tqdm.tqdm(filenames)):
        # print(i, file)
        fname_in_csv = os.path.split(file)[-1].split('.png')[0]

        if len(df_tiles['class'][df_tiles.file_name == fname_in_csv]) == 0:
            # print('Tile info in csv not found')
            continue
        class_of_tile = df_tiles['class'][df_tiles.file_name == fname_in_csv].item()
        fname = class_of_tile

        # img = Image.open(file).convert("RGB")
        img_gray = Image.open(file).convert("L")
        image = np.asarray(img_gray)

        image_biased = image.astype('float') - image.max()/2

        zero_crossings_acm = list()
        for row in range(image_biased.shape[0]):
            zero_crossings = np.where(np.diff(np.sign(image_biased[row, :])))[0].size
            zero_crossings_acm.append(zero_crossings)

        zero_crossings_col_acm = list()
        for col in range(image_biased.shape[1]):
            zero_crossings = np.where(np.diff(np.sign(image_biased[:, col])))[0].size
            zero_crossings_col_acm.append(zero_crossings)

        perc_stat_col = [np.percentile(zero_crossings_col_acm, 75), np.percentile(zero_crossings_col_acm, 95), np.percentile(zero_crossings_col_acm, 98)]
        perc_stat_row = [np.percentile(zero_crossings_acm, 75), np.percentile(zero_crossings_acm, 95), np.percentile(zero_crossings_acm, 98)]

        g = [file, class_of_tile]
        g += ([j for i in [perc_stat_row, perc_stat_col] for j in i])
        df_res.loc[len(df_res)] = g

        if (i%10)==0:
            df_res.to_csv(os.path.join(result_dir, 'zero_crossing.csv'), index=False)

        if 0:
            plt.figure()
            plt.semilogy(image_biased)
            # plt.semilogy(spectOmni_norm1)
            plt.title("_".join(fname_in_csv.split('_')[1:]) + tag)
            plt.ylim([1e-7, 1])
            plt.grid()
            plt.text(0.2, 0.2, s="{}".format([val.__format__('.2e') for val in perc_stat]))
            plt.savefig(os.path.join(result_dir,
                                     "_".join(fname_in_csv.split('_')[1:]) + tag + 'omni_spectrum.png'), format="png")

            plt.close()

    df_res.to_csv(os.path.join(result_dir, 'zero_crossing.csv'), index=False)



def main(args: list = None):
    parser = ArgumentParser()

    parser.add_argument("--database-root", type=str, required=True, metavar='PATH',
                                        help="full path to the neural network model to use")

    parser.add_argument('--dataset-split-csv', type=str, required=False, metavar='PATH',
                             help="path to the csv defining the train-test split")

    parser.add_argument('--gpu-id', type=int, default=0, metavar='INT',
                        help="cuda device id ")

    args = parser.parse_args(args)

    args.result_dir = os.path.join(args.database_root, 'spectrum')

    n_tiles_calc = False
    spectrum_calc = False
    zc_calc = False

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    df_tiles = pd.read_csv(args.dataset_split_csv, index_col=False)

    vollath_bad, vollath_good = vollath_fx_tiles_stat(df_tiles=df_tiles, result_dir=args.result_dir,
                                                      database_root=args.database_root, volath_type=4)

    plt.hist(vollath_bad, bins=100, alpha=0.5, label='bad')
    plt.hist(vollath_good, bins=100, alpha=0.5, label='good')
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()

    if spectrum_calc:
        spectrum_analyse(df_tiles=df_tiles, result_dir=args.result_dir, database_root=args.database_root)
    if zc_calc:
        zero_crossing(df_tiles=df_tiles, result_dir=args.result_dir, database_root=args.database_root)

    if n_tiles_calc:
        n_tiles_bad, n_tiles_good = n_tiles_stat(df_tiles=df_tiles, result_dir=args.result_dir, database_root=args.database_root)

        plt.hist(n_tiles_bad, bins=100, alpha=0.5, label='bad')
        plt.hist(n_tiles_good, bins=100, alpha=0.5, label='good')
        plt.legend(loc='upper right')
        plt.grid()
        plt.show()

        all_predictions = [np.array(n_tiles_bad), np.array(n_tiles_good)]
        all_predictions = np.concatenate(all_predictions)
        all_targets = [np.zeros_like(n_tiles_bad), np.ones_like(n_tiles_good)]
        all_targets = np.concatenate(all_targets)

        auc = sklearn.metrics.roc_auc_score(y_true=all_targets, y_score=all_predictions)
        print("auc{}".format(auc))
        roc_plot(labels=all_targets, predictions=all_predictions, positive_label=1, save_dir=args.result_dir)

    return


if __name__ == "__main__":
    main()


"""
--database-root /hdd/hanoch/runmodels/img_quality/results/inference_production/holdout_soft_pred_new_model --dataset-split-csv /hdd/hanoch/data/objects-data-bbox-20191106-simple-sharded-part/tile_data/file_quality_tile_eileen_good_bad_val_bad_9_20_avg_pool_filt_conf_filt_no_edges_trn_tst_fitjar.csv
nohup python -u ./src/CNN/utillities/hand_crafted_poc.py --database-root /hdd/hanoch/runmodels/img_quality/results/inference_production/holdout_soft_pred_new_model --dataset-split-csv /hdd/hanoch/data/objects-data-bbox-20191106-simple-sharded-part/tile_data/file_quality_tile_eileen_good_bad_val_bad_9_20_avg_pool_filt_conf_filt_no_edges_trn_tst_fitjar.csv& tail -f nohup.out
"""