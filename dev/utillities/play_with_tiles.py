import os
import pickle
import datetime
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import csv
import cv2
from collections import defaultdict
import sklearn.metrics
from sklearn.metrics import accuracy_score
import pandas as pd
from src.CNN.inference import p_r_plot, roc_plot

all_train_test_csv = True

if all_train_test_csv:
    tiles_folder = '/hdd/hanoch/data/objects-data-bbox-20191106-simple-sharded-part/tile_data'
    # csv_annotations = '/hdd/hanoch/data/objects-data-bbox-20191106-simple-sharded-part/tile_data/file_quality_tile_eileen_good_bad_val_bad_9_20_avg_pool_filt_conf_filt_no_edges_trn_tst_fitjar_hcf.csv'
    csv_annotations = '/hdd/hanoch/data/objects-data-bbox-20191106-simple-sharded-part/tile_data/merged_full_path.csv'
    file_name_field = 'full_file_name'
    ext = ''

else:
    tiles_folder = '/hdd/hanoch/data/cages_holdout/annotated_images_eileen/tile_data/test'
    csv_annotations = '/hdd/hanoch/data/objects-data-bbox-20191106-simple-sharded-part/tile_data/Cages_n5_no_marginals_tiles_eileen_blind_test_tiles_ntiles.csv'
    file_name_field = 'file_name'
    ext = '.png'


if 1:
    d = dict()
    with open(csv_annotations, newline='') as f:
        a = csv.reader(f)
        for ii,l in enumerate(a):
            if ii==0:
                file_index = l.index(file_name_field)
                l1 = l
                continue
            d[l[file_index]] = {a:b for a,b in zip(l1[1:],l[1:])}
    list_of_files = list(d.keys())

else:
    dataframe = pd.read_csv(csv_annotations, index_col=False)
    if all_train_test_csv:
        if 1:
            val_df = None
            train_df = dataframe.loc[dataframe['train_or_test'] == 'train']
            test_df = dataframe.loc[dataframe['train_or_test'] == 'test']
            if len(train_df) >0 :
                val_df = train_df[train_df['val'] == 1]
                train_df = train_df[train_df['val'] != 1]
            if len(train_df) >0 :
                train_df['full_file_name'] = train_df.apply(
                    lambda x: os.path.join(tiles_folder, 'train', x['class'], x['file_name'] + '.png'), axis=1)
                # Sanity check
                assert train_df['full_file_name'].apply(
                    lambda x: os.path.isfile(x)).all(), "Some images referenced in the CSV file were not found"

            if val_df is not None :
                val_df['full_file_name'] = val_df.apply(
                    lambda x: os.path.join(tiles_folder, 'val', x['class'], x['file_name'] + '.png'), axis=1)

                assert val_df['full_file_name'].apply(
                    lambda x: os.path.isfile(x)).all(), "Some images referenced in the CSV file were not found"

            if len(test_df) >0 :
                test_df['full_file_name'] = test_df.apply(
                    lambda x: os.path.join(tiles_folder, 'test', x['class'], x['file_name'] + '.png'), axis=1)
                assert test_df['full_file_name'].apply(
                    lambda x: os.path.isfile(x)).all(), "Some images referenced in the CSV file were not found"
            df_acm = pd.DataFrame()
            df_acm = df_acm.append((train_df))
            df_acm = df_acm.append((val_df))
            df_acm = df_acm.append((test_df))
            ols_to_keep = ['full_file_name', 'label', 'media_uuid']
            df_acm.loc[: , ols_to_keep].to_csv(os.path.join(os.path.split(csv_annotations)[0],
                                                                'merged_full_path' + '.csv'), index=False)


        list_of_files = dataframe['full_file_name'].to_list()
        d = dataframe
        ext = ''
    else:
        list_of_files = dataframe['file_name'].to_list()
        ext = '.png'

def vollath(gimg , k):
    gimg = gimg/np.max(gimg)
    return np.sum(img[:,:-2*k]*(img[:,k:-k] - img[:,2*k:]))


flag_show = False
MOD_K = 50
canny_upper_th = 80
lower_th = 50
scores = defaultdict(lambda: defaultdict(list))
all_scores_per_image = defaultdict(lambda: {'scores': [], 'label': ''})
df_stat = pd.DataFrame(columns=['file', 'mean_gray', 'std_gray', 'class', 'canny_50_70'])
for ii, file in enumerate(list_of_files):
    if not np.mod(ii, MOD_K):
        file_with_path = os.path.join(tiles_folder, file + ext)
        img = mpimg.imread(file_with_path)
        gimg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gimg = gimg ** 0.5
        # edges = cv2.Canny(np.uint8(gimg * 255), 17, 70)   # HK
        edges = cv2.Canny(np.uint8(gimg * 255), lower_th, canny_upper_th)
        edges_score = np.sum(edges)/256/100

        df_stat.loc[len(df_stat)] = [file, gimg.mean(), gimg.std(), d[file]['label'], edges_score]

        vollath_score = vollath(gimg, 2)
        if isinstance(d, dict):
            all_scores_per_image[d[file]['media_uuid']]['label'] = d[file]['label']
        else:
            all_scores_per_image[d[file]['media_uuid']]['label'] = d[d['full_file_name'] == file]['label'].item()

        all_scores_per_image[d[file]['media_uuid']]['scores'].append(edges_score)

        for ii in np.arange(1, 10, 1):
            vollath_score = vollath(gimg, ii)
            scores['vollath k=' + str(ii)][d[file]['label']].append(vollath_score)
        for ii in np.arange(lower_th-9, lower_th, 1):
            jj = canny_upper_th
            edges = cv2.Canny(np.uint8(gimg * 255), ii, jj)
            edges_score = np.sum(edges) / 256 / 100
            scores['edges (' + str(ii) + ',' + str(jj) + ')'][d[file]['label']].append(edges_score)
        if flag_show:
            fig, axs = plt.subplots(1, 2, sharey=True, gridspec_kw={'wspace': 0})
            axs[0].imshow(gimg, cmap='gray', vmin=0, vmax=1)

            axs[1].imshow(edges, cmap='gray', vmin=0, vmax=1)
            fig.suptitle('score: ' + d[file] + ' vollath:' + str(vollath_score))
            plt.show()

df_stat.to_csv(os.path.join(os.path.split(csv_annotations)[0],
                            os.path.basename(csv_annotations).split('.csv')[0] + '_upper_th' +str(canny_upper_th) + '_canny_stat.csv'), index=False)



def plot_hist_of_all_keys(in_dict):
    nfigs = len(list(in_dict.keys()))
    N1 = np.int(np.ceil(np.sqrt(nfigs)))
    if N1*(N1-1) >= nfigs:
        N2 = N1-1
    else:
        N2 = N1
    fig, axs = plt.subplots(N2,N1)
    fig2, axs2 = plt.subplots()
    axs = np.reshape(axs, (-1,))

    fig.set_size_inches(10, 10)
    fig.set_dpi(100)
    fig2.set_size_inches(10, 10)
    fig2.set_dpi(100)

    for ind,(k,v) in enumerate(in_dict.items()):
        k2 = list(v.keys())
        v2 = list(v.values())

        x = np.sort(np.concatenate(v2))
        bins = np.linspace(x[np.int(np.floor(len(x) * 0))],
                           x[np.int(np.floor(len(x) - 1))],
                           num=np.int(np.round(np.sqrt(len(x)))))
        print(k)
        labels = list()
        all_predictions = list()
        for ii in range(len(v2)):
            x, y = np.histogram(np.array(v2[ii]), bins=bins, density=False)
            if k2[ii] == '3':
                tpr = np.cumsum(x) / np.sum(x)
                labels += [np.ones_like(v2[ii])]
            else:
                fpr = np.cumsum(x) / np.sum(x) # actually the TNR
                labels += [np.zeros_like(v2[ii])]
            all_predictions.append(v2[ii])

            axs[ind].plot(y[:-1], x, label=k2[ii])
        predictions = np.concatenate(all_predictions)
        labels = np.concatenate(labels)
        ap = sklearn.metrics.average_precision_score(labels, predictions, pos_label=1)
        auc = sklearn.metrics.roc_auc_score(labels, predictions)
        print("Feature {} index {} AP={} AUC {}".format(k, ind, ap, auc))
        axs2.plot(tpr, fpr, label=k)

        axs[ind].title.set_text(k)
        axs[ind].legend()
    axs2.legend()
    fig.show()
    fig2.show()

scores['Entire Image egs(' + str(lower_th) + ',' + str(canny_upper_th) + ')']['3'] = [np.mean(b['scores']) for b in list(all_scores_per_image.values()) if b['label'] == '3']
scores['Entire Image egs(' + str(lower_th) + ',' + str(canny_upper_th) + ')']['1'] = [np.mean(b['scores']) for b in list(all_scores_per_image.values()) if b['label'] == '1']


for ii in np.arange(60,100,5):
    scores['voting>' + str(ii)]['3'] = [np.mean(np.array(b['scores'])>ii) for b in list(all_scores_per_image.values()) if b['label'] == '3']
    scores['voting>' + str(ii)]['1'] = [np.mean(np.array(b['scores'])>ii) for b in list(all_scores_per_image.values()) if b['label'] == '1']


plot_hist_of_all_keys(scores)

#
# xg = [np.mean(b['scores']) for b in list(all_scores_per_image.values()) if b['label'] == '3']
# xb = [np.mean(b['scores']) for b in list(all_scores_per_image.values()) if b['label'] == '1']
#
# x = np.sort(np.concatenate((xg,xb)))
# bins = np.linspace(x[np.int(np.floor(len(x) * 0))],
#                    x[np.int(np.floor(len(x) - 1))],
#                    num=20)
#
# x, y = np.histogram(np.array(xg), bins=bins, density=False)
# plt.plot(y[:-1], x, label='3')
# x, y = np.histogram(np.array(xb), bins=bins, density=False)
# plt.plot(y[:-1], x, label='1')
#
# plt.show()

"""
path = '/hdd/hanoch/data/objects-data-bbox-20191106-simple-sharded-part/tile_data'
file = 'file_quality_tile_eileen_good_bad_val_bad_9_20_avg_pool_filt_conf_filt_no_edges_trn_tst_fitjar_hcf.csv'
df = pd.read_csv()
df = pd.read_csv(os.path.join(path, file), index_col=False)
df.keys()
df.label.unique()
g_val = df[df.label==3]['canny_edges_sum_50_70']
len(g_val)
b_val = df[df.label==1]['canny_edges_sum_50_70']
len(b_val)
type(b_val)
b_val = np.array(df[df.label==1]['canny_edges_sum_50_70'])
b_val.shape
g_val = np.array(df[df.label==3]['canny_edges_sum_50_70'])
g_val.shape
gt = [np.ones_like(g_val), np.zeros_like(b_val)]
gt = np.array(np.ones_like(g_val), np.zeros_like(b_val))
gt = np.concatenate((np.ones_like(g_val), np.zeros_like(b_val)))
pred = np.concatenate((g_val, b_val))
precision, recall, thresholds = sklearn.metrics.precision_recall_curve(gt, pred,
                                                                       pos_label=1)

ap = sklearn.metrics.average_precision_score(gt, pred,
                                             pos_label=1)

"""