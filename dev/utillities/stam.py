import pandas as pd
import os
import subprocess
import numpy as np



path = '/hdd/hanoch/data/objects-data-bbox-20191106-simple-sharded-part/tile_data/cages_marginal/tiles/csv'
path_cutout = '/hdd/hanoch/data/objects-data-bbox-20191106-simple-sharded-part/tile_data/cages_marginal'
file = 'merged_eileen_blind_tests_marginals.csv'
target_path = '/hdd/hanoch/data/objects-data-bbox-20191106-simple-sharded-part/tile_data/cages_marginal/tiles_fixed_names'
path_data = '/hdd/hanoch/data/objects-data-bbox-20191106-simple-sharded-part/tile_data/cages_marginal/tiles'

df_acm = pd.read_csv(os.path.join(path, 'merge.csv'), index_col=False)
df = pd.read_csv(os.path.join(path_cutout, file), index_col=False)

df_acm['fname'] = ""
df_acm['fname'] = df_acm.cutout_uuid
df_acm['cutout_uuid'] = ""



if 0:
    df_merge = pd.merge(df, df_acm, how='outer', on='fname')
    df_merge_2 = df_merge[['cutout_id', 'file_name_x', 'cutout_uuid',  'class',  'label',  'percent_40',  'percent_60',  'percent_80',  'in_outline_tiles_id', 'train_or_test',
         'media_uuid',
         'all_image_weighted_hue',
         'n_tiles',
         'fname']]

    df_merge_2 = df_merge_2.drop(columns=['cutout_uuid'])
    df_merge_2['cutout_uuid'] = ""
    df_merge_2['cutout_uuid'] = df_merge_2['cutout_id']
    df_merge_2 = df_merge_2.drop(columns=['cutout_id'])

    df_merge_2 = df_merge_2.drop(columns=['class'])
    df_merge_2['class'] = ""
    df_merge_2['class'] = df_merge.Eileen
    df_merge_2['label'] = 2
else:
    df_merge_2 = df_acm
    df_merge_2['file_name_dest'] = ""

for fname in df_acm['fname'].unique():
    ind = np.where(df.fname == fname)[0].item()
    cutout_id = df.cutout_id.iloc[ind]
    for inx in np.where(df_acm['fname'] == fname)[0]:
    # df_acm[df_acm['fname'] == fname]['cutout_uuid'] = cutout_id
        df_acm.loc[inx, 'cutout_uuid']= cutout_id

df_acm['class'] = ""
df_acm['class'] = df.Eileen
df_acm['label'] = 2
del df_acm['file_name_dest']
df_acm['media_uuid'] = df_acm['cutout_uuid']
df_acm.to_csv(os.path.join('/hdd/hanoch/data/objects-data-bbox-20191106-simple-sharded-part/tile_data', 'cages_marginal.csv'), index=False)
if 0:
    # df_merge_2.to_csv(os.path.join('/hdd/hanoch/data/objects-data-bbox-20191106-simple-sharded-part/tile_data', 'cages_marginal.csv'), index=False)
    for index, row in df_merge_2.iterrows():
        co = row.fname
        # print(co)
        ind = np.where(df.fname == row.fname)[0].item()
        cutout_id = df.cutout_id.iloc[ind]
        df_merge_2.cutout_uuid.loc[index] = cutout_id
        # co_real = row.cutout_id
        file_full_pathes = subprocess.getoutput('find ' + path_data + ' -iname ' + '"*' + co + '*"')
        if '\n' in file_full_pathes:
            file_full_path_es_tiles = file_full_pathes.split('\n')
            for file_full_path1 in file_full_path_es_tiles:
                # print(file_full_path1)
                tile_no = file_full_path1.split('_tile_')[-1].split('.png')[0]
                dest_file = cutout_id + '_' + row.fname.split('.png')[0] + '_' + tile_no + '.png'
                df_merge_2['file_name_dest'].loc[index] = cutout_id + '_' + row.fname.split('.png')[0] + '_' + tile_no
                # print(tile_no)
                ans2 = subprocess.getoutput('cp -p ' + file_full_path1 + ' ' + ' ' + target_path + '/' + dest_file)


df_tbl['val'] = 0
for idx, co in enumerate(df.cutout_uuid.unique()):
     if idx < 65:
         df_tbl.loc[(df_tbl['cutout_uuid'] == co), ['val']]  = 0
     else:
         df_tbl.loc[(df_tbl['cutout_uuid'] == co), ['val']]  = 1

path2 = '/hdd/hanoch/data/objects-data-bbox-20191106-simple-sharded-part/tile_data'
file2 = 'file_quality_tile_eileen_good_bad_val_bad_9_20_avg_pool_filt_conf_filt_no_edges_trn_tst_fitjar_hcf_marginal_ext.csv'
path_data = '/hdd/hanoch/data/objects-data-bbox-20191106-simple-sharded-part/cutout_data'
df = pd.read_csv(os.path.join(path2, file2), index_col=False)
fname_empty = list()
for cu in df.cutout_uuid.unique():
    fname = cu
    file_full_path = subprocess.getoutput('find ' + path_data + ' -iname ' + '"*' + fname + '*"')
    if file_full_path == '':
        fname_empty.append(fname)
        # print(fname)
