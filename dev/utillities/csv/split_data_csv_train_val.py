
# split tot train/val
import os
import pandas as pd

path = '/hdd/hanoch/data/objects-data-bbox-20191106-simple-sharded-part/cutout_data/4_5cages_298ex'
file = 'data_tiles.csv'
df = pd.read_csv(os.path.join(path, file), index_col=False)
df.train_or_test = 'train'
df['val'] = 0

df_filtered_marginals_only = pd.DataFrame()

df = df.dropna(axis=0, subset=['file_name'])
cu_acm = list()
for ind, cu in enumerate(df.cutout_uuid.unique()):
    if df[df.cutout_uuid == cu]['class'].iloc[0] == 'marginal':
        df_filtered_marginals_only = df_filtered_marginals_only.append((df[df.cutout_uuid == cu]))
        cu_acm.append(cu)


train_th = int(len(cu_acm)*0.8)
for idx, co in enumerate(cu_acm):
    if idx < train_th:
        df_filtered_marginals_only.loc[(df_filtered_marginals_only['cutout_uuid'] == co), ['val']] = 0
    else:
        df_filtered_marginals_only.loc[(df_filtered_marginals_only['cutout_uuid'] == co), ['val']] = 1
dest_csv = os.path.join(path, 'val_train_split'+file)
df_filtered_marginals_only.to_csv(dest_csv, index=False)

import dev.utillities.csv.check_data_csv_integrity
dev.utillities.csv.check_data_csv_integrity.main(['--path-major', dest_csv, '--n_classes', '1'])
print('End ')

"""
path2 = '/hdd/hanoch/data/objects-data-bbox-20191106-simple-sharded-part/tile_data/annotations_quality_tile_pool_filt_conf_filt_no_edges_trn_tst_fitjar_marginal_reannot2_marginals_tile_pos.csv'
df_acm = pd.DataFrame()
df2 = pd.read_csv(path2, index_col=False)
df_acm = df_acm.append((df_filtered_marginals_only))
df_acm = df_acm.append((df2))
df_acm.to_csv('/hdd/hanoch/data/objects-data-bbox-20191106-simple-sharded-part/tile_data/annotations_quality_tile_pool_filt_conf_filt_no_edges_trn_tst_fitjar_marginal_reannot2_marginals_tile_pos_155marg.csv', index=False)
"""