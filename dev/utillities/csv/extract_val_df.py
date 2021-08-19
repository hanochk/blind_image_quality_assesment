import pandas as pd
import os
import numpy as np

# df_train_test = pd.DataFrame(
#     columns=['file_name', 'cutout_uuid', 'class', 'label', 'percent_40', 'percent_60', 'percent_80', 'train_or_test'])
train_test_pth = '/hdd/hanoch/data/objects-data-bbox-20191106-simple-sharded-part/tile_data/train_test_quality_merged_tile.csv'
df_train_test = pd.DataFrame(
    columns=['file_name'])

train_test_pth = '/hdd/hanoch/data/objects-data-bbox-20191106-simple-sharded-part/tile_data/train_test_quality_merged_tile.csv'
df = pd.read_csv(train_test_pth, index_col=False)
df['val'] = 0
path_input = '/hdd/hanoch/data/objects-data-bbox-20191106-simple-sharded-part/tile_data/val'
for root, _, filenames in os.walk(path_input):
    filenames.sort()
    for file in filenames:
        # df_train_test.loc[len(df_train_test)] = [fname_save, train_cut_uuid, clsss_quality[label], label, percent_40,
        #                                          percent_60, percent_80, train_or_test]
        df_train_test.loc[len(df_train_test)] = [file]
        ind = np.where(df['file_name'] == file.split('.')[0])
        if ind[0].size >0:
            df.loc[int(ind[0]), 'val'] = 1
            print(ind)
        # # ind = np.where(df['file_name'] == df.iloc[0]['file_name'])[0].item()
        # if file in df:
        #     print(file)

df_train_test.to_csv(os.path.join(path_input, 'val_tile.csv'), index=False)
df.to_csv(os.path.join(path_input, 'train_test_val_quality_merged_tile_tile.csv'), index=False)

