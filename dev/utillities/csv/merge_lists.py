import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import tqdm
import pandas as pd
import tqdm

path = '/hdd/hanoch/data/objects-data-bbox-20191106-simple-sharded-part/tile_data/temp_files'
major_path = 'annotations_fix_eileen8_with_fitjar_marginal_cages.csv'
path2 = 'holdout_holdout_marginal_cutout_from_blindfinder_tile.csv'
add_to_major = False

if not add_to_major:
    print("Secondary path supports only existed cutouts id hence new one are dropped")
#Merge list into major
df_eileen = pd.read_csv(os.path.join(path, path2), index_col=False)
df_major = pd.read_csv(os.path.join(path, major_path), index_col=False)
df_major = df_major.dropna(axis=0, subset=['file_name'])
print("df_major len {}", len(df_major))
for cu in tqdm.tqdm(df_eileen.cutout_uuid.unique()):
    if df_major[df_major.cutout_uuid == cu]['class'].size == 0:
        if add_to_major:
            df_major = df_major.append([df_eileen.loc[df_eileen.cutout_uuid == cu]])
            print("CU {} added to the major".format(cu))
        else:
            print("not needed filtered from ref")
            continue
    tr_tst = df_major[df_major.cutout_uuid==cu]['train_or_test'].iloc[0]
    df_eileen.loc[df_eileen.cutout_uuid==cu, 'train_or_test'] = tr_tst
    val = df_major[df_major.cutout_uuid==cu]['val'].iloc[0] #Correct way to set value on a sobjects in pandas [duplicate]
    df_eileen.loc[df_eileen.cutout_uuid==cu, 'val'] = val
    cls_major = df_major[df_major.cutout_uuid==cu]['class'].iloc[0]
    label_major = df_major[df_major.cutout_uuid==cu]['label'].iloc[0]
    if df_eileen[df_eileen.cutout_uuid == cu]['class'].size == 0: # that CU isn't found in original list hence no drop allowed
        cls_eileen = 'nogo'
        print(cu)
        print("skip!!no ext. info not match ref {} curr {} ".format(len(df_eileen[df_eileen.cutout_uuid==cu]), len(df_major[df_major.cutout_uuid==cu])))
        continue
    else:
        cls_eileen = df_eileen[df_eileen.cutout_uuid==cu]['class'].iloc[0]
    if cls_major != cls_eileen:
        print("class not match {} cls_major {} cls_eileen {} taking major".format(cu, cls_major, cls_eileen))
        df_eileen.loc[df_eileen.cutout_uuid == cu, 'class'] = cls_major
        df_eileen.loc[df_eileen.cutout_uuid == cu, 'label'] = label_major
    if len(df_eileen[df_eileen.cutout_uuid == cu]) != len(df_major[df_major.cutout_uuid==cu]):
        print("{} no tiles not match ref {} curr {} ".format(cu, len(df_eileen[df_eileen.cutout_uuid==cu]), len(df_major[df_major.cutout_uuid==cu])))
        if np.abs(len(df_eileen[df_eileen.cutout_uuid == cu]) - len(df_major[df_major.cutout_uuid==cu]))>10:
            print('delta >10!!!!!!!!!!!')
    if 0:
        ind_drop = df_major[df_major.cutout_uuid == cu].index
        df_major = df_major.drop(ind_drop)
    else:
        df_major = df_major[df_major.cutout_uuid != cu]
    # print("Append {} dropped {} delta = {}".format(len(df_eileen.loc[df_eileen.cutout_uuid == cu].index), len(ind_drop),
    #                                                len(df_eileen.loc[df_eileen.cutout_uuid == cu].index) - len(ind_drop)))
    df_major = df_major.append([df_eileen.loc[df_eileen.cutout_uuid == cu]])
    df_major['file_name'] = df_major.apply(lambda x: str(x.file_name).replace("unknown-tested", str(x['class'])),
                                           axis=1)

df_major.to_csv(
    r'C:\Users\hanoch.kremer\OneDrive - ALLFLEX EUROPE\HanochWorkSpace\Data\DataBaseInfo\tile_location\annotations_fix_eileen9_with_fitjar_marginal_cages_readd_holdout_meta.csv',
    index=False)


# ref has more valueable info i.e updated tiles no.
if 0 :
    major_path = r'C:\Users\hanoch.kremer\OneDrive - ALLFLEX EUROPE\HanochWorkSpace\Data\blind_quality_svm\tile_data\file_quality_tile_eileen_good_bad_val_bad_9_20_avg_pool_filt_conf_filt_no_edges_trn_tst_fitjar_marginal_reannot_marginals.csv'
    path2 = r'C:\Users\hanoch.kremer\OneDrive - ALLFLEX EUROPE\HanochWorkSpace\Data\DataBaseInfo\tile_location\merged_eileen_data.csv'
    df_eileen = pd.read_csv(path2, index_col=False)
    df_major = pd.read_csv(major_path, index_col=False)
    df_eileen.val = ""
    df_eileen.train_or_test = ""
    for cu in df_eileen.cutout_uuid.unique():
        tr_tst = df_major[df_major.cutout_uuid == cu]['train_or_test'].iloc[0]
        df_eileen.loc[df_eileen.cutout_uuid == cu, 'train_or_test'] = tr_tst
        val = df_major[df_major.cutout_uuid == cu]['val'].iloc[
            0]  # Correct way to set value on a sobjects in pandas [duplicate]
        df_eileen.loc[df_eileen.cutout_uuid == cu, 'val'] = val
        cls_major = df_major[df_major.cutout_uuid == cu]['class'].iloc[0]
        label_major = df_major[df_major.cutout_uuid == cu]['label'].iloc[0]
        cls_eileen = df_eileen[df_eileen.cutout_uuid == cu]['class'].iloc[0]
        if cls_major != cls_eileen:
            print("class not match {} cls_major {} cls_eileen {}".format(cu, cls_major, cls_eileen))
            df_eileen.loc[df_eileen.cutout_uuid == cu, 'class'] = cls_major
            df_eileen.loc[df_eileen.cutout_uuid == cu, 'label'] = label_major

        ind_drop = df_major[df_major.cutout_uuid == cu].index
        df_major = df_major.drop(ind_drop)
        df_major = df_major.append(df_eileen.loc[df_eileen.cutout_uuid == cu])

    df_major = df_major.dropna(axis=0, subset=['file_name'])
    df_major['file_name'] = df_major.apply(lambda x: str(x.file_name).replace("unknown-tested", str(x['class'])),
                                           axis=1)
    # df_major['file_name'] = df_major['file_name'].apply(lambda x:str(x).replace("unknown-tested", str(df_major['class'])))
    # for ind, row in df_major.iterrows():	row['file_name'].replace("unknown-tested", str(row['class']))
    df_major.to_csv(
        r'C:\Users\hanoch.kremer\OneDrive - ALLFLEX EUROPE\HanochWorkSpace\Data\DataBaseInfo\tile_location\annotations_fix_eileen.csv',
        index=False)
