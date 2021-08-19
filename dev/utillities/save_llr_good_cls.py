import pickle
import pandas as pd
import numpy as np
import os
path_data = '/hdd/annotator_uploads'

import subprocess

# run inference and when getting the conf/labels out then continue :

df = test_df[all_targets==1]
det_good_cls = all_predictions[:, class_labels['good']][all_targets==1]
k = np.argsort(det_good_cls)
sr = np.sort(det_good_cls)
df_low_conf_good_cls = df.iloc[k]

df_low_conf_good_cls["good_cls_prob"] = sr
df_low_conf_good_cls["cage_id"] = ""
target_path = os.path.join(args.result_dir, 'low_conf_good_cls')
import subprocess
# copy low conf softmax
target_path = os.path.join(args.result_dir, 'low_conf_good_cls')
for idx, row in df_low_conf_good_cls.iterrows():
    fname = row.full_file_name.split('/')[-1].split('.png')[0]
    file_full_path = subprocess.getoutput('find ' + path_data + ' -iname ' + '"*' + fname + '.png' + '*"')
    cages_id = file_full_path.split('/hdd/annotator_uploads')[1].split('/')[1]
    df_low_conf_good_cls.loc[idx, 'cage_id'] = cages_id
    if row.good_cls_prob < 0.05:
        full_file_name = row.full_file_name
        dest_file = full_file_name.split('.png')[0].split('/')[-1] + '_tile' + row.file_name.split('tile_')[-1] + '_llr_' + str(row.good_cls_prob.__format__('.3e')) + '.png'
        ans2 = subprocess.getoutput('cp -p ' + row.full_file_name + ' ' + ' ' + target_path + '/' + dest_file)

    else:
        continue


# copy high conf softmax
target_path = os.path.join(args.result_dir, 'high_conf_good_cls')
for idx, row in df_low_conf_good_cls.iterrows():
    if row.good_cls_prob > 0.2:
        full_file_name = row.full_file_name
        dest_file = full_file_name.split('.png')[0].split('/')[-1] + '_tile' + row.file_name.split('tile_')[-1] + '_llr_' + str(row.good_cls_prob.__format__('.3e')) + '.png'
        ans2 = subprocess.getoutput('cp -p ' + row.full_file_name + ' ' + ' ' + target_path + '/' + dest_file)
    else:
        continue

df_bad = test_df[all_targets==0]
det_bad_cls = all_predictions[:, class_labels['bad']][all_targets==0] # llr of bad class that should be classified as bad
k = np.argsort(det_bad_cls,)
sr = np.sort(det_bad_cls)
df_high_conf_bad_cls = df_bad.iloc[k]

df_high_conf_bad_cls["bad_cls_prob"] = sr

target_path = os.path.join(args.result_dir, 'high_conf_bad_cls')
for idx, row in df_high_conf_bad_cls.iterrows():
    fname = row.full_file_name.split('/')[-1].split('.png')[0]
    file_full_path = subprocess.getoutput('find ' + path_data + ' -iname ' + '"*' + fname + '.png' + '*"')
    cages_id = file_full_path.split('/hdd/annotator_uploads')[1].split('/')[1]
    df_high_conf_bad_cls.loc[idx, 'cage_id'] = cages_id

    if row.bad_cls_prob < 0.05:
        full_file_name = row.full_file_name
        dest_file = full_file_name.split('.png')[0].split('/')[-1] + '_tile' + row.file_name.split('tile_')[-1] + '_llr_' + str(row.bad_cls_prob.__format__('.3e')) + '.png'
        ans2 = subprocess.getoutput('cp -p ' + row.full_file_name + ' ' + ' ' + target_path + '/' + dest_file)
    else:
        continue


cage_model_res = {'all_predictions' :all_predictions, 'all_targets': all_targets, 'model': args.model_path,
                  'df_low_conf_good_cls': df_low_conf_good_cls, 'df_high_conf_bad_cls': df_high_conf_bad_cls}

with open(os.path.join(args.result_dir, 'model_res' + args.model_path.split('__')[-1]) + '.pkl', 'wb') as f:
    pickle.dump(cage_model_res, f)

df_high_conf_bad_cls.to_csv(os.path.join(args.result_dir, 'high_conf_bad_cls.csv'), index=False)

df_low_conf_good_cls.to_csv(os.path.join(args.result_dir, 'low_conf_good_cls.csv'), index=False)
