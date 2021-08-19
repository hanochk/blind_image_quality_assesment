import pandas as pd
import os
import numpy as np
import subprocess




# path_input = '/hdd/hanoch/data/cages_holdout/annotated_images_eileen_fitjar/res/csv/csv_blind'
path_input = '/hdd/hanoch/data/objects-data-bbox-20191106-simple-sharded-part/cutout_data/fitjar/res'
annot_file = 'list_ready_cage_fitjar_blind_test_results.xlsx'
result_dir = os.path.join(path_input, 'output')
df_tbl = pd.DataFrame()
skip_marginal_image = False
comment = 'fitjar'
label_2_clsss_quality = {'bad': 1, 'marginal': 2, 'good': 3}

filenames = [os.path.join(path_input, x) for x in os.listdir(path_input)
             if x.endswith('csv')]

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

annot_df = load_csv_xls_2_df(os.path.join(path_input, annot_file))
annot_df['class'] = annot_df['Eileen'].apply(lambda x: x.lower())
if skip_marginal_image:
    print('Marginal images are omitted due to in tile uncertianty')
    annot_df['label'] = annot_df['Eileen'].apply(lambda x: 3 if x.lower() == 'good' else 1)
else:
    annot_df['label'] = annot_df['Eileen'].apply(lambda x: label_2_clsss_quality[x.lower()])
i = 0
len_tbl = 0
for file in filenames:
    if file.endswith(".csv"):

        df_image = pd.read_csv(file, index_col=False)
        fname = df_image.cutout_uuid.iloc[0]
        # if fname == 'blind_20200818T080111.772086Z_1.png': #'blind_20200815T102414.004297Z_0.png':
        #     print('ka')
        df_image.cutout_uuid = df_image['file_name'].apply(lambda x: x.split('_blind')[0])
        df_image['train_or_test'] = 'train'
        df_image['val'] = 0
        df_image['comment'] = comment
        df_image['label'] = ""
        df_image['class'] = ""
        df_image['label'] = annot_df[annot_df.cutout_id == df_image.cutout_uuid.iloc[0]]['label'].item()
        df_image['class'] = annot_df[annot_df.cutout_id == df_image.cutout_uuid.iloc[0]]['class'].item()
        if skip_marginal_image:
            if df_image['class'].iloc[0] == 'marginal':
                print("Marginal file omitted {}".format(file))
                continue
            else: #may append to final list
                if 1:
                    df_image['file_name'] = df_image.apply(lambda x: x.file_name.replace("bad", x['class']), axis=1)

                if 0:
                    target_path = '/hdd/hanoch/data/cages_holdout/annotated_images_eileen_fitjar/res/png'
                    curr_path = '/hdd/hanoch/data/cages_holdout/annotated_images_eileen_fitjar/res'
                    for idx, row in df_image.iterrows():
                        filename = row.file_name + '.png'
                        dest_file = filename.replace("bad", df_image['class'].iloc[0])
                        full_file_name = os.path.join(curr_path, filename)
                        ans2 = subprocess.getoutput('cp -p ' + full_file_name + ' ' + ' ' + target_path + '/' + dest_file)

                i += 1
                len_tbl += len(df_tbl)
                df_tbl = df_tbl.append([df_image])
                print("{} File {} Added".format(i, file))
        else:
            i += 1
            len_tbl += len(df_tbl)
            print("Total lenth of records so far {} ".format(len(df_tbl)))
            print("{} File {} Added".format(i, file))
            df_tbl = df_tbl.append([df_image])

# df_tbl.columns = ['images_quality', 'crop_uuid']
print(len(df_tbl))
df_tbl.to_csv(os.path.join(result_dir, 'merged_fitjar_csv.csv'), index=False)
