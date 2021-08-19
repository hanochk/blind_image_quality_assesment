import pandas as pd
import tqdm
import os
from dev.utillities.file import load_csv_xls_2_df

path2 = r'C:\Users\hanoch.kremer\OneDrive - ALLFLEX EUROPE\HanochWorkSpace\Data\blind_quality_svm\tile_data\marginals_reannotate\marginal_to_annotate 5.18.21.xlsx'
major_path = r'C:\Users\hanoch.kremer\OneDrive - ALLFLEX EUROPE\HanochWorkSpace\Data\DataBaseInfo\tile_location\annotations_fix_eileen9_with_fitjar_marginal_cages_readd_holdout_meta.csv'
path_to_save = os.getcwd()
quality_to_labels_annotator_version = {'poor': 1, 'marginal': 2, 'excellent': 3}
clsss_quality_label_2_str = {1: 'bad', 2: 'marginal', 3: 'good', 4: 'unknown-tested'}


df_eileen = load_csv_xls_2_df(path2, index_col=False)
df_major = pd.read_csv(major_path, index_col=False)
df_major = df_major.dropna(axis=0, subset=['file_name'])
df_eileen = df_eileen.dropna(axis=0, subset=['merge'])
print("df_major len", len(df_major))
for cu in tqdm.tqdm(df_eileen.cutout_uuid.unique()):
    if isinstance(df_eileen[df_eileen.cutout_uuid == cu]["merge"].item(), str):
        label_from_csv = df_eileen[df_eileen.cutout_uuid == cu]["merge"].item().lower()
        label = quality_to_labels_annotator_version.get(label_from_csv, None)
        if label is None:
            print("CU {} the annotated label {} is not one of the valid close set".format(cu, label_from_csv))
            continue

        class_ = clsss_quality_label_2_str[label]
        if df_major[df_major.cutout_uuid == cu]['class'].iloc[0].lower() != class_:
            print("CU {} annotated class : {} is different than prime : {} csv ->override".format(cu, class_, df_major[df_major.cutout_uuid == cu]['class'].iloc[0]))
            df_major.loc[df_major.cutout_uuid == cu, 'class'] = class_
            df_major.loc[df_major.cutout_uuid == cu, 'label'] = label
    else:
        print("CU label {} is not s string".format(df_eileen[df_eileen.cutout_uuid == cu]["merge"].item()))
df_major.to_csv(os.path.join(path_to_save, 'merged_with_annotators.csv'), index=False)
