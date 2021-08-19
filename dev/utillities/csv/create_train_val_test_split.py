import pandas as pd
import os
import tqdm
from sklearn import model_selection
from dev.test_blind_quality import quality_to_labels_annotator_version, clsss_quality_label_2_str
import matplotlib.pyplot as plt
import numpy as np
# append all per image CSV
if 0:
    path = '/hdd/hanoch/data/database/blind_quality/quality_based_all_annotations_24_6/png/tiles/csv_list'
    path_to_save = '/hdd/hanoch/data/database/blind_quality/quality_based_all_annotations_24_6/png/tiles/csv/data_list.csv'
    filenames = [os.path.join(path, x) for x in os.listdir(path)
                 if x.endswith('csv')]
    df_acm = pd.DataFrame()
    for idx, file in enumerate(tqdm.tqdm(filenames)):
        df = pd.read_csv(file, index_col=False)
        df_acm = df_acm.append((df))

    df_acm.to_csv(path_to_save, index=False)


    print(len(df_acm.cutout_uuid.unique()))
else:
    path = '/hdd/hanoch/data/database/blind_quality/quality_based_all_annotations_24_6/png/tiles/csv'
    data_file = 'data_list.csv'
    path_metadata = '/hdd/hanoch/data/database/blind_quality/quality_based_all_annotations_24_6'
    meta_file = 'quality_based_all_annotations_24_6_CUmeta.csv'
    annotators_file = 'quality_based_all_annotations_24_6_CUannotator_hist.csv'
    test_set_penn_dependant = True
    label_concensus = True
    prev_label = ''

    if label_concensus:
        df_annot = pd.read_csv(os.path.join(path, annotators_file), index_col=False)
        df_annot['cutout_uuid'] = df_annot['cutout_id']
        df_annot = df_annot.drop(['cutout_id'], axis=1)
        df_annot.concencus_vote = ""
        df_annot.label_concencus = ""
        for ind, row in df_annot.iterrows():
            row_ = row.drop('cutout_uuid')
            #row_[pd.notna(row_)]
            # print(row_[pd.notna(row_)])
            vote = row_.mode().to_list()
            if len(vote) >1: # no majority then pick eileen  = eb
                vote = row_.eb
            else:
                vote = vote[0]
            df_annot.loc[ind, 'concencus_vote'] = vote
            df_annot.loc[ind, 'label_concencus'] = quality_to_labels_annotator_version[vote]
            # print(row_[pd.notna(row_)], row_.mode().to_list())

    df_ = pd.read_csv(os.path.join(path, data_file), index_col=False)
    if 1:
        df_input = pd.read_csv(os.path.join(path, 'post_proc_' + data_file), index_col=False)
    else:
    # df_input is in image level
        df_input = pd.DataFrame()
        df_input['cutout_uuid'] = df_['cutout_uuid'].unique()
        df_input["label"] = df_input.apply(lambda x:  df_[df_['cutout_uuid']==x.cutout_uuid].label.iloc[0], axis=1)
    # df_input.to_csv(os.path.join(path, 'post_proc_' + data_file), index=False)
    if label_concensus:
        prev_label = 'label_concensus_'
        df_input_annot = pd.merge(df_input, df_annot[['cutout_uuid', 'label_concencus']], how='inner', on='cutout_uuid')
        print("the original label will be replaced by the concencus")
        h = np.array(df_input_annot[['label']]) - np.array(df_input_annot[['label_concencus']])
        a_b = np.histogram(np.array(h).astype('int'), bins=[-2.5, -1.9, -0.9, 0.3, 1.2, 2.2, 2.7])
        bins_loc_b = (a_b[1][0:-1] + a_b[1][1:]) / 2
        fig, ax = plt.subplots()
        # ax.semilogy(bins_loc_b, a_b[0] / sum(a_b[0]))
        ax.bar(np.round(bins_loc_b), 100 * (a_b[0] / sum(a_b[0])), width=0.05)
        plt.xlabel('Average of annotators - supervisor decision')
        plt.ylabel('Percentage')
        plt.grid()
        plt.show()
        # Override the label with concencus label
        df_input = df_input_annot
        df_input['label'] = df_input['label_concencus']
        df_input = df_input.drop('label_concencus', axis=1)



    save_str = ''
    if test_set_penn_dependant:
        save_str = 'test_set_penn_dependant_' + prev_label
    #merge with pen location data
        df_meta = pd.read_csv(os.path.join(path_metadata, meta_file), index_col=False)
        df_meta['cutout_uuid'] = df_meta['cutout_id']
        df_meta = df_meta[["cutout_uuid", 'pen']]
        df_input = pd.merge(df_input, df_meta,  how='inner', on='cutout_uuid')
        print([(pen, len(df_input[df_input.pen==pen])) for pen in df_input.pen.unique()])
        df_test = df_input[df_input["pen"] == 'Eidesvik']
        print("Len of test in full images", len(df_test))
        df_input_no_test = pd.concat([df_input, df_test]).drop_duplicates(keep=False) # renmove df_test out of total THEN SPLIT REMAAINDER TO VAL/TRAIN ONLY
        # frac_test = 0
    #   SPLIT REMAAINDER TO VAL/TRAIN ONLY
        frac_val = 0.1*(1-len(df_test)/(len(df_test) + len(df_input_no_test))) # 0.1 related to whole data

        random_state = None
        stratify_colname = 'label'
        X = df_input_no_test  # Contains all columns.
        y = df_input_no_test[[stratify_colname]]  # Dataframe of just the column on which to stratify.

        # Split original dataframe into train and temp dataframes.
        df_train, df_temp, y_train, y_temp = model_selection.train_test_split(X,
                                                                              y,
                                                                              stratify=y,
                                                                              test_size=frac_val,
                                                                              random_state=random_state)
        df_val = df_temp
    else:
        frac_test = 0.1
        # stratification over class
        frac_val = 0.1

        frac_train = 1 - (frac_val + frac_test)
        random_state = None
        stratify_colname = 'label'
        X = df_input  # Contains all columns.
        y = df_input[[stratify_colname]]  # Dataframe of just the column on which to stratify.

        # Split original dataframe into train and temp dataframes.
        df_train, df_temp, y_train, y_temp = model_selection.train_test_split(X,
                                                                              y,
                                                                              stratify=y,
                                                                              test_size=(1.0 - frac_train),
                                                                              random_state=random_state)
        # Split the temp dataframe into val and test dataframes.
        relative_frac_test = frac_test / (frac_val + frac_test)
        df_val, df_test, y_val, y_test = model_selection.train_test_split(df_temp,
                                                          y_temp,
                                                          stratify=y_temp,
                                                          test_size=relative_frac_test,
                                                          random_state=random_state)


    assert len(df_input) == len(df_train) + len(df_val) + len(df_test)

    df_train['val'] = 0
    df_train['train_or_test'] = 'train'

    df_val['val'] = 1
    df_val['train_or_test'] = 'train'

    df_test['val'] = 0
    df_test['train_or_test'] = 'test'

    df_acm = pd.DataFrame()
    df_acm = df_acm.append((df_train))
    df_acm = df_acm.append((df_val))
    df_acm = df_acm.append((df_test))

    for ind, row in df_acm.iterrows():
        df_.loc[df_['cutout_uuid'] == row.cutout_uuid, 'train_or_test'] = row.train_or_test
        df_.loc[df_['cutout_uuid'] == row.cutout_uuid, 'val'] = row.val
        if label_concensus: # need to override the concencus label
            df_.loc[df_['cutout_uuid'] == row.cutout_uuid, 'label'] = row.label
            df_.loc[df_['cutout_uuid'] == row.cutout_uuid, 'class'] = clsss_quality_label_2_str[row.label]

    assert len(df_acm) == len(df_input)
    df_.to_csv(os.path.join(path, 'stratified_train_val_test_' + save_str + data_file), index=False)
    from dev.utillities.csv import check_data_csv_integrity
    check_data_csv_integrity.main(['--path-major', os.path.join(path, 'stratified_train_val_test_' + save_str + data_file), '--n_classes', '3'])

# concencus in terms of voting. if there are only two then Eileen takes
