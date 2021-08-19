import pandas as pd
import tqdm
from argparse import ArgumentParser
import matplotlib.pyplot as plt
# this endpoint is for comparing two data csv when adding more data or attributes per record/entry
# To create data csv run create_tile_db.py in blind_quality_svm repo, for training and holdout set (hold out has 2 flavours w/ and w/o mariginal).
#then run Eileen private collection with test_blind_quality.py (with calculate_metadata_only switch) than merge all the csv per image to final data csv with the following data collections
# data/objects-data-bbox-20191106-simple-sharded-part/cutout_data/EileenQuality_needtoRunblindFinder_4_outline/all_cutouts
# nohup python -u src/CNN/test_blind_quality.py --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/clip_grad_new_norm/sgd_filt_list/saved_state_mobilenet_v2_256_win_n_lyrs___1606226252 --database-root /hdd/hanoch/data/cages_holdout/annotated_images_eileen_fitjar --result-dir /hdd/hanoch/data/cages_holdout/annotated_images_eileen_fitjar/res --outline-pickle-path /hdd/hanoch/data/cages_holdout/omri_outfulljoin7_2020_fname_keys.pkl --result-pickle-file inference_best_qual.pkl --gpu-id=3 --confidence-threshold 0.1 --image-fusion-voting-th 0.7  --image-fusion-voting-method concensus_and_vote --outline-file-format misc  --statistics-collect --process-folder-non-recursively > ./scripthueWeigthedChan.py.log </dev/null 2>&1 & tail -f scripthueWeigthedChan.py.log
# Futjar
#nohup python -u test_blind_quality.py mobilenet_v2_256_win_n_lyrs --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/clip_grad_new_norm/sgd_filt_list/saved_state_mobilenet_v2_256_win_n_lyrs___1606226252 --database-root /hdd/hanoch/data/objects-data-bbox-20191106-simple-sharded-part/cutout_data/fitjar --result-dir /hdd/hanoch/data/objects-data-bbox-20191106-simple-sharded-part/cutout_data/fitjar/res --outline-pickle-path /hdd/hanoch/data/objects-data-bbox-20191106-simple-sharded-part/cutout_data/merged_outlines.pkl --result-pickle-file inference_best_qual.pkl --gpu-id=3 --confidence-threshold 0.1 --image-fusion-voting-th 0.7  --image-fusion-voting-method concensus_and_vote --outline-file-format fname_n_cutout_id --statistics-collect --process-folder-non-recursively --calculate-metadata-only > ./scripthueWeigthedChan.py.log </dev/null 2>&1 & tail -f scripthueWeigthedChan.py.log

# For Fitjar run merge_test_blind_csvs_and_with_annotations.py over the csv to merge with the annotations list_ready_cage_fitjar_blind_test_results.xlsx
def main(args: list = None):

    parser = ArgumentParser()

    parser.add_argument("--path-major", type=str, required=True, metavar='PATH',
                        help="path to the csv/pkl files")

    parser.add_argument("--path-secondary", type=str, required=True, metavar='PATH',
                        help="path to the csv/pkl files")
    args = parser.parse_args(args)

    print("File major {}".format(args.path_major))
    print("File Minor {}".format(args.path_secondary))

    df_eileen = pd.read_csv(args.path_secondary, index_col=False)
    df_eileen = df_eileen.dropna(axis=0, subset=['file_name'])
    df_major = pd.read_csv(args.path_major, index_col=False)
    df_major = df_major.dropna(axis=0, subset=['file_name'])
    all_relative_to_major_percent = list()

    for cu in tqdm.tqdm(df_major.cutout_uuid.unique()):
        if df_eileen[df_eileen.cutout_uuid == cu]['class'].size == 0:
            print("CU from the major list {} isn't in the Minor no update skipped ".format(cu))
            continue

        if df_major[df_major.cutout_uuid == cu]['class'].size != df_eileen[df_eileen.cutout_uuid == cu]['class'].size:
            relative_to_major_percent = (df_major[df_major.cutout_uuid == cu]['class'].size/df_eileen[df_eileen.cutout_uuid == cu]['class'].size - 1)*100
            print("CU {} not identical entries Major {} vs. Minor {} relative_to_major_percent {}".format(cu,
                df_major[df_major.cutout_uuid == cu]['class'].size, df_eileen[df_eileen.cutout_uuid == cu]['class'].size, relative_to_major_percent))
            all_relative_to_major_percent.append(relative_to_major_percent)

        else:
            # comapre file names just in case the list are identicals otherwise of course there is a list
            sorted_major = sorted(df_major[df_major.cutout_uuid == cu].file_name)
            sorted_minor = sorted(df_eileen[df_eileen.cutout_uuid == cu].file_name)
            if sorted_major != sorted_minor:
                diff_list = set(sorted_major) - set(sorted_minor)
                print("CU {} file lists are not equal {}".format(cu, diff_list))

        cls_eileen = df_eileen[df_eileen.cutout_uuid == cu]['class'].iloc[0]
        cls_major = df_major[df_major.cutout_uuid == cu]['class'].iloc[0]
        if cls_major != cls_eileen:
            print("CU {} Class are different Major {} vs. Minor {} ".format(cu, cls_major, cls_eileen))

        label_major = df_major[df_major.cutout_uuid == cu]['label'].iloc[0]
        label_eileen = df_eileen[df_eileen.cutout_uuid == cu]['label'].iloc[0]
        if label_major != label_eileen:
            print("CU {} labels are different Major {} vs. Minor {} ".format(cu, label_major, label_eileen))

        train_or_test_major = df_major[df_major.cutout_uuid == cu]['train_or_test'].iloc[0]
        train_or_test_eileen = df_eileen[df_eileen.cutout_uuid == cu]['train_or_test'].iloc[0]
        if train_or_test_major != train_or_test_eileen:
            print("CU {} train_or_test are different Major {} vs. Minor {} ".format(cu, train_or_test_major, train_or_test_eileen))

        val_major = df_major[df_major.cutout_uuid == cu]['val'].iloc[0]
        val_eileen = df_eileen[df_eileen.cutout_uuid == cu]['val'].iloc[0]
        if val_major != val_eileen and train_or_test_major != 'test': # val is effective only upon train
            Warning("CU {} val field are different Major {} vs. Minor {} ".format(cu, val_major, val_eileen))

    plt.hist(all_relative_to_major_percent, bins=100)


if __name__ == "__main__":
    main()


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
        df_major = df_major[df_major.cutout_uuid != cu]
        df_major = df_major.append(df_eileen.loc[df_eileen.cutout_uuid == cu])

    df_major = df_major.dropna(axis=0, subset=['file_name'])
    df_major['file_name'] = df_major.apply(lambda x: str(x.file_name).replace("unknown-tested", str(x['class'])),
                                           axis=1)
    # df_major['file_name'] = df_major['file_name'].apply(lambda x:str(x).replace("unknown-tested", str(df_major['class'])))
    # for ind, row in df_major.iterrows():	row['file_name'].replace("unknown-tested", str(row['class']))
    df_major.to_csv(
        r'C:\Users\hanoch.kremer\OneDrive - ALLFLEX EUROPE\HanochWorkSpace\Data\DataBaseInfo\tile_location\annotations_fix_eileen.csv',
        index=False)
