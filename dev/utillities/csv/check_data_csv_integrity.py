import pandas as pd
import tqdm
from argparse import ArgumentParser
from dev.test_blind_quality import clsss_quality_label_2_str
import warnings
# path = '/hdd/hanoch/data/objects-data-bbox-20191106-simple-sharded-part/tile_data/temp_files/annotations_fix_eileen8_with_fitjar_marginal_cages.csv'
def main(args: list = None):

    parser = ArgumentParser()

    parser.add_argument("--path-major", type=str, required=True, metavar='PATH',
                        help="path to the csv/pkl files")

    parser.add_argument("--n_classes",  type=int, default=3, metavar='INT', help="")

    args = parser.parse_args(args)

    print("File major {}".format(args.path_major))

    df_major = pd.read_csv(args.path_major, index_col=False)
    df_major = df_major.dropna(axis=0, subset=['file_name'])

    if df_major["class"].unique().shape[0] == 2:
        if 'bad' not in df_major["class"].unique():
            raise
        if 'good' not in df_major["class"].unique():
            raise
    elif df_major["class"].unique().shape[0] == 3:
        if 'bad' not in df_major["class"].unique():
            raise
        if 'good' not in df_major["class"].unique():
            raise
        if 'marginal' not in df_major["class"].unique():
            raise
    else:
        if args.n_classes > 1:
            raise ValueError("Number of classes should be 2 or 3 ")
#TODO chaeck file name has the label of the class
# check label, train_or_test, val fileds are exist
    # check per cu all tiles has the same label, class, train_or_test, val
    for cu in tqdm.tqdm(df_major.cutout_uuid.unique()):
        if df_major[df_major.cutout_uuid == cu]['class'].size == 0:
            print(" CU {} No entries in ".format(cu))
        if df_major[df_major.cutout_uuid == cu].label.unique().shape[0] >1:
            print(" CU {} more than one label per CU !!! ".format(cu))
        if df_major[df_major.cutout_uuid == cu]["class"].unique().shape[0] >1:
            print(" CU {} more than one class per CU !!! ".format(cu))
        if df_major[df_major.cutout_uuid == cu].train_or_test.unique().shape[0] >1:
            print(" CU {} more than one train_or_test per CU !!! ".format(cu))
        if df_major[df_major.cutout_uuid == cu].train_or_test.iloc[0] == 'train':
            if df_major[df_major.cutout_uuid == cu].val.unique().shape[0] >1:
                print(" CU {} more than one val per CU {}!!! ".format(cu, df_major[df_major.cutout_uuid == cu].val.unique()))

        if df_major[df_major.cutout_uuid == cu].N_rows.unique().shape[0] >1:
            print(" CU {} more than one N_rows per CU !!! ".format(cu))
        if df_major[df_major.cutout_uuid == cu].M_cols.unique().shape[0] >1:
            print(" CU {} more than one N_rows per CU !!! ".format(cu))
        if df_major[df_major.cutout_uuid == cu].tile_ind.max() > df_major[df_major.cutout_uuid == cu].N_rows.iloc[0]*df_major[df_major.cutout_uuid == cu].M_cols.iloc[0]:
            print(" CU {} tile id out of blind grid {}!!! ".format(cu, df_major[df_major.cutout_uuid == cu].tile_ind.max()))

# TODO validate that class name is equiv to label

        if pd.isnull(df_major[df_major.cutout_uuid == cu].train_or_test).any():
            print(" CU {}  !!! train_or_test is null".format(cu))
        if pd.isnull(df_major[df_major.cutout_uuid == cu]["class"]).any(): # null will give nan hence shape =2 will be failed before
            print(" CU {}  !!! class is null".format(cu))
        if pd.isnull(df_major[df_major.cutout_uuid == cu]["N_rows"]).any():
            print(" CU {}  !!! class is null".format(cu))
        if pd.isnull(df_major[df_major.cutout_uuid == cu]["M_cols"]).any():
            print(" CU {}  !!! class is null".format(cu))


        if df_major[df_major.cutout_uuid == cu].media_uuid.unique().shape[0] >1:
            print(" CU {}  !!! has more than one media_id prob duplicated image {}".format(cu, df_major[df_major.cutout_uuid == cu].media_uuid.unique()))

    print('data csv integrity check pass')
#check unified media_is and cutout_id


if __name__ == "__main__":
    main()
