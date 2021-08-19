import os
import pandas as pd
from argparse import ArgumentParser

def main(args: list = None):
    parser = ArgumentParser()
    parser.add_argument('--path', type=str, default='/hdd/hanoch/runmodels/img_quality/results/summary_results', metavar='PATH',
                        help="if given, all output of the training will be in this folder. "
                             "The exception is the tensorboard logs.")

    args = parser.parse_args(args)
    if 0:
        path = '/hdd/hanoch/data/objects-data-bbox-20191106-simple-sharded-part/tile_data/test_eileen_best_qual/csv'
        filenames = [os.path.join(path, x) for x in os.listdir(path)
                     if x.endswith('csv')]

        df_acm = pd.DataFrame()
        for file in filenames:
            df = pd.read_csv(file, index_col=False)
            file_patt = df.full_file_name[0].split('/')[-1].split('.')[0].split('_')[1:]
            df['file_name'] = file_patt[0] + '_' +  "_".join(df.full_file_name[0].split('/')[-1].split('.')[0].split('_')[1:])
            df['val'] = 0
            df_acm = df_acm.append((df))

        cols = df_acm.columns.to_list()
        cols2 = [cols[-2]] + cols[2:-2] + [cols[-1]]
        cols3 = cols2[:-3] + cols2[-2:]
        df_acm = df_acm[cols3]
        df_acm.to_csv(os.path.join(path, 'merged.csv'), index=False)

    else:
        path = args.path
        from pathlib import Path
        Path(os.path.join(path, 'merged')).mkdir(parents=True, exist_ok=True)

        filenames = [os.path.join(path, x) for x in os.listdir(path)
                     if x.endswith('csv')]

        df_acm = pd.DataFrame()
        for file in filenames:
            df = pd.read_csv(file, index_col=False)
            if 1:
                df.columns = df.iloc[0]
                df = df[1:2]
            print(file)
            df_acm = df_acm.append((df))

        # df_acm = df_acm.reindex(sorted(df_acm.columns), axis=1)
        df_acm.to_csv(os.path.join(path, 'merged', 'merged.csv'), index=False)


if __name__ == '__main__':
    main()

