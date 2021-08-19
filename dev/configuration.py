import enum
import os
import pandas as pd
# RUN_CONFIGS = [
#     *PLAYGROUND_CONFIGS, *IMPORTANT_EXPERIMENT_CONFIGS, *RELEASED_CONFIGS
# ]
#
#
# def get_run_configuration(experiment_name, raise_on_error=True):
#     for experiment in RUN_CONFIGS:
#         if experiment.name == experiment_name:
#             return experiment
#     if raise_on_error:
#         raise KeyError("Experiment configuration not found: " + experiment_name)
#     return None


class CLARG_SETS(enum.Enum):
    """Identifiers for use with sets of cmd line arguments."""

    COMMON = 0
    DATASET = 1  # for specifying the dataset to operate on
    TRAINING = 2   # for specifying the training setup
    ARCHITECTURE = 3  # for specifying the network (model) architecture
    SETUP = 4 # patch for the model name not needed in inference only in training
    # INFERENCE = 4   # inference specifics
    # EXPORT = 5  # export specifics


def add_clargs(args_parser, clarg_set: CLARG_SETS):
    """
    Configure the specified cmd line argument parser with arguments in the referenced arguments set.
    """

    if clarg_set == CLARG_SETS.SETUP:
        args_parser.add_argument('run_config_name', type=str, metavar='RUN_CONFIGURATION',
                                 help="name of an existing run configuration")

    elif clarg_set == CLARG_SETS.COMMON:
        args_parser.add_argument('--database-root', type=str, required=True, metavar='PATH',
                                                    help="path to the database")

        args_parser.add_argument('--dedicated-subdir-per-class', action='store_true',
                                        help='train val and test taken from different subdir else from same subdir' )

        args_parser.add_argument('--dataset-split-csv', type=str, required=False, metavar='PATH',
                                                        help="path to the csv defining the train-test split")

        args_parser.add_argument('--get-image-name-item', action='store_true',
                                        help='get_image_name_item for per image result')



    elif clarg_set == CLARG_SETS.DATASET:
        clarg_group_name = 'dataset arguments'
        args_parser = args_parser.add_argument_group(clarg_group_name)
        args_parser.add_argument('--normalization', type=str, default=None, metavar='STRING', help="TODO")
        args_parser.add_argument('--pre-load-images', type=int, default=0, help="If 1, the images will be pre-loaded "
                                                                                "before training.")

    elif clarg_set == CLARG_SETS.ARCHITECTURE:
        clarg_group_name = 'architecture arguments'
        args_parser = args_parser.add_argument_group(clarg_group_name)
        args_parser.add_argument('--dropout', type=float, default=0.0, metavar='FLOAT', help="TODO")

        args_parser.add_argument('--handcrafted-features', '--list', nargs='+',
                            help='name of hand crafted feature as appeared in the csv', required=False)

        args_parser.add_argument('--fine-tune-pretrained-model-plan', type=str, default=None,
                                 choices=['freeze_pretrained_add_nn_avg_pooling', 'freeze_pretrained_add_nn_lp', 'freeze_pretrained_add_gated_atten',
                                          'freeze_pretrained_add_vit'],
                            # TODO add json/yaml that tells what to freeze and what to add to the current NN
                            help='If given, the training will start with a pre-trained model.')

        args_parser.add_argument('--classify-image-all-tiles', action='store_true',
                            help='MIL scheme')

        args_parser.add_argument('--pooling-at-classifier', action='store_false',
                            help='Preprocessing the Hue mean offset to 0.45')

        args_parser.add_argument('--fc-sequential-type', type=str, default='2FC',
                            # for modified model load predefined, remove head and implant new  fc_sequential_type before classifier
                            choices=['2FC', '1FC_all_dropout'],
                            help='NN non linear head before the classifier ')

        args_parser.add_argument("--positional-embeddings", default=None,
                            choices=['raster_bitmap_8_8_1D', 'raster_bitmap_8_8_mlp_1D', 'additive_raster_bitmap_8_8_mlp_1D'],
                            help="Adding positional embeddings for attention over tiles scheme")

        args_parser.add_argument("--transformer-param-list", nargs="+", default=["256", "2", "200", "2", "1"],
                                 help="d_model, nhead=2, nhid=200, nlayers=2, [1,2,3] for pos embeddings type: embed_dim must be divisible by num_heads")


    elif clarg_set == CLARG_SETS.TRAINING:
        clarg_group_name = 'training arguments'
        args_parser = args_parser.add_argument_group(clarg_group_name)
        args_parser.add_argument('--batch-size', type=int, default=None, metavar='INT', help="TODO")
        args_parser.add_argument('--optimizer', type=str, default='adam', metavar='STRING', help="TODO")
        args_parser.add_argument('--adamw-epsilon', type=float, default=1e-8, metavar='FLOAT', help="TODO")
        args_parser.add_argument('--lr', type=float, default=0.001, metavar='FLOAT', help="TODO")
        args_parser.add_argument('--lr-decay-base', type=float, default=0.9, metavar='FLOAT', help="lr base exponential base ")
        args_parser.add_argument('--npoch-decay-step', type=float, default=-1, metavar='FLOAT', help="gear shift time interval in iterations")

        args_parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='FLOAT', help="TODO")
        args_parser.add_argument('--epochs', type=int, default=20, metavar='INT', required=True, help="TODO")
        args_parser.add_argument('--max-iterations', type=int, default=None, metavar='INT', help="TODO")
        args_parser.add_argument('--train-by-all-data', action='store_true', help='')

    else:
        raise ValueError("Unknown command line arguments group: {}".format(clarg_set))



def print_arguments(args):
    """Print the specified map object ordered by key; one line per mapping"""
    header = "Command line arguments:"
    print('\n' + header)
    print(len(header) * '-')
    args_dict = vars(args)
    arguments_str = '\n'.join(["{}: {}".format(key, args_dict[key]) for key in sorted(args_dict)])
    print(arguments_str + '\n')


def load_csv_or_zipped(csv_path):
    """
    Some CSV files are too big to push to git, so we zip them. This method makes loading format-agnostic.
    """
    if os.path.splitext(csv_path)[1] == '.csv':
        dataframe = pd.read_csv(csv_path)
    else:
        dataframe = pd.read_csv(csv_path, compression='zip')

    if 'database_name' in dataframe.columns:
        dataframe = dataframe.astype({'database_name': str})

    return dataframe


def get_train_test_dataframes_from_csv(csv_path, train_on_all=False, selected_fold=None):
    """
    From the pre-compiled train-test split describing CSV path, create the train and test dataframes.

    If the CSV describes a crossvalidation set, ensures that it's compatible with the training framework.
    A selected_fold can be given to specify which fold is the test set for the training run. Default everything is
    training.
    """

    dataframe = load_csv_or_zipped(csv_path)

    # dataframe = cast_to_training_framework_format(dataframe, selected_fold)

    if train_on_all:
        df_train = dataframe
        df_test = df_train[0:0]
    else:
        df_train = dataframe.loc[dataframe['train_or_test'] == 'train']
        df_test = dataframe.loc[dataframe['train_or_test'] == 'test']

    # if len(df_train) == 0 and csv_path.endswith(('holdout.csv', 'holdout_used_subjects.csv')):
    #     train_csv_path = csv_path.replace('holdout.csv', 'training.csv')\
    #         .replace('holdout_used_subjects.csv', 'training.csv')

        # df_train = load_csv_or_zipped(csv_path)
        # df_train = cast_to_training_framework_format(df_train, selected_fold=None)

    if len(df_test) == 0 and csv_path.endswith('training.csv'):
        pass

    return df_train, df_test
