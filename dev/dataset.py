import copy

import torch
from torchvision import datasets, transforms
import os
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF
import matplotlib.colors as mcolors
from typing import List
clsss_quality_label_2_str = {1: 'bad', 2: 'marginal', 3: 'good', 4: 'unknown-tested'}
quality_to_labels_annotator_version = {'poor': 1, 'marginal': 2, 'excellent': 3, 'unknown-tested' :4}
quality_to_labels_basic_version = {'bad': 1, 'marginal': 2, 'good': 3, 'unknown-tested' :4}
softmax_map_to_final_labels = {0 :1, 1: 3, 2: 2}
class_labels = {'bad': 0, 'good': 1, 'marginal': 2, 'unknown-tested': 4}  # maintain same convention as load_data_train_val_test()


# Old imp for basic transfer learning
def load_data_train_val_test(input_size, data_dir, batch_size=64, num_workers=12):

    data_transforms = {
        'train': transforms.Compose([
            # transforms.RandomResizedCrop(input_size), was originally but here we dont want resize
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(), #HK@@
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            # transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            # transforms.Resize(input_size), # when inputs are withing diff resolution
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val', 'test']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                  shuffle=True, num_workers=12) for x in ['train', 'val']}
# No shuffle for test data loader in order to analyse
    dataloaders.update({x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                        shuffle=False, num_workers=num_workers) for x in ['test']})

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    for k, v in dataset_sizes.items():
        print("Set: {} Support:{}".format(k, v))

    class_names = image_datasets['train'].classes

    return dataloaders, dataset_sizes, class_names


def prepare_dataloaders(args, train_on_all=False, train_filter_percentile40={}, test_filter_percentile40={}, **kwargs):
    #data loading
    csv_path = args.dataset_split_csv
    if os.path.splitext(csv_path)[1] == '.csv':
        dataframe = pd.read_csv(csv_path)
    else: # hk@@ TODO : pickle
        raise

    if not args.classify_image_all_tiles:
         dataframe = dataframe[dataframe["class"] != 'marginal']

    if not kwargs['overfitting_test']: #Partach Incase model is expected to overfit than train=val=test observing loss~0 AP~1
        dataframe = dataframe[~dataframe['file_name'].duplicated()] # remove dup take 1st replica

    # train_on_all = False
    train_df = dataframe.loc[dataframe['train_or_test'] == 'train']
    test_df = dataframe.loc[dataframe['train_or_test'] == 'test']
    val_df = train_df[train_df['val'] == 1]
    train_df = train_df[train_df['val'] != 1]
    if args.replace_val_with_test:
        temp_df = copy.deepcopy(val_df)
        val_df = test_df
        test_df = temp_df

    if args.dedicated_subdir_per_class:
        train_df['full_file_name'] = train_df.apply(
            lambda x: os.path.join(args.database_root, 'train', x['class'], x['file_name'] + '.png'), axis=1)

        val_df['full_file_name'] = val_df.apply(lambda x: os.path.join(args.database_root, 'val', x['class'], x['file_name'] + '.png'), axis=1)

        test_df['full_file_name'] = test_df.apply(
            lambda x: os.path.join(args.database_root, 'test', x['class'], x['file_name'] + '.png'), axis=1)
    else:
        train_df['full_file_name'] = train_df.apply(
            lambda x: os.path.join(args.database_root, x['file_name'] + '.png'), axis=1)

        val_df['full_file_name'] = val_df.apply(
            lambda x: os.path.join(args.database_root, x['file_name'] + '.png'), axis=1)

        test_df['full_file_name'] = test_df.apply(
            lambda x: os.path.join(args.database_root, x['file_name'] + '.png'), axis=1)

    if args.classify_image_all_tiles:#collapse marginal to bad
        if kwargs['num_classes'] == 2: #fusion but gettting binary classification out of trenary class based data
            train_df = train_df[train_df["class"] != 'marginal'] # remove marginal from training the concept is not good to mix labels in tile level
            # val set collapse trinary to binary
            val_df["class"].loc[val_df["class"] == 'marginal'] = "bad"
            val_df["label"].loc[val_df["label"] == quality_to_labels_basic_version['marginal']] = quality_to_labels_basic_version['bad']
            # test set collapse trinary to binary
            test_df["class"].loc[test_df["class"] == 'marginal'] = "bad"
            test_df["label"].loc[test_df["label"] == quality_to_labels_basic_version['marginal']] = quality_to_labels_basic_version['bad']

    # Sanity check
    assert train_df['full_file_name'].apply(lambda x: os.path.isfile(x)).all(), "Some images referenced in the CSV file were not found"
    assert val_df['full_file_name'].apply(lambda x: os.path.isfile(x)).all(), "Some images referenced in the CSV file were not found"

    assert test_df['full_file_name'].apply(lambda x: os.path.isfile(x)).all(), "Some images referenced in the CSV file were not found"

    if args.train_by_all_data:
        train_df = train_df.append([test_df])
        train_df = train_df.append([val_df])
        test_df = train_df[0:0]
        val_df = train_df[0:0]

    if args.classify_image_all_tiles:
        print("Trainset support: {}".format(len(train_df['cutout_uuid'].unique())))
        print("Valset support : {}".format(len(val_df['cutout_uuid'].unique())))
        print("Testset support: {}".format(len(test_df['cutout_uuid'].unique())))
    else:
        print("Number of samples/tiles trainset: {}".format(len(train_df)))
        print("Number of samples/tiles valset  : {}".format(len(val_df)))
        print("Number of samples/tiles testset : {}".format(len(test_df)))

    # train_df['full_file_name'][train_df['full_file_name'].apply(lambda x: os.path.isfile(x)) == False]
    colour_norm = args.colour_norm if 'colour_norm' in args else None
    gamma_inv_corr = args.gamma_inv_corr if 'gamma_inv_corr' in args else None
    hsv_hue_jitter = args.hsv_hue_jitter if 'hsv_hue_jitter' in args else 0
    hsv_sat_jitter = args.hsv_sat_jitter if 'hsv_sat_jitter' in args else 0
    single_channel_input_grey = args.single_channel_input_grey if 'single_channel_input_grey' in args else None
    hue_norm_preprocess = args.hue_norm_preprocess if 'hue_norm_preprocess' in args else False
    hue_norm_preprocess_type = args.hue_norm_preprocess_type if 'hue_norm_preprocess_type' in args else False
    handcrafted_features_type = args.handcrafted_features if 'handcrafted_features' in args else None
    balancing_sampling = args.balancing_sampling if 'balancing_sampling' in args else None
    classify_image_all_tiles = args.classify_image_all_tiles if 'classify_image_all_tiles' in args else None
    get_image_name_item = args.get_image_name_item if 'get_image_name_item' in args else None

    kwargs_loc = {'gamma_inv_corr': gamma_inv_corr, 'hsv_hue_jitter': hsv_hue_jitter, 'hsv_sat_jitter': hsv_sat_jitter,
              'single_channel_input_grey': single_channel_input_grey, 'colour_norm': colour_norm,
              'hue_norm_preprocess': hue_norm_preprocess,
              'hue_norm_preprocess_type': hue_norm_preprocess_type,
              'handcrafted_features_type': handcrafted_features_type,
              'balancing_sampling': balancing_sampling,
              'classify_image_all_tiles': classify_image_all_tiles,
              'get_image_name_item': get_image_name_item,
              'in_outline_tiles_id_type': 'default'}

    kwargs.update(kwargs_loc)
    print('Train dataset')
    train_dataset = Dataset(train_df, mirror_ver=0.5,
                 mirror_hor=0.5, center_crop_size=args.input_size, filter_percentile40=train_filter_percentile40,
                            pre_load_images=False,
                            train_dataset=True, gamma_aug_corr=args.gamma_aug_corr, **kwargs)

    print('Validation dataset')
    val_dataset = Dataset(val_df, mirror_ver=0.0,
                 mirror_hor=0.0, center_crop_size=args.input_size, pre_load_images=False, **kwargs)

    print('Test dataset')
    test_dataset = Dataset(test_df, mirror_ver=0.0,
                 mirror_hor=0.0, center_crop_size=args.input_size, filter_percentile40=test_filter_percentile40,
                           pre_load_images=False, **kwargs)

# only train dataloader sample and shuffle attribute are of interest
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                                    shuffle=train_dataset.isShuffle,
                                                    pin_memory=False, num_workers=args.num_workers,
                                                    collate_fn=train_dataset.collate_fn,
                                                    sampler=train_dataset.sampler)
    sampler = None
    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=args.batch_size,
                                                    shuffle=False,
                                                    pin_memory=False, num_workers=args.num_workers,
                                                    collate_fn=train_dataset.collate_fn,
                                                    sampler=val_dataset.sampler)

    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                                                    shuffle=False,
                                                    pin_memory=False, num_workers=args.num_workers,
                                                    collate_fn=train_dataset.collate_fn,
                                                    sampler=test_dataset.sampler)

    return train_dataloader, val_dataloader, test_dataloader

def prepare_test_dataloader(args, test_df, test_filter_percentile40={}, **kwargs):
    # test_df  must have  the follwoing fields ['full_file_name'], ["class"] {good, or bad}

    if not args.classify_image_all_tiles:
         test_df = test_df[test_df["class"] != 'marginal']

        # Sanity check
    if args.dedicated_subdir_per_class:
        if 'full_file_name' not in test_df:
            test_df['full_file_name'] = test_df.apply(lambda x: os.path.join(args.database_root, args.filter_by_train_val_test , str(x['class']), x['file_name'] + '.png'), axis=1)
    else:
        test_df['full_file_name'] = test_df.apply(
            lambda x: os.path.join(args.database_root, x['file_name'] + '.png'), axis=1)

    assert test_df['full_file_name'].apply(lambda x: os.path.isfile(x)).all(), "Some images referenced in the CSV file were not found"

    if args.classify_image_all_tiles:
        print("Testset support: {}".format(len(test_df['cutout_uuid'].unique())))
    else:
        print("Testset support: {}".format(len(test_df)))

    gamma_inv_corr = args.gamma_inv_corr if 'gamma_inv_corr' in args else None
    hsv_hue_jitter = args.hsv_hue_jitter if 'hsv_hue_jitter' in args else 0
    hsv_sat_jitter = args.hsv_sat_jitter if 'hsv_sat_jitter' in args else 0
    single_channel_input_grey = args.single_channel_input_grey if 'single_channel_input_grey' in args else None
    get_image_name_item = args.get_image_name_item if 'get_image_name_item' in args else None
    pre_load_images = args.pre_load_images if 'pre_load_images' in args else False
    hue_norm_preprocess = args.hue_norm_preprocess if 'hue_norm_preprocess' in args else False
    hue_norm_preprocess_type = args.hue_norm_preprocess_type if 'hue_norm_preprocess_type' in args else False
    handcrafted_features_type = args.handcrafted_features if 'handcrafted_features' in args else None
    classify_image_all_tiles = args.classify_image_all_tiles if 'classify_image_all_tiles' in args else None


    kwargs_loc = {'gamma_inv_corr': gamma_inv_corr, 'hsv_hue_jitter': hsv_hue_jitter, 'hsv_sat_jitter': hsv_sat_jitter,
                  'single_channel_input_grey': single_channel_input_grey,
                  'get_image_name_item': get_image_name_item, 'hue_norm_preprocess': hue_norm_preprocess,
                  'hue_norm_preprocess_type': hue_norm_preprocess_type,
                  'handcrafted_features_type': handcrafted_features_type,
                  'classify_image_all_tiles': classify_image_all_tiles,
                  'in_outline_tiles_id_type': 'default'}

    kwargs.update(kwargs_loc)
    # train_dataset=False by default to avoid augmentation on test time
    test_dataset = Dataset(test_df, mirror_ver=0.0,
                               mirror_hor=0.0, center_crop_size=args.input_size,
                               filter_percentile40=test_filter_percentile40,
                               pre_load_images=pre_load_images, **kwargs)

    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                                                        shuffle=False,
                                                        pin_memory=False, num_workers=args.num_workers,
                                                        collate_fn=test_dataset.collate_fn,
                                                        sampler=test_dataset.sampler)

    return test_dataloader


def prepare_test_dataloader_tile_based(args, test_df, test_filter_percentile40={},
                                       pre_process_gamma_aug_corr=1, missing_label=False, **kwargs):
    # test_df  must have  the follwoing fields ['full_file_name'], ["class"] {good, or bad}

    if 'full_file_name' not in test_df:
        test_df['full_file_name'] = test_df['file_name'].apply(
            lambda x: os.path.join(args.database_root, 'test', x.split('_')[-3], x + '.png'))

    # Sanity check
    assert test_df['full_file_name'].apply(lambda x: os.path.isfile(x)).all(), "Some images referenced in the CSV file were not found"

    print("Number of samples testset: {}".format(len(test_df)))
    data_len = len(test_df)
    hsv_hue_jitter = args.hsv_hue_jitter if 'hsv_hue_jitter' in args else 0
    hsv_sat_jitter = args.hsv_sat_jitter if 'hsv_sat_jitter' in args else 0
    single_channel_input_grey = args.single_channel_input_grey if 'single_channel_input_grey' in args else None
    hue_norm_preprocess = args.hue_norm_preprocess if 'hue_norm_preprocess' in args else False
    hue_norm_preprocess_type = args.hue_norm_preprocess_type if 'hue_norm_preprocess_type' in args else False
    handcrafted_features_type = args.handcrafted_features if 'handcrafted_features' in args else None
    classify_image_all_tiles = args.classify_image_all_tiles if 'classify_image_all_tiles' in args else None
    get_image_name_item = args.get_image_name_item if 'get_image_name_item' in args else None

    kwargs_loc = {'hsv_hue_jitter': hsv_hue_jitter, 'hsv_sat_jitter': hsv_sat_jitter, 'missing_label': missing_label,
                  'single_channel_input_grey': single_channel_input_grey, 'hue_norm_preprocess': hue_norm_preprocess,
                  'hue_norm_preprocess_type': hue_norm_preprocess_type,
                  'handcrafted_features_type': handcrafted_features_type,
                  'classify_image_all_tiles': classify_image_all_tiles,
                  'get_image_name_item': get_image_name_item,
                  'in_outline_tiles_id_type' : 'pre_calc'}

    kwargs.update(kwargs_loc)

    test_dataset = Dataset(test_df, mirror_ver=0.0,
                           mirror_hor=0.0, center_crop_size=args.input_size,
                           filter_percentile40=test_filter_percentile40,
                           pre_load_images=False, train_dataset=False,
                           gamma_aug_corr=pre_process_gamma_aug_corr, **kwargs)
    # Batch size cant be greater than amount of tiles
    len_dataset = len(test_dataset.imagenames)
    actual_batch_size = min(args.batch_size, len_dataset)
#    actual_batch_size = len_dataset
#    print(actual_batch_size)

    if actual_batch_size == 0:
        return -1, actual_batch_size

    print("Batch size of dataloader was changed to {} Dataset filtered {}".format(actual_batch_size, data_len-len_dataset))

    sampler = None
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=actual_batch_size,
                                                  shuffle=False,
                                                  pin_memory=False, num_workers=args.num_workers,
                                                  collate_fn=test_dataset.collate_fn,
                                                  sampler=sampler)

    return test_dataloader, actual_batch_size



class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, dataframe, mirror_ver=0.0,
                 mirror_hor=0.0, center_crop_size=224, filter_percentile40={}, pre_load_images=False,
                train_dataset=False, gamma_aug_corr=1.0, **kwargs):

        # filter df according to percentile information less than (when image is darkish)  HK:TODO: validate after training that testset which is filtered as well react well
        if bool(filter_percentile40):
            if filter_percentile40['lessthan'] is not None:
                dataframe = dataframe[dataframe['percent_40'] > filter_percentile40['lessthan']]
            # filter df according to percentile information greater than (when image is saturated)
            if filter_percentile40['greater'] is not None:
                dataframe = dataframe[dataframe['percent_40'] < filter_percentile40['greater']]

        self.train_dataset = train_dataset

# extract per image/tile data
        self.imagenames = [name for name in dataframe["full_file_name"]]

        self.get_image_name_item = kwargs['get_image_name_item']

        self.classify_image_all_tiles = kwargs['classify_image_all_tiles']
        # Dont touch self.in_outline_tiles_id while self.get_image_name_item=True since in tile_based inference prodcution the id is already computed online by the create_tiles_and_meta_data()
        if kwargs['in_outline_tiles_id_type'] != 'pre_calc':
            self.in_outline_tiles_id = None
            if self.get_image_name_item: # tile based names
                self.in_outline_tiles_id = [image.split("/")[-1] for image in self.imagenames]
        else:
            # alredy given no need to recompute the [] for the getitem that will take index #0 of the entire indeces in the list
            if self.classify_image_all_tiles: # for all image we need all indeces in one access hence list otherwise
                self.in_outline_tiles_id = [np.array([name for name in dataframe["in_outline_tiles_id"]]).astype(np.float32)]
            else:
                self.in_outline_tiles_id = np.array([name for name in dataframe["in_outline_tiles_id"]]).astype(np.float32)

# MULTI INSTANCE LEARNING : traning all tiles blind #TODO add option to calculate all embeddings offline since FE isn;t trainable
        self.pooling_method = kwargs['pooling_method']

        self.collate_fn = None
        self.imagenames_all_blind = list()
        self.labels_all_blind = list()
        self.positional_embeddings = kwargs['positional_embeddings']
        if self.positional_embeddings:
            self.pos_n_rows_m_cols = list()
            self.tile_index_pos = list()

        if self.classify_image_all_tiles: # processing bag of instances/tiles
            for cu in dataframe['cutout_uuid'].unique():
                self.imagenames_all_blind.append(dataframe[dataframe['cutout_uuid'] == cu]["full_file_name"].tolist())
                self.labels_all_blind.append(class_labels[dataframe[dataframe['cutout_uuid'] == cu]["class"].unique().item()])
                if self.positional_embeddings is not None:
                    self.pos_n_rows_m_cols.append(np.array([dataframe[dataframe['cutout_uuid'] == cu]["N_rows"].unique().item(),
                                              dataframe[dataframe['cutout_uuid'] == cu]["M_cols"].unique().item()]))
                    self.tile_index_pos.append(np.array(dataframe[dataframe['cutout_uuid'] == cu]["tile_ind"])) # it is in the same order of the image names

            if self.pooling_method == 'gated_attention' or self.pooling_method == 'avg_pooling' or self.pooling_method == 'transformer_san':
                self.collate_fn = self._fn_collate_gather_var_batch_size


            if self.get_image_name_item and kwargs['in_outline_tiles_id_type'] != 'pre_calc':#override the item list to image list but not in cae of tile based processing infernce production
                self.in_outline_tiles_id = list()
                for images2 in self.imagenames_all_blind:
                    self.in_outline_tiles_id.append(os.path.split(images2[0])[-1].split('_tile_')[0])
                assert (len(self.in_outline_tiles_id) == len(self.labels_all_blind))

            if self.train_dataset: # only on training we care for balance to let production inference to rn over few samples
                assert (kwargs['num_classes'] == np.unique(self.labels_all_blind).shape[0])
            else:
                if (kwargs['num_classes'] != np.unique(self.labels_all_blind).shape[0]):
                    Warning('!!!!!!!!!!!!!   Data set does not contain all labels !!!!!!!!!!!!!!  ')



        missing_label = kwargs['missing_label'] if 'missing_label' in kwargs else False
        if missing_label is False:
            self.labels = np.array([class_labels[name] for name in dataframe["class"]]).astype(np.float32)
        else:
            self.labels = np.array([-1 for name in dataframe["class"]]).astype(np.float32)

#Class weights
        self.class_weights = list()
        if self.classify_image_all_tiles:
            for l in np.sort(np.unique(self.labels_all_blind)):  ##TODO: check if there is no bug in the index mapping from NN out to class id
                print('Support {} from label {}'.format((self.labels_all_blind == l).astype('int').sum(), l))
                self.class_weights.append((self.labels_all_blind == l).astype('int').sum())
        else:
            for l in np.sort(np.unique(self.labels)):
                print('Support {} from label {}'.format((self.labels == l).astype('int').sum(), l))
                self.class_weights.append((self.labels == l).astype('int').sum())
        self.class_weights = np.array(self.class_weights)
        self.class_weights = self.class_weights / self.class_weights.sum()
        self.class_weights = 1 / self.class_weights
        self.class_weights = self.class_weights / self.class_weights.sum()

# hand crafted features
        self.use_handcrafted_features = False
        if kwargs['handcrafted_features_type'] is not None:
            self.handcrafted_features_type = kwargs['handcrafted_features_type']
            self.use_handcrafted_features = True
            self.handcrafted_features_metadata_dictionary = dict()

            self.handcrafted_features = list()
            for hcf_type in self.handcrafted_features_type:
                if hcf_type not in dataframe:
                    raise Exception("No hand crafted value in this split csv file !!!!!!!!!")

                mu = dataframe[f"{hcf_type}_mean"].iloc[0]
                std = dataframe[f"{hcf_type}_std"].iloc[0]
                self.handcrafted_features_metadata_dictionary.update({hcf_type: {'mean': mu, 'std': std}})

                handcrafted_feature = np.array([name for name in dataframe[hcf_type]]).astype(np.float32)
                # Normalizee the HCF
                handcrafted_feature = (handcrafted_feature - dataframe[f"{hcf_type}_mean"].iloc[0])/dataframe[f"{hcf_type}_std"].iloc[0]
                self.handcrafted_features.append(handcrafted_feature)

        self.tta = kwargs['tta'] if 'tta' in kwargs else 0
        self.hue_norm_preprocess = kwargs['hue_norm_preprocess'] if 'hue_norm_preprocess' in kwargs else 0
        self.hue_norm_preprocess_type = kwargs['hue_norm_preprocess_type'] if 'hue_norm_preprocess_type' in kwargs else 0

        norm_image = 'matt_norm'
        if norm_image == 'imagenet_norm':
            self.normalize_rgb_mean = [0.485, 0.456, 0.406]
            self.normalize_rgb_std = [0.229, 0.224, 0.225]
            # self.transform_op += [
            #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        elif norm_image == 'matt_norm':
            self.normalize_rgb_mean = [0.13804892, 0.21836744, 0.20237076]
            self.normalize_rgb_std = [0.08498618, 0.07658653, 0.07137364]


        # a normalization factor of weighted average hue over all image
        if self.hue_norm_preprocess:
            self.all_image_weighted_hue = [name for name in dataframe["all_image_weighted_hue"]]
            if self.hue_norm_preprocess_type == 'weighted_hue_by_sv_recenter_sat':
                self.all_image_mean_sat = [name for name in dataframe["all_image_mean_sat"]]

        if 'colour_norm' in kwargs:
            self.colour_norm = kwargs['colour_norm']

        self.mirror_ver = mirror_ver
        self.mirror_hor = mirror_hor
        if center_crop_size == 0:
            raise Exception('Center crop size must be gt 0: {:}'.format(center_crop_size))

        self.center_crop_size = center_crop_size
        self.gamma_aug_corr = gamma_aug_corr
        self.hsv_hue_jitter = kwargs['hsv_hue_jitter'] # additive Uniform[- val, val]
        self.hsv_sat_jitter = kwargs['hsv_sat_jitter']  # a multiplier Uniform[max(0, 1 - saturation), 1 + saturation] ] => 90% to 110%
        if 'single_channel_input_grey' in kwargs:
            self.single_channel_input_grey = kwargs['single_channel_input_grey']
        else:
            self.single_channel_input_grey = False


        # train_dataset: # only then augmentation is allowed : on test set is a pre-processing
        if self.gamma_aug_corr != 1:
            self.gamma_inv_corr = [kwargs['gamma_inv_corr'] if kwargs['gamma_inv_corr'] is not False else False][0]
            self.gamma_prob = self.gamma_aug_corr
            if self.train_dataset:
                print("Gamma augemntation: {} with prob: {}".format(self.gamma_aug_corr, self.gamma_prob))
            else:
                print("Gamma pre-process (testset): {} with prob: {}".format(self.gamma_aug_corr, self.gamma_prob))

        self.pre_load_images = pre_load_images
        if self.pre_load_images:
            print("Starting pre-loading images!!")
            self.images = [
                self.load_image(image_path) for image_path in self.imagenames
            ]
            print("Done pre-loading images!!")

        if self.train_dataset:
            self.isShuffle = True

        self.balancing_sampling = kwargs['balancing_sampling'] if 'balancing_sampling' in kwargs else False
        if self.balancing_sampling and self.train_dataset:
            if self.classify_image_all_tiles:
                self.sampler = self._minority_database_oversampling_bag_all_image()
            else:
                self.sampler = self._minority_database_oversampling()
        else:
            self.sampler = None
        return

    def _fn_collate_gather_var_batch_size(self, batch):
        data = [item[0] for item in batch]
        target = [item[1] for item in batch]
        target = torch.LongTensor(target)
        if self.get_image_name_item:
            if self.positional_embeddings:
                raise ValueError("Option not implemented")
            image_name = [item[2] for item in batch]
            return [data, target, image_name]
        else:
            if self.positional_embeddings:
                assert(self.pooling_method != 'transformer_san')
                pos_n_rows_m_cols = [item[2] for item in batch]
                pos_n_rows_m_cols = torch.stack([torch.from_numpy(b) for b in pos_n_rows_m_cols], 0)
                tile_index_pos = [item[3] for item in batch]
                tile_index_pos = [torch.from_numpy(b.astype('int')) for b in tile_index_pos]
                return [data, target, pos_n_rows_m_cols, tile_index_pos]
            else:
                return [data, target]

    def _minority_database_oversampling_bag_all_image(self): #TODO: check sampler balanced batch
        cls_id = np.array(self.labels_all_blind)
        class_sample_count = np.array([len(np.where(cls_id == t)[0]) for t in np.unique(cls_id)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in cls_id])
        # replacement=True : If you specify replacement=False and keep the size as the whole dataset length, only your batches at the beginning will
        # be balanced using the weights until all minority classes are “used”. You could try to decrease the length so that most of
        # your batches will be balanced.
        sampler = torch.utils.data.WeightedRandomSampler(
            torch.from_numpy(samples_weight).type(torch.DoubleTensor), len(samples_weight), replacement=True)
        self.isShuffle = False  # shuffle is built in the Weighted Random sample
        return sampler

    def _minority_database_oversampling(self):
        cls_id = np.zeros(len(self.imagenames)).astype(int)
        cls_id[self.labels==1] = 1
        class_sample_count = np.array([len(np.where(cls_id == t)[0]) for t in np.unique(cls_id)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in cls_id])
        # replacement=True : If you specify replacement=False and keep the size as the whole dataset length, only your batches at the beginning will
        # be balanced using the weights until all minority classes are “used”. You could try to decrease the length so that most of
        # your batches will be balanced.
        sampler = torch.utils.data.WeightedRandomSampler(
            torch.from_numpy(samples_weight).type(torch.DoubleTensor), len(samples_weight), replacement=True)
        self.isShuffle = False  # shuffle is built in the Weighted Random sample
        return sampler

    def transform(self, image):

        #default by model already trained with ImageNet - Tx learning
        self.transform_op = []

        #only for test/val center_crop_size
        if self.train_dataset:
            self.transform_op += [transforms.RandomCrop(self.center_crop_size)]
            if self.mirror_ver > 0:
                self.transform_op += [transforms.RandomVerticalFlip(p=self.mirror_ver)]

            if self.mirror_hor > 0:
                self.transform_op += [transforms.RandomHorizontalFlip(p=self.mirror_hor)]
        else:
            self.transform_op += [transforms.CenterCrop(self.center_crop_size)]

        self.transformations = transforms.Compose(self.transform_op)
        image = self.transformations(image)


        # apply gamma correction
        if self.train_dataset:
            if self.gamma_aug_corr != 1:
                if np.random.uniform() < 0.5:
                    # randomly augment by gamma_aug_corr or 1/gamma_aug_corr
                    if np.random.uniform() < 0.5: #iid no worries
                        image = TF.adjust_gamma(image, 1/self.gamma_aug_corr, gain=1)
                    else:
                        image = TF.adjust_gamma(image, self.gamma_aug_corr, gain=1)
        else:
            if self.gamma_aug_corr != 1:
                image = TF.adjust_gamma(image, self.gamma_aug_corr, gain=1)
#here transformation breaks into two steps so image is already transfered so far

        self.transform_op = []
        # Normalization and augmentation doesn't occur simultanuously
        if self.train_dataset and not self.colour_norm:
            if self.hsv_hue_jitter > 0 or self.hsv_sat_jitter > 0:
                self.transform_op = [transforms.ColorJitter(hue=(-self.hsv_hue_jitter, self.hsv_hue_jitter),
                                                            saturation=(self.hsv_sat_jitter))]
        # if self.colour_norm:
        #     image_hsv = mcolors.rgb_to_hsv(image.permute(1, 2, 0))
        #     mu_hsv = image_hsv.mean(axis=0).mean(axis=0)
        #     image_hsv = image_hsv - mu_hsv + [0.5, 0.5, 0]


        # test time aug only in test time  : reset the assignment for the transform
        if self.tta and not self.train_dataset:
            if self.tta == 1:
                    self.transform_op += [transforms.RandomVerticalFlip(p=1)]
                    self.transform_op += [transforms.RandomHorizontalFlip(p=1)]
                    # combine and apply and transfer to image PIL
                    self.transformations = transforms.Compose(self.transform_op)
                    image = self.transformations(image)

                    image = TF.adjust_hue(image, -0.05)
                    image = TF.adjust_saturation(image, 0.95)
                    self.transform_op = []
            elif self.tta == 2:
                self.transform_op += [transforms.RandomVerticalFlip(p=1)]
                self.transform_op += [transforms.RandomHorizontalFlip(p=1)]

                # combine and apply and transfer to image PIL
                self.transformations = transforms.Compose(self.transform_op)
                image = self.transformations(image)

                image = TF.adjust_hue(image, -0.05)
                self.transform_op = []
            elif self.tta == 3:
                self.transform_op += [transforms.RandomVerticalFlip(p=1)]
                self.transform_op += [transforms.RandomHorizontalFlip(p=1)]

                # combine and apply and transfer to image PIL
                self.transformations = transforms.Compose(self.transform_op)
                image = self.transformations(image)

                image = TF.adjust_hue(image, -0.1)
                self.transform_op = []
            elif self.tta == 4:
                self.transform_op += [transforms.RandomVerticalFlip(p=1)]
                self.transform_op += [transforms.RandomHorizontalFlip(p=1)]
            elif self.tta == 5:
                self.transform_op += [transforms.RandomVerticalFlip(p=1)]
                self.transform_op += [transforms.RandomHorizontalFlip(p=1)]

                # combine and apply and transfer to image PIL
                self.transformations = transforms.Compose(self.transform_op)
                image = self.transformations(image)

                image = TF.adjust_hue(image, +0.05)
                self.transform_op = []
            elif self.tta == 6:
                self.transform_op += [transforms.RandomVerticalFlip(p=1)]
                self.transform_op += [transforms.RandomHorizontalFlip(p=1)]
                # combine and apply and transfer to image PIL
                self.transformations = transforms.Compose(self.transform_op)

                image = self.transformations(image)
                image = TF.adjust_saturation(image, 1.05)
                self.transform_op = []
            elif self.tta == 7:
                image = TF.adjust_hue(image, -0.05)
                image = TF.adjust_saturation(image, 1.05)
                self.transform_op = []
            elif self.tta == 8:
                    image = TF.adjust_hue(image, 0.05)
                    image = TF.adjust_saturation(image, 0.95)
                    self.transform_op = []
            elif self.tta == 9:
                image = TF.adjust_saturation(image, 0.95)
                self.transform_op = []
            elif self.tta == 10:
                image = TF.adjust_saturation(image, 0.95)
                self.transform_op = []
            else:
                print("Warning!! not a valid tta option {}".format(self.tta))


        self.transform_op += [transforms.ToTensor()]

        self.transform_op += [
            transforms.Normalize(self.normalize_rgb_mean, self.normalize_rgb_std)]

        # else: # my statistics
        #     self.transform_op += [
        #         transforms.Normalize([0.13804892, 0.21836744, 0.20237076], [0.08498618, 0.07658653, 0.07137364])]

        # Actually replicate the grey level 3 times
        if self.single_channel_input_grey:
            transforms.Grayscale(num_output_channels=3)

        self.transformations = transforms.Compose(self.transform_op)
        # from torchvision.utils import save_image
        # save_image(image, 'img1.png')
        image = self.transformations(image)
        return image

    # img_to_save = image.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    # Image.fromarray(np.uint8(image.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()), 'RGB').save('after_jitter15.png')
    def __len__(self):
        'Denotes the total number of samples'
        if self.classify_image_all_tiles:
            return len(self.imagenames_all_blind)
        else:
            return len(self.imagenames)

    def _getitem_tile_based_by_index(self, index):
        # Load data and get label
        img = Image.open(self.imagenames[index])
        img = img.convert('RGB')
        # label = torch.from_numpy(self.labels[index])
        label = self.labels[index]

        if self.hue_norm_preprocess :#and not self.train_dataset:
            img_weighted_hue = self.all_image_weighted_hue[index]
            if self.hue_norm_preprocess_type == 'weighted_hue_by_sv_recenter_sat':
                img_mean_sat = self.all_image_mean_sat[index]
            else:
                img_mean_sat = 0
            img = self._hue_norm_preprocess_apply(img, img_weighted_hue, img_mean_sat)
        if 0:
            image_hsv = mcolors.rgb_to_hsv(img)
            print(label, image_hsv[:, :, 0].min(), image_hsv[:, :, 0].max(), image_hsv[:, :, 0].mean())

        # if self.transform is not None:
        img = self.transform(img)
        data_list = list()
        data_list.append(img)
        # data_dict = {'image_data': img}

        if self.use_handcrafted_features:
            for idx, _ in enumerate(self.handcrafted_features_type):
                hcf = self.handcrafted_features[idx][index]
                data_list.append(hcf)
                # data_dict.update({'hcf_data': hcf})

        if self.get_image_name_item:
            in_outline_tiles_id = self.in_outline_tiles_id[index]
            return data_list, label, in_outline_tiles_id
        else:
            return data_list, label

    def _getitem_normalized_tile_based_by_full_file_name(self, full_file_name):
        # Load data and get label
        img = Image.open(full_file_name)
        img = img.convert('RGB')

        # if self.transform is not None:
        img = self.transform(img)

        if self.use_handcrafted_features:
            raise #not imp. yet
        return img

    def _getitem_all_image_based(self, index):
        all_data_list = list()
        files = self.imagenames_all_blind[index]
        label = self.labels_all_blind[index]

        if self.positional_embeddings is not None:
            # pos_n_rows_m_cols = torch.tensor(self.pos_n_rows_m_cols[index], dtype=torch.float32)
            pos_n_rows_m_cols = self.pos_n_rows_m_cols[index]
            tile_index_pos = self.tile_index_pos[index]

        for file in files:
            data_tens = self._getitem_normalized_tile_based_by_full_file_name(file) # labels are identicals
            # if self.in_outline_tiles_id is not None:
            #     raise # not imp yet
            all_data_list.append(data_tens.unsqueeze(dim=0))
        # print(len(all_data_list))
        # print(file)
        all_data_list = torch.cat(all_data_list, dim=0)

        if self.get_image_name_item:
            if self.positional_embeddings is not None:
                raise ValueError("Not supported all theses options")
            in_outline_tiles_id = self.in_outline_tiles_id[index]
            return all_data_list, label, in_outline_tiles_id
        else:
            if self.positional_embeddings is not None:
                return all_data_list, label, pos_n_rows_m_cols, tile_index_pos
            else:
                return all_data_list, label


    def __getitem__(self, index):
        'Generates one sample of data'
        if self.classify_image_all_tiles:
            return self._getitem_all_image_based(index)
        else:
            return self._getitem_tile_based_by_index(index)

    def augment_data(self, data):
        raise
        return data

    def load_image(self, path):
        img = Image.open(path)
        img = img.convert('RGB')
        return img

    def _hue_norm_preprocess_apply(self, img, img_weighted_hue, all_image_mean_sat=0):

        if self.hue_norm_preprocess_type == 'weighted_hue_by_sv': # per image normalization of hue: the factor was calculated offline and is in the csv
            # renorm by mean of hsv per image to a predefined mean which correspond to the train set stat
            image_hsv = mcolors.rgb_to_hsv(np.array(img)/255)
            h_norm = image_hsv[:, :, 0] - img_weighted_hue + 0.5  # 0.5 works fine also
            h_norm[h_norm < 0] = h_norm[h_norm < 0] + 1 # wrap arround % 1.0
            image_hsv_new = np.dstack((h_norm, image_hsv[:, :, 1], image_hsv[:, :, 2]))
            hsv2rgb = mcolors.hsv_to_rgb(image_hsv_new)
            hsv2rgb = hsv2rgb * 255.0
            if 0:
                import pickle
                path = '/hdd/hanoch/data/temp/hsv2rgb.pkl'
                with open(path, 'wb') as f:
                    pickle.dump(hsv2rgb, f)
                np.seterr(all='raise')

            hsv2rgb[hsv2rgb>255.0] = 255.0
            img = Image.fromarray(hsv2rgb.astype(np.uint8))
            # TODO: add numpy.seterr(all='raise') to catch the   RuntimeWarning: invalid value encountered in greater
            return img
        elif self.hue_norm_preprocess_type == 'weighted_hue_by_sv_recenter_sat':
            if all_image_mean_sat == 0:
                Warning('Saturation mean of image shouldnot be zero !!!!')

            image_hsv = mcolors.rgb_to_hsv(np.array(img)/255)
            # Normalizes SAT
            image_hsv[:, :, 1] = image_hsv[:, :, 1] - all_image_mean_sat + 0.5
            # Weighted norm of Hue
            h_norm = image_hsv[:, :, 0] - img_weighted_hue + 0.5  # 0.5 works fine also
            h_norm[h_norm < 0] = h_norm[h_norm < 0] + 1 # wrap arround % 1.0
            image_hsv_new = np.dstack((h_norm, image_hsv[:, :, 1], image_hsv[:, :, 2]))
            hsv2rgb = mcolors.hsv_to_rgb(image_hsv_new)
            hsv2rgb = hsv2rgb * 255.0
            hsv2rgb[hsv2rgb>255.0] = 255.0
            img = Image.fromarray(hsv2rgb.astype(np.uint8))
            # TODO: add numpy.seterr(all='raise') to catch the   RuntimeWarning: invalid value encountered in greater
            return img

        else:
            raise
        # else: # renorm by mean of hsv per tile to a predefined meanwhich correspond to the train set stat
        #     image_hsv = mcolors.rgb_to_hsv(img)
        #     hsv_mean = image_hsv.mean(axis=0).mean(axis=0)
        #     # print(hsv_mean)
        #     h_norm = image_hsv[:, :, 0] - hsv_mean[0] + 0.45 #0.45 works fine also
        #     h_norm[h_norm < 0] = h_norm[h_norm < 0] + 1 # wrap arround % 1.0
        #     # h_norm[h_norm < 0] = 0
        #
        #     # print(h_norm.min(), image_hsv[:, :, 0].min(), h_norm.max(), image_hsv[:, :, 0].max())
        #     # sh_norm = image_hsv[:,:,1]*0.929/hsv_mean[1]
        #     sh_norm = image_hsv[:, :, 1]
        #     image_hsv_new = np.dstack((h_norm, sh_norm,image_hsv[:, :, 2]))
        #     # image_hsv = [0.5, 0.9, 50]*image_hsv/hsv_mean
        #     # print(image_hsv_new.mean(axis=0).mean(axis=0))
        #     img = Image.fromarray((mcolors.hsv_to_rgb(image_hsv_new) * 255).astype(np.uint8))
        #     return img



# Dataset with given loaded images
class DatasetOnLine(Dataset):
    def __init__(self, data, target, center_crop_size=224,  tta=False, to_scale=True):
        # self.data = torch.from_numpy(data).float()
        # self.target = torch.from_numpy(target).long()
        self.data = data
        self.target = target
        self.center_crop_size = center_crop_size
        # self.transform = transform
        self.tta = tta
        self.to_scale = to_scale

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)

    def transform(self, image):
        if 1:
            self.transformations = transforms.Compose([transforms.ToPILImage(), transforms.CenterCrop(self.center_crop_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            if self.to_scale:
                image = self.transformations((image * 255.0).astype(np.uint8))
            else:
                image = self.transformations(image)

            # image = Image.fromarray((tiles[tile_no, :, :, :] * 255.0).astype(np.uint8))


        else:
            #default by model already trained with ImageNet - Tx learning
            self.transform_op = []

            self.transform_op += [transforms.CenterCrop(self.center_crop_size)]

            # test time aug only in test time  : reset the assignment for the transform
            if self.tta and not self.train_dataset:
                if self.tta == 1:
                        self.transform_op += [transforms.RandomVerticalFlip(p=1)]
                        self.transform_op += [transforms.RandomHorizontalFlip(p=1)]
                        # combine and apply and transfer to image PIL
                        self.transformations = transforms.Compose(self.transform_op)
                        image = self.transformations(image)

                        image = TF.adjust_hue(image, -0.05)
                        image = TF.adjust_saturation(image, 0.95)
                        self.transform_op = []
                elif self.tta == 2:
                    self.transform_op += [transforms.RandomVerticalFlip(p=1)]
                    self.transform_op += [transforms.RandomHorizontalFlip(p=1)]

                    # combine and apply and transfer to image PIL
                    self.transformations = transforms.Compose(self.transform_op)
                    image = self.transformations(image)

                    image = TF.adjust_hue(image, -0.05)
                    self.transform_op = []
                elif self.tta == 3:
                    self.transform_op += [transforms.RandomVerticalFlip(p=1)]
                    self.transform_op += [transforms.RandomHorizontalFlip(p=1)]

                    # combine and apply and transfer to image PIL
                    self.transformations = transforms.Compose(self.transform_op)
                    image = self.transformations(image)

                    image = TF.adjust_hue(image, -0.1)
                    self.transform_op = []
                elif self.tta == 4:
                    self.transform_op += [transforms.RandomVerticalFlip(p=1)]
                    self.transform_op += [transforms.RandomHorizontalFlip(p=1)]
                elif self.tta == 5:
                    self.transform_op += [transforms.RandomVerticalFlip(p=1)]
                    self.transform_op += [transforms.RandomHorizontalFlip(p=1)]

                    # combine and apply and transfer to image PIL
                    self.transformations = transforms.Compose(self.transform_op)
                    image = self.transformations(image)

                    image = TF.adjust_hue(image, +0.05)
                    self.transform_op = []
                elif self.tta == 6:
                    self.transform_op += [transforms.RandomVerticalFlip(p=1)]
                    self.transform_op += [transforms.RandomHorizontalFlip(p=1)]

                    # combine and apply and transfer to image PIL
                    self.transformations = transforms.Compose(self.transform_op)
                    image = self.transformations(image)
                    image = TF.adjust_saturation(image, 1.05)
                    self.transform_op = []
                else:
                    print("Warning!! not a valid tta option")


            self.transform_op += [transforms.ToTensor()]
            self.transform_op += [
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]

            self.transformations = transforms.Compose(self.transform_op)
            image = self.transformations(image)
        return image

"""
nice code for sampler

self.label2idxes = {
            label: np.arange(len(labels))[labels == label].tolist()
            for label in set(labels)
        }

self.images = [np.asarray(self.load_image(image_path)) for image_path in self.imagenames[:10000]]
mean_val = [np.mean(img, axis=tuple(range(img.ndim-1))) for img in self.images] # calc for RGB at once
mean_dataset = np.mean(np.vstack(mean_val),axis=0)
print(mean_dataset)

std_val = [np.std(img, axis=tuple(range(img.ndim-1))) for img in self.images  # calc for RGB at once]
std_dataset = np.std(np.vstack(std_val),axis=0)
print(std_dataset)

self._hue_norm_preprocess_apply(self.images[np.where(cc['good_det'])[0][ll]]).save(os.path.join(path,'hit_det_good_cls_' + str(np.where(cc['good_det'])[0][ll])+ '.png'))
self.images[np.where(cc['good_det'])[0][ll]].save(os.path.join(path,'hit_det_good_cls_' + str(np.where(cc['good_det'])[0][ll]) + '_orig'+ str(cc['all_predictions'][np.where(cc['good_det'])[0][ll],1]) +'.png'))
with open(os.path.join(path,'meta_data.pkl'), 'rb') as f:
    cutout_2_outline_n_meta = pickle.load(f)   
"""
