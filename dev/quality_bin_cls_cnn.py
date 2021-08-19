from __future__ import print_function, division

import os
import time
from argparse import ArgumentParser
from pathlib import Path
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import torch
# torch.multiprocessing.set_sharing_strategy('file_system') # due to RuntimeError: received 0 items of ancdata https://discuss.pytorch.org/t/runtimeerror-received-0-items-of-ancdata/4999/2
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
# from warmup_scheduler import GradualWarmupScheduler
import pytorch_warmup as warmup  #https://github.com/Tony-Y/pytorch_warmup :pip install -U pytorch_warmup
from torch.utils.tensorboard import SummaryWriter

from dataset import prepare_dataloaders
from models import initialize_model
from configuration import add_clargs, CLARG_SETS, print_arguments
from evaluation import train_eval_model
from optim import FocalLoss

#https://kevinmusgrave.github.io/   pip install pytorch-metric-learning
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.reducers import MeanReducer
from pytorch_metric_learning.regularizers import LpRegularizer
from pytorch_metric_learning import losses, miners


def list_public_members(obj):
    """Return list of all the specified object's members supposedly for public access."""
    return [member for member in dir(obj) if member[0] != '_']


def get_customized_run_configuration(run_config, args):
    """
    Return a run configuration with the specified name, cloned from an existing one and then updated from the
    specified arguments map where applicable.
    """

    def update_run_config_from_clargs(config):
        for attrib in list_public_members(config):
            clarg_value = getattr(args, attrib, None)
            if clarg_value is not None:
                setattr(config, attrib, clarg_value)

    def add_to_dict_if_not_none(args, attribute_name, config_dict, key_name=None):
        value = getattr(args, attribute_name, None)

        key_name = attribute_name if key_name is None else key_name

        if value is not None:
            config_dict[key_name] = value

    # run_config = get_run_configuration(name).clone()
    #
    # update_run_config_from_clargs(run_config)

    # Dataset args
    # add_to_dict_if_not_none(args, 'normalization', run_config.dataset_args, 'image_normalization')
    # add_to_dict_if_not_none(args, 'pre_load_images', run_config.dataset_args)

    # Architecture_args
    add_to_dict_if_not_none(args, 'dropout', run_config['train_args'])

    # Training args
    add_to_dict_if_not_none(args, 'batch_size', run_config['train_args'])
    add_to_dict_if_not_none(args, 'optimizer', run_config['train_args'])
    add_to_dict_if_not_none(args, 'lr', run_config['train_args'])
    add_to_dict_if_not_none(args, 'lr_decay_base', run_config['train_args'])
    add_to_dict_if_not_none(args, 'weight_decay', run_config['train_args'])
    add_to_dict_if_not_none(args, 'epochs', run_config['train_args'])
    add_to_dict_if_not_none(args, 'max_iterations', run_config['train_args'])

    # Pre-trained weights
    run_config['pretrained_weights_path'] = getattr(args, 'pretrained_weights_path', None)

    return run_config

def set_padding_mode_for_torch_compatibility(model):
    """
    Pytorch 1.1 fails to load pre-1.1 models because it expects the newly added padding_mode attribute.
    This function fixes the problem by adding the missing attribute with the default value.
    """
    modules = model.modules()
    for layer in modules:
        if isinstance(layer, nn.Conv2d):
            setattr(layer, 'padding_mode', 'zeros')

def weights_init(model):
    if isinstance(model, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(model.weight.data)
        if hasattr(model.bias, 'data'):
            model.bias.data.zero_()

def load_model(model_path, *args, **kwargs):
    model = torch.load(model_path, *args, **kwargs)

    if torch.__version__ == '1.1.0':
        set_padding_mode_for_torch_compatibility(model)

    return model

def save_checkpoint(path, epoch, model, optimizer, loss, model_name=None):
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'model_name': model_name,
                }, path)

def load_checkpoint(path, use_cuda=False):
    if use_cuda:
        return load_model(path)
    else:
        return load_model(path, map_location=lambda storage, loc: storage)

def load_model_checkpoint_state(model, checkpoint_path):
    checkpoint = load_checkpoint(checkpoint_path)

    model_dict = model.state_dict()
    pretrained_dict = {}
    pretrained_layer_names = set()
    for checkpoint_layer_name, checkpoint_tensor in checkpoint.state_dict().items():

        if checkpoint_layer_name in model_dict:
            if model_dict[checkpoint_layer_name].shape == checkpoint_tensor.shape:
                pretrained_dict[checkpoint_layer_name] = checkpoint_tensor

            else:
                print(
                    "Warning: not using weight for {} because shape does not match. Model: {}; weights: {}".format(
                        checkpoint_layer_name, list(model_dict[checkpoint_layer_name].shape),
                        list(checkpoint_tensor.shape)))

        pretrained_layer_names.add(checkpoint_layer_name)
    model_layer_names = set(model_dict.keys())

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


def init_logger(tensorboard_log_dir, unique_run_name):
    """
    Init the tensorboard logger. Requires the directory and the run tag (this will be visible in TB).
    """
    # Init logger
    if tensorboard_log_dir is not None:
        if unique_run_name is not None:
            log_dir = os.path.join(tensorboard_log_dir, unique_run_name)
        else:
            log_dir = tensorboard_log_dir
        tb_logger = torch.utils.tensorboard.writer.SummaryWriter(log_dir=log_dir)
        print('Tensorboard logging to:', log_dir)
    else:
        tb_logger = None

    return tb_logger

def set_random_seed(random_seed=None, cudnn_tuner=True):
    if random_seed is not None:
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    else:
        random.seed(int(time.time()))
        torch.manual_seed(int(time.time()))
    if cudnn_tuner:
        torch.backends.cudnn.benchmark     = True  ##uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -
        torch.backends.cudnn.enabled       = True
        # torch.backends.cudnn.deterministic = True  #HK TODO in the future may be used

def prepare_clargs_parser():
    parser = ArgumentParser()

    parser.add_argument('--random-seed', type=int, default=None, metavar='INT', help="random seed to set")

    parser.add_argument('--tensorboard-log-dir', default=None, metavar='PATH', help="where to write logfiles serving"
                                                                                    "as input to tensorboard")

    parser.add_argument('--nlayers-finetune', type=int, default=3, metavar='INT',
                        help="how many layers to finetune with last classifier included")

    parser.add_argument('--single-channel-input-grey', action='store_true',
                        help='Reduction to grey level based image')

    parser.add_argument('--gpu-id', type=int, default=3, metavar='INT',
                        help="cuda device id ")

    parser.add_argument('--loss-opt', type=str, default='CE', choices=['CE', 'FL', 'margin'], metavar='STRING',
                        help='')
#simillarity matching
    parser.add_argument('--margin-loss-margin', type=float, default=0.3, help='')

    parser.add_argument('--margin-loss-type', type=str, default='contrastive_cosine', help='')

    parser.add_argument('--margin-multi-task-weight', nargs='+', type=float, default=[1.0, 1.0], # --margin-multi-task-weight 1.0 0.5
                        help='1st value for 1st task (CE) 2nd value for 2nd task such as margin based')

    parser.add_argument('--loss-smooth', action='store_true',
                        help='Label smoothing https://arxiv.org/pdf/1512.00567.pdf')

    parser.add_argument('--loss-smooth-weight', type=float, default=0.1,
                        help="weight for the one hot in the CE")

    parser.add_argument('--gamma-aug-corr', type=float, default=1.0,
                        help="Gamma correction for augmrntation : 1.0 no correction")

    parser.add_argument('--hsv-hue-jitter', type=float, default=0,
                        help="HSV color jitter [-0.5 : 0.5] ")

    parser.add_argument('--hsv-sat-jitter', type=float, default=0,
                        help="HSV color jitter [-0.5 : 0.5] ")

    parser.add_argument('--gamma-inv-corr', action='store_true',
                        help=""
                             " "
                             ".")
    parser.add_argument('--monte-carlo-dropout-bayes', action='store_true',
                        help=" Monte carlo bayes approximation of model uncertianty"
                             " "
                             ".")

    parser.add_argument("--select-best-model-in-val", type=bool, default=True, help="If true.")

    parser.add_argument("--colour-norm", action='store_true', help="If true.")



    parser.add_argument('--run-tag', type=str, default=None, metavar='STRING',
                        help="name to be used for the run in tensorboard")
    parser.add_argument('--num-workers', type=int, default=12, metavar='INT',
                        help="number of worker processes for the data loader")
    parser.add_argument('--weighted-loss', type=bool, default=False, help="Weighing the loss for class imbalance")
    parser.add_argument('--save-onnx', action='store_true', help="save model to ONNX after training")
    parser.add_argument('--evaluation-frequency', type=int, default=1000, metavar='INT',
                        help="number of iterations until next evaluation")
    parser.add_argument('--deterministic-cudnn', action='store_true',
                        help="CuDNN is by default non-deterministic for the sake of faster "
                             "processing. This flag makes training deterministic again at the "
                             "price of slower runtime.")
    parser.add_argument('--crossvalidation-fold', type=str, default=None, metavar='STRING',
                        help="if a cross-validation set is inspected, the user can set a "
                             "train-test separation to a given fold. If not given, "
                             "everything is shown as training set.")

    parser.add_argument('--result-dir', type=str, default=None, metavar='PATH',
                        help="if given, all output of the training will be in this folder. "
                             "The exception is the tensorboard logs.")
    parser.add_argument('--print-on-eval', default=1, type=int,
                        help="If True, printing of the results will happen on every evaluation during training."
                             "If False, printing is only done during the last evaluation. Default False."
                             "Whatever is given, results are still logged to tensorboard.")
    parser.add_argument('--devlog', action='store_true',
                        help='Add "devlog" level to the logs directories before last dir. '
                             'For example, if the log directory is defined as "/a/b/c" '
                             'the result folder layout would be "/a/b/devlog/c". '
                             'The flag affects "result_dir" and "tensorboard_log_dir" directories. ')
    parser.add_argument('--pretrained-weights-path', type=str, default=None,
                        help='If given, the training will start with a pre-trained model.')

    # parser.add_argument('--fine-tune-pretrained-model-plan', type=str, default=None, # TODO add json/yaml that tells what to freeze and what to add to the current NN
    #                     help='If given, the training will start with a pre-trained model.')

    parser.add_argument('--hue-norm-preprocess', action='store_true',
                            help='Preprocessing the Hue mean offset to 0.45')

    parser.add_argument('--hue-norm-preprocess-type', type=str, default='weighted_hue_by_sv', metavar='STRING',
                        help=".")

    parser.add_argument('--evaluate-testset', action='store_false',
                            help='Test with test set?') # it is a tricky switch if not set that evaluate-testset=True otherwise skip test set
    # gradient_clip_value
    parser.add_argument('--gradient-clip-value', type=float, default=100,
                        help="gradient-clip-value")

    parser.add_argument('--debug-verbose', action='store_true',
                        help='Verbosity')

    parser.add_argument("--lr-decay-policy", default='StepLR', choices=['StepLR', 'reduced_platue', 'cosine_anneal_warm_restart',
                                                                        'cosine_anneal', 'warmup_over_cosine_anneal',
                                                                        'warmup_linear_over_cosine_anneal',
                                                                        'cyclic_lr'],
                                                     help="Print the statistics for the test or training set only,")

    parser.add_argument('--lr-decay-hyperparam', type=int, default=2000, metavar='INT',
                        help="like T_max ")

    parser.add_argument('--replace-relu', action='store_true',
                            help='REplaceRelu by Leaky relu')
#--handcrafted-features n_tiles
    # parser.add_argument('--handcrafted-features', '--list', nargs='+', help='name of hand crafted feature as appeared in the csv', required=False)
    #
    parser.add_argument('--balancing-sampling', action='store_true',
                        help='Weighted sampling')
    #
    parser.add_argument('--class-imb-weighted-loss', action='store_true',
                        help='Weighted the loss function by class imbalance')

    parser.add_argument('--tta', type=int, default=0, metavar='INT',
                        help="TTA ")

    parser.add_argument('--replace-val-with-test', action='store_true',
                        help='replace_val_with_test')


    return parser

def update_logs_dirs(args):
    def _update_path(path):
        p = Path(path)
        return str(Path.joinpath(p.parent, 'devlog', p.name))

    dargs = vars(args)
    if dargs.get('devlog', False):
        keys = ('result_dir', 'tensorboard_log_dir')
        dargs.update({log_dir:_update_path(dargs[log_dir]) for log_dir in keys if dargs.get(log_dir, None) != None})

    return args



######################################################################
# Visualize a few images
# ^^^^^^^^^^^^^^^^^^^^^^
# Let's visualize a few training images so as to understand the data
# augmentations.

def imshow(inp, title=None, show=True, img_fname = '', path=''):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)

    fig = plt.figure()

    # im, norm = imshow_norm(image, ax, origin='lower',
    #                        interval=MinMaxInterval(), stretch=SqrtStretch())
    # plt.imshow(inp, origin='lower', norm=norm)
    plt.imshow(inp)

    if title is not None:
        fig.suptitle(title, fontsize=8)
    if img_fname is not None:
        fig.savefig(os.path.join(path, img_fname + '.png'))
    plt.close()

    cv2.imshow('title', inp)
    cv2.imwrite(os.path.join(path, img_fname + 'cv2.png'), (inp * 255).astype(np.uint8))
    # pilimg = Image.fromarray((inp * 255).astype(np.uint8))
    # d = ImageDraw.Draw(pilimg)
    # d.text((10, 10), title)
    #
    # # pilimg.show(title=title)
    # pilimg.save(os.path.join(path, img_fname + 'PIL.png'))
    # pilimg.close()


    plt.pause(0.001)  # pause a bit so that plots are updated





######################################################################
# Visualizing the model predictions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Generic function to display predictions for a few images
#

######################################################################
# Finetuning the convnet
# ----------------------
#
# Load a pretrained model and reset final fully connected layer.
#
def main(args: list = None):
    parser = prepare_clargs_parser()

    add_clargs(parser, CLARG_SETS.SETUP)
    add_clargs(parser, CLARG_SETS.COMMON)
    add_clargs(parser, CLARG_SETS.DATASET)
    add_clargs(parser, CLARG_SETS.ARCHITECTURE)
    add_clargs(parser, CLARG_SETS.TRAINING)

    args = parser.parse_args(args)
    args = update_logs_dirs(args)
    args.focal_loss_gama = 2
    start_time = time.time()
    clock = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))
    print("Time/Date is {}".format(clock))
    args.clock = clock
    print(args.run_config_name)
    set_random_seed(args.random_seed)

    train_eval_model_phases = ['train', 'val']
    if args.train_by_all_data:
        args.evaluate_testset = False
        train_eval_model_phases = ['train']

    plot_decisions = False

    model_name = args.run_config_name  #"mobilenet_v2_256_win" #"mobilenet_v2" #"resnet" #"squeezenet"
    args.model_name = model_name
    run_config_name = model_name  #+ 'percentile filtering' # TODO set to  run_config.name   the name of the model config

    data_dir = args.database_root

    name = args.run_config_name
    unique_run_name = str(int(time.time()))
    print("unique run name : {}".format(unique_run_name))

    run_tag = args.run_tag if args.run_tag is not None else 'reg_run_{}_finetune'.format(name)

    device = torch.device("cuda:" + str(args.gpu_id) + "" if torch.cuda.is_available() else "cpu")

    # Number of classes in the dataset

    feature_extract = True
    args.dropblock_group = None
    if 'dropblock' in model_name.split('_'):
        args.dropblock_group = '4'
        print('dropblock_group was set')
    # Initialize the model for this run

    len_handcrafted_features = 0
    if args.handcrafted_features is not None:
        len_handcrafted_features = len(args.handcrafted_features)
        if len_handcrafted_features == 0 :
            raise ValueError("hand crafted feature was defines but w/o a type!! ")

    pooling_method = ['avg_pooling' if args.fine_tune_pretrained_model_plan == 'freeze_pretrained_add_nn_avg_pooling' else
                      'lp_mean_pooling' if args.fine_tune_pretrained_model_plan == 'freeze_pretrained_add_nn_lp' else
                       'gated_attention' if args.fine_tune_pretrained_model_plan == 'freeze_pretrained_add_gated_atten' else
                       'transformer_san' if args.fine_tune_pretrained_model_plan == 'freeze_pretrained_add_vit' else None][0] # SAN= Self Attention Network

    if args.pretrained_weights_path is None:
        pretrained_type = {'source': 'imagenet', 'path': None}
    else:
        pretrained_type = {'source': 'checkpoint', 'path': args.pretrained_weights_path}

    num_classes = [3 if model_name == 'mobilenet_v2_2FC_w256_fusion_3cls' else 2][0]

    if model_name == 'mobilenet_v2_2FC_w256_fusion_3cls' or model_name == 'mobilenet_v2_2FC_w256_fusion_2cls':
        args.classify_image_all_tiles = 1
    else:
        args.classify_image_all_tiles = 0


    model_ft, input_size = initialize_model(model_name, num_classes=num_classes, feature_extract=feature_extract,
                                            pretrained_type=pretrained_type, n_layers_finetune=args.nlayers_finetune,
                                            dropout=args.dropout, dropblock_group=args.dropblock_group,
                                            replace_relu=args.replace_relu,
                                            len_handcrafted_features=len_handcrafted_features,
                                            pooling_method=pooling_method,
                                            device=device, pooling_at_classifier=args.pooling_at_classifier,
                                            fc_sequential_type=args.fc_sequential_type,
                                            positional_embeddings=args.positional_embeddings,
                                            transformer_param_list=[int(x) for x in args.transformer_param_list])

    # validate args are not conflicting : only method named "_custom_forward_impl()" supports HCF by convention
    if not hasattr(model_ft, '_custom_forward_impl') and len_handcrafted_features > 0:
        raise ValueError("You asked for hand crafted feature but model doesn't support but mobilenet_v2_2FC_w256_nlyrs type is !!!!!")

    if 0:
        from torchsummary import summary
        device_id = 0
        summary(model_ft.to(device_id),list(((64, 3, 256, 256), (64, 3, 256, 256))))

    args.input_size = input_size

    train_data_filter = {}
    test_data_filter = {}
    if 1:
        train_data_filter = {}  # {} # 'greater': 150}
        test_data_filter = {}  # {} # 'greater': 150}
    else:
        train_data_filter = {'lessthan':0, 'greater': 150} # {} # 'greater': 150}
        # test_data_filter = {'lessthan':25, 'greater': 150} # {} # 'greater': 150}

    args.test_data_filter = test_data_filter
    args.train_data_filter = train_data_filter

    print(model_name)
    # print(model_ft)

    model_ft = model_ft.to(device)
    kwargs = dict()

    train_dataloader, val_dataloader, test_dataloader = prepare_dataloaders(args,
                                                        train_on_all=False, train_filter_percentile40=train_data_filter,
                                                        test_filter_percentile40=test_data_filter,
                                                        positional_embeddings=args.positional_embeddings, pooling_method=pooling_method,
                                                        overfitting_test=False, num_classes=num_classes, **kwargs)

    allocated_mem = torch.cuda.memory_allocated()
    print("memory allocated {} [Bytes]".format(allocated_mem))
    # memory_summary = torch.cuda.memory_summary(device=device)
    # print("memory summary {} [Bytes]".format(memory_summary))

    dataloaders = dict()
    dataset_sizes = dict()

    dataloaders['train'] = train_dataloader
    dataloaders['val'] = val_dataloader
    dataloaders['test'] = test_dataloader

    dataset_sizes['train']  = len(train_dataloader.dataset)
    dataset_sizes['val']    = len(val_dataloader.dataset)
    dataset_sizes['test']   = len(test_dataloader.dataset)

### L O S S
    if args.loss_opt == 'CE':
        weights = None
        if args.class_imb_weighted_loss:
            weights = torch.FloatTensor(train_dataloader.dataset.class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights) #reduction: str = 'mean' is implicit
    elif args.loss_opt == 'FL': #Focal loss
        if args.loss_smooth:
            raise ValueError('FL with loss smooth was not defined')
        criterion = FocalLoss(logits=True, alpha=1, gamma=args.focal_loss_gama)
#TODO add vizualization utilities  UMAP in https://github.com/lmcinnes/umap
    elif args.loss_opt == 'margin': #pair or triplet together with CE hence primary loss is CE and scondary pairwise simillarity
        criterion = nn.CrossEntropyLoss()
        if args.margin_loss_type == 'contrastive_lp':
            criterion_second = losses.ContrastiveLoss(pos_margin=0, neg_margin=args.margin_loss_margin) #+ embedding_regularizer=LpRegularizer() cause all embedding to be zero
            # if overfitting lower pos_margin increase neg margin on cosine
            # criterion_second = losses.ContrastiveLoss(pos_margin=0, neg_margin=args.margin_loss_margin)
            mining_func = miners.MultiSimilarityMiner(epsilon=args.margin_loss_margin/2) # changed to /2
        elif args.margin_loss_type == 'contrastive_cosine': #CosineSimilarity is an inverted metric (large values indicate higher similarity)
            criterion_second = losses.ContrastiveLoss(pos_margin=1, neg_margin=args.margin_loss_margin,  # inverted distance cosine similarity = 1-cosine_dist
                                                      distance=CosineSimilarity(), reducer=MeanReducer()) # miner will omitt th dist<low_th , reducer=ThresholdReducer(low=0, high=3)
                    # , embedding_regularizer=LpRegularizer() killed the ap and an loss to zero
                    #try reducer=MeanReducer()
            if 0: # check this out
                criterion_second = losses.ContrastiveLoss(pos_margin=1, neg_margin=0, distance=CosineSimilarity(), embedding_regularizer = LpRegularizer()) # inverted distance cosine similarity = 1-cosine_dist

            mining_func = miners.MultiSimilarityMiner(epsilon=args.margin_loss_margin/2)# changed to /2

        elif args.margin_loss_type == 'triplet_dist_lp_dist':
            criterion_second = losses.TripletMarginLoss(margin=args.margin_loss_margin, #0.05
                                                 swap=False,
                                                 smooth_loss=False, #Use the log-exp version of the triplet loss like in "In the defence of triplet loss"
                                                 triplets_per_anchor="all")
            # mining_func = miners.BatchHardMiner(pos_strategy="hard", neg_strategy="hard")
            mining_func = miners.MultiSimilarityMiner(epsilon=args.margin_loss_margin)
        elif args.margin_loss_type == 'triplet_dist_lp_dist_batch_hard':
            criterion_second = losses.TripletMarginLoss(margin=args.margin_loss_margin,  # 0.05
                                                        swap=False,
                                                        smooth_loss=False,
                                                        # Use the log-exp version of the triplet loss like in "In the defence of triplet loss"
                                                        triplets_per_anchor="all") # embedding_regularizer=LpRegularizer() makes the embeddings go to zero
            mining_func = miners.BatchHardMiner()
        elif args.margin_loss_type == 'triplet_dist_cosine_dist': #
            criterion_second = losses.TripletMarginLoss(distance=CosineSimilarity(),
                                                 #reducer=ThresholdReducer(high=0.3),
                                                 embedding_regularizer=LpRegularizer(),#L2 embeddings normalization
                                                 margin=args.margin_loss_margin)
        elif args.margin_loss_type == 'cosface_dist':
            criterion_second = losses.CosFaceLoss(num_classes=num_classes, embedding_size=-1,
                                           margin=args.margin_loss_margin, scale=64) # scale=64 from the paper margin=0.35
        else:
            raise ValueError('margin loss is un supported/defined')
    else:
        raise ValueError("Not imp yet")

    criterion_dict = {'criterion_prime': criterion, 'miner': [mining_func if 'mining_func' in globals() else None][0],
                      'secondary_loss_type': args.loss_opt,
                      'criterion_second': [criterion_second if 'criterion_second' in globals() else None][0],
                      'margin_multi_task_weight': args.margin_multi_task_weight,
                      'margin': [args.margin_loss_margin if args.margin_loss_margin else None][0]}
#### OPTIMISER
    # Observe that all parameters are being optimized
    if args.optimizer == 'adam':
        # optimizer_ft = torch.optim.Adam(model_ft.parameters(), lr=args.lr,
        #                                              weight_decay=args.weight_decay)
        # bug fixed by AdamW : it decouples the learning rate from the weight decay in contrast to adam() =>no need to adjust Wdecay if lr changes , however if batch size getting smaller 64->16 more wieght decays occurs !!
        optimizer_ft = torch.optim.AdamW(model_ft.parameters(), lr=args.lr, eps=args.adamw_epsilon,
                                                     weight_decay=args.weight_decay)

    elif args.optimizer == 'sgd':
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=args.lr,
                                 momentum=0.9, weight_decay=args.weight_decay)
    else:
        raise
#####  LR scheduler
    if args.lr_decay_base>1.0:
        raise ValueError("lr_decay_base should be less than 1 !!!")

    warmup_scheduler = None
    # convert from iterations to the basis of the training loop: EPOCH
    if args.npoch_decay_step == -1: # default means 1 epoch
        npoch_decay_step_epochs = 1
    else:
        if args.npoch_decay_step < 12:
            Warning("npoch_decay_step<10 it should be set in iterations R U sure !!!!")
        npoch_decay_step_epochs = int(args.npoch_decay_step/(len(train_dataloader.dataset)/args.batch_size) + 0.5)
        assert (npoch_decay_step_epochs != 0) # if database increases it may happen
        print("npoch_decay_step converted from iterations to Epochs {}".format(npoch_decay_step_epochs))

    if args.lr_decay_policy == 'StepLR':
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=npoch_decay_step_epochs, gamma=args.lr_decay_base)  #HK was :step_size=7
    elif args.lr_decay_policy == 'reduced_platue':
        exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.5, patience=0,
                                                          threshold=0.01, threshold_mode='rel', cooldown=0, min_lr=1e-05,
                                                          eps=1e-08, verbose=True)
    elif args.lr_decay_policy == 'cosine_anneal_warm_restart': #HK TODO not working well
        exp_lr_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer_ft,  #T_0 period of 1st wamup Number of iterations for the first restart  ;T_mult=1 increase T_0 each period
                                                                    T_0=args.lr_decay_hyperparam, T_mult=1, eta_min=args.lr/10, last_epoch=-1) #lr range test take max warmup/4 for CLR https://arxiv.org/abs/1803.09820s
    elif args.lr_decay_policy == 'cosine_anneal':
        exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer_ft, T_max=10, eta_min=0, last_epoch=-1)
    elif args.lr_decay_policy == 'warmup_over_cosine_anneal':
        num_steps = len(dataloaders['train']) * args.epochs
        exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer_ft, T_max=num_steps) #, eta_min=0, last_epoch=-1
        warmup_scheduler = warmup.UntunedLinearWarmup(optimizer_ft)
    elif args.lr_decay_policy == 'warmup_linear_over_cosine_anneal':
        num_steps = len(dataloaders['train']) * args.epochs # one haf cycle of cosine
        exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer_ft, T_max=num_steps, eta_min=args.lr/10) # lr range test take max warmup/4 for CLR https://arxiv.org/abs/1803.09820s #, eta_min=0, last_epoch=-1
        warmup_scheduler = warmup.LinearWarmup(optimizer_ft, warmup_period=args.lr_decay_hyperparam)
    elif args.lr_decay_policy == 'cyclic_lr': # batch wise
        num_steps = len(dataloaders['train']) * args.epochs
        exp_lr_scheduler = lr_scheduler.CyclicLR(optimizer=optimizer_ft, base_lr=args.lr*args.lr_decay_base, #=lr_max/4 usually or 1/10,1/20 for one cycle according to https://arxiv.org/pdf/1506.01186.pdf 1e-4:1e-2
                                                 max_lr=args.lr,
                                                 step_size_up=args.npoch_decay_step, step_size_down=None, mode='triangular', # args.npoch_decay_step should be in iterations
                                                 gamma=1.0, scale_fn=None, scale_mode='cycle', cycle_momentum=False, base_momentum=0.8,
                                                 max_momentum=0.9, last_epoch=-1)
    else:
        raise

    # print("Model {} Npoch {} Batch {} lr {} Wdecay {} LRdecay {} LRdecayNpoch {} ".format(model_name, args.epochs, args.batch_size, args.lr,
    #                                                           args.weight_decay, args.lr_decay_base, args.npoch_decay_step))

    print_arguments(args)

    result_dict = {'Model': model_name,
                    'data_dir': data_dir,
                    'unique_run_name': unique_run_name,
                    'trainset_support': dataset_sizes['train'],
                    'validation_support': dataset_sizes['val'],
                    'test_support': dataset_sizes['test']}

    result_dict.update(vars(args))

######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#class_imb_weighted_loss = args.class_imb_weighted_loss
# It should take around 15-25 min on CPU. On GPU though, it takes less than a
# minute.
#
    lr_scheduler_dict = {'exp_lr_scheduler': exp_lr_scheduler, 'lr_decay_policy': args.lr_decay_policy,
                         'warmup_scheduler': warmup_scheduler}
    tb_logger = init_logger(args.tensorboard_log_dir, unique_run_name=run_tag + '-' + run_config_name + '_' + unique_run_name)

    model_ft, result_tr_dict, _ = train_eval_model(model_ft, args, dataloaders, device, dataset_sizes,
                                   criterion_dict, optimizer_ft, lr_scheduler_dict,
                                   tb_logger, phases=train_eval_model_phases, num_epochs=args.epochs,
                                    select_best_model_in_val=args.select_best_model_in_val,
                                    gradient_clip_value=args.gradient_clip_value,
                                    debug=args.debug_verbose)

    # Save model state (for later fine tuning)
    state_file_name = 'saved_' + 'state_' + run_config_name + '___' + unique_run_name
    torch.save(model_ft.state_dict(), os.path.join(args.result_dir, state_file_name))
    save_checkpoint(path=os.path.join(args.result_dir, state_file_name + 'checkpoint'), epoch=args.epochs,
                    model=model_ft, optimizer=optimizer_ft, loss=0,
                    model_name=run_config_name)

    if (args.evaluate_testset):
        model_ft, result_test_dict, predictions_dict_test = train_eval_model(model_ft, args, dataloaders, device, dataset_sizes,
                                                                                criterion_dict, optimizer_ft, exp_lr_scheduler,
                                                                                tb_logger, ['test'], 1)
# due to the tkinter.TclError: couldn't connect to display "localhost:10.0" exception
#     if result_test_dict['conf_mat'] is not None:
#         pilimg = Image.fromarray(result_test_dict['conf_mat'])
#         fname = os.path.join(args.result_dir, run_config_name + '_' + unique_run_name + '_testset_conf_mat.png')
#         pilimg.save(fname)

    else:
        result_test_dict = {}
    if 0:
        import inference
        threshold = 0.18
        inference_split_csv_list = '/hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/EileenQuality_needtoRunblindFinder_4_outline/eileen_best_images_merged_list.csv'

        str_image_hsv_new = None
        if args.hue_norm_preprocess:
            str_image_hsv_new = '--hue-norm-preprocess'

        acc_inference_th, ap_inference_th = inference.main([args.run_config_name, '--model-path', str(os.path.join(args.result_dir, state_file_name)),
                        '--database-root', '/hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/EileenQuality_needtoRunblindFinder_4_outline/tiles',
                        '--dataset-split-csv',
                                                            inference_split_csv_list,
                        '--result-dir', '/hdd/hanoch/runmodels/img_quality/results', '--gpu-id', str(args.gpu_id),
                                           '--confidence-threshold', str(threshold), str_image_hsv_new])
        inference_dict = {'acc_inference_@th': acc_inference_th, 'th_inference' : threshold, 'inference_split_csv_list' :inference_split_csv_list, 'ap_inference_th': ap_inference_th}
        result_dict.update(inference_dict)

        # threshold = 0.5
        test_csv_list = args.dataset_split_csv
        acc_inference, ap_inference = inference.main([args.run_config_name, '--model-path', str(os.path.join(args.result_dir, state_file_name)),
                        '--database-root', '/hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data',
                        '--dataset-split-csv',
                                                      test_csv_list,
                        '--result-dir', '/hdd/hanoch/runmodels/img_quality/results', '--gpu-id', str(args.gpu_id),
                                        '--confidence-threshold', str(threshold), str_image_hsv_new])
        avg_inference = 0.5*(acc_inference + acc_inference_th)
        delta_percentage = 100*np.abs(1- acc_inference/acc_inference_th)
        inference_dict = {'acc_inference@th_testset': acc_inference, 'avg_inferences': avg_inference, 'delta_percentage': delta_percentage, 'ap_inference':ap_inference}
        result_dict.update(inference_dict)

    result_dict.update(result_tr_dict)
    result_dict.update(result_test_dict)
    result_dict.update({'time_elapsed': time.time() - start_time})
    #save exp summary result to csv
    df_result = pd.DataFrame.from_dict(list(result_dict.items()))
    df_result = df_result.transpose()
    df_result.to_csv(os.path.join(args.result_dir, run_tag + '-' + unique_run_name + '.csv'), index=False)
# TODO: in the future the threshold per tile will be calculated on end of training + call to fuse_blind_decision_based_on_tiles_prediction()
#   blind_quality_decision_by_fuse_tiles_soft_hard_prediction(tile_threshold=tile_threshold, method=method)
    model_metadata_dictionary = {
            'model_architecture': args.model_name,
            'model_unique_id' : unique_run_name,
            'confidence_threshold': round(0.85, 3),
            'voting_confidence_threshold': round(0.85, 3), # hard fusion/pooling : 2 options exist
            'average_pooling_threshold': round(0.9, 3), # soft fusion/pooling
            'average_pooling_threshold_loose_for_secondary_use': round(0.86, 3), # looser soft fusion/pooling thershold for the product upon taking images to the database
            'n_model_outputs': num_classes,
            'tta': 0,
            'normalize_rgb_mean': train_dataloader.dataset.normalize_rgb_mean,
            'normalize_rgb_std': train_dataloader.dataset.normalize_rgb_std,
            'model_input_resolution_w': args.input_size,
            'model_input_resolution_h': args.input_size
    }

    if "handcrafted_features_metadata_dictionary" in train_dataloader.dataset:
        model_metadata_dictionary.update({'hcf': train_dataloader.dataset.handcrafted_features_metadata_dictionary})

    meta_data_file_name = run_config_name + '_' + unique_run_name + '.json'
    with open(os.path.join(args.result_dir, meta_data_file_name), 'w') as f:
        json.dump(model_metadata_dictionary, f)


    print("Finished in %.2fs" % (time.time() - start_time))
    max_allocated = torch.cuda.max_memory_allocated()
    print("Max memory allocated to the GPU {} [Bytes]".format(max_allocated))
    return
    ######################################################################
#

######################################################################
# Further Learning
# -----------------
#
# If you would like to learn more about the applications of transfer learning,
# checkout our `Quantized Transfer Learning for Computer Vision Tutorial <https://pytorch.org/tutorials/intermediate/quantized_transfer_learning_tutorial.html>`_.

if __name__ == '__main__':
    main()

#HK@@ TODO 1.common.py control seed effect with cudnn determinism
#  TODO Mixup augmentation 2.	GEM Generalized mean pooling 3. load all images in advance :self.pre_load_images
# TODO: [ make sure random generator determined each run.  : class imbalance train bad/good 87Kvs. 19.5K ;
#TODO add more FC to the model for more dropout/ uncertuianty estimation
# Consider slim MobileNet models TODO: tiles with overlap, tiles of 1024^2
# add         roc_plot() in train_eval_model()
# TODO : label smoothing - debug the implementation
# TODO : MC-dropout, REview the images wit hCNN decoisions overfitting over few cleaned examples like Eileen shared : compare tiles to the ones Eileen sent ,
# TODO : test/validation set maybe poor ? clean data, train 2 last layers,
#   Consider more augmentations, normalization is by ImageNet?! , change tile to 1024x1024]
# TODO : Blakish/saturated tiles =>remove from training but has to be in the test? remove from test as well by the same prefiltering rule
# add them to the training but with bad label though they are part of good quality blind
#TODO: Inference : In the inference make sure that tiles with that criterion of percentile would be classified automatically as bad quality
"""
 
python -u quality_bin_cls_cnn.py --tensorboard-log-dir /hdd/hanoch/runmodels/img_quality/results
nohup python -u quality_bin_cls_cnn.py --tensorboard-log-dir /hdd/hanoch/runmodels/img_quality/results --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/train_test_tile.csv --batch-size 64 & tail -f nohup.out

--tensorboard-log-dir /hdd/hanoch/runmodels/img_quality/results --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data --result-dir /hdd/hanoch/runmodels/img_quality/results --batch-size 64 --epochs 24 --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/train_test_val_quality_merged_tile_tile.csv --gpu-id 3
--tensorboard-log-dir /hdd/hanoch/runmodels/img_quality/results --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/train_test_tile.csv --batch-size 64 --result-dir /hdd/hanoch/runmodels/img_quality/results

--tensorboard-log-dir /hdd/hanoch/runmodels/img_quality/results --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/train_test_tile.csv --result-dir /hdd/hanoch/runmodels/img_quality/results --epochs 1 --batch-size 64
tile 64
                                mobilenet_v2_cls_head_256 --tensorboard-log-dir /hdd/hanoch/runmodels/img_quality/results --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/train_test_val_quality_tile_filtered_eileen_good_bad_val.csv --result-dir /hdd/hanoch/runmodels/img_quality/results --lr-decay-base 0.98 --hsv-hue-jitter 0.2 --hsv-sat-jitter 0.2 --dropout 0.5 --weight-decay 0.001 --epochs 1 --lr 0.001  --batch-size 128  --nlayers-finetune 6 --gpu-id 1
python -u quality_bin_cls_cnn.py mobilenet_v2_256_win_n_lyrs --tensorboard-log-dir /hdd/hanoch/runmodels/img_quality/results --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data_64 --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data_64/train_eileenVal_test_tile_64.csv --result-dir /hdd/hanoch/runmodels/img_quality/results/tile_64 --batch-size 64 --lr-decay-base 0.98 --gpu-id 0 --nlayers-finetune 10 --hsv-hue-jitter 0.2 --epochs 1 --loss-smooth >> ./scriptFiltGoodImghue-norm.py.log </dev/null 2>&1

python -u $CNN_WORKDIR/quality_bin_cls_cnn.py mobilenet_v2_256_win_n_lyrs --tensorboard-log-dir /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/lr_test_gamma_aug --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/file_quality_tile_eileen_good_bad_val_bad_9_20_avg_pool_filt_conf_filt_no_edges_trn_tst_fitjar_ntiles.csv --result-dir /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/lr_test_gamma_aug --gpu-id 3 --nlayers-finetune 20 --hsv-hue-jitter 0.1 --hsv-sat-jitter 0.2 --gradient-clip-value 5 --batch-size 64 --lr 0.001 --balancing-sampling --lr-decay-base 0.70 --dropout 0.2 --epochs 20 --npoch-decay-step 1700 --gamma-aug-corr 0.75 >> ./lr_test_gamma_aug.py.log </dev/null 2>&1

MIL - avg poling
mobilenet_v2_2FC_w256_fusion_3cls --tensorboard-log-dir /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/test --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/file_quality_tile_eileen_good_bad_val_bad_9_20_avg_pool_filt_conf_filt_no_edges_trn_tst_fitjar_hcf.csv --result-dir /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/test --gpu-id 1 --nlayers-finetune 3 --hsv-hue-jitter 0.1 --hsv-sat-jitter 0.2 --gradient-clip-value 5 --batch-size 2 --balancing-sampling --epochs 16 --lr 0.01 --pretrained-weights-path /hdd/hanoch/runmodels/img_quality/results/backup_models/saved_state_mobilenet_v2_256_win_n_lyrs___1609764290 --fine-tune-pretrained-model-plan freeze_pretrained_add_nn_avg_pooling --classify-image-all-tiles --num-workers 1
MIL 3 classes
mobilenet_v2_2FC_w256_fusion_3cls --tensorboard-log-dir /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/test --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data --dataset-split-csv file_quality_tile_eileen_good_bad_val_bad_9_20_avg_pool_filt_conf_filt_no_edges_trn_tst_fitjar_hcf_marginal.csv --result-dir /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/test --gpu-id 1 --nlayers-finetune 3 --hsv-hue-jitter 0.1 --hsv-sat-jitter 0.2 --gradient-clip-value 5 --batch-size 8 --balancing-sampling --epochs 16 --lr 0.01 --pretrained-weights-path /hdd/hanoch/runmodels/img_quality/results/backup_models/saved_state_mobilenet_v2_256_win_n_lyrs___1609764290 --fine-tune-pretrained-model-plan freeze_pretrained_add_gated_atten --classify-image-all-tiles --num-workers 1
mobilenet_v2_2FC_w256_fusion_3cls --tensorboard-log-dir /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/gated_atten_reannotate_marg --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/annotations_quality_tile_pool_filt_conf_filt_no_edges_trn_tst_fitjar_marginal_reannot2_marginals_tile_pos.csv --result-dir /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/gated_atten_reannotate_marg --pretrained-weights-path /hdd/hanoch/runmodels/img_quality/results/backup_models/saved_state_mobilenet_v2_256_win_n_lyrs___1609764290 --fine-tune-pretrained-model-plan freeze_pretrained_add_gated_atten --classify-image-all-tiles --gpu-id 1 --nlayers-finetune 3 --hsv-hue-jitter 0.1 --hsv-sat-jitter 0.2 --gradient-clip-value 5 --balancing-sampling --lr 0.005 --epochs 30 --npoch-decay-step 410  --lr-decay-base 0.70 --batch-size 32 --class-imb-weighted-loss >> ./gated_atten_reannotate_marg.py.log </dev/null 2>&1

mobilenet_v2_2FC_w256_fusion_3cls --tensorboard-log-dir /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/gated_atten_reannotate_marg --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/annotations_quality_tile_pool_filt_conf_filt_no_edges_trn_tst_fitjar_marginal_reannot2_marginals_tile_pos.csv --result-dir /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/gated_atten_reannotate_marg --pretrained-weights-path /hdd/hanoch/runmodels/img_quality/results/backup_models/saved_state_mobilenet_v2_256_win_n_lyrs___1609764290 --fine-tune-pretrained-model-plan freeze_pretrained_add_gated_atten --classify-image-all-tiles --gpu-id 3 --nlayers-finetune 3 --hsv-hue-jitter 0.1 --hsv-sat-jitter 0.2 --gradient-clip-value 5 --balancing-sampling --lr 0.005 --epochs 30 --npoch-decay-step 410  --lr-decay-base 0.70 --batch-size 4 --positional-embeddings raster_bitmap_8_8  
HCF n_tiles
mobilenet_v2_2FC_w256_nlyrs --tensorboard-log-dir /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/canny_edges_sum_50_70 --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/file_quality_tile_eileen_good_bad_val_bad_9_20_avg_pool_filt_conf_filt_no_edges_trn_tst_fitjar_ntiles.csv --result-dir /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/canny_edges_sum_50_70 --nlayers-finetune 22 --hsv-hue-jitter 0.1 --hsv-sat-jitter 0.2 --gradient-clip-value 5 --batch-size 32 --epochs 20 --lr 0.0001 --lr-decay-base 0.7 --dropout 0.2 --balancing-sampling  --handcrafted-features n_tiles --gpu-id 3

Pos embedding
mobilenet_v2_2FC_w256_fusion_3cls --tensorboard-log-dir /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/gated_atten_reannotate_marg --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/annotations_quality_tile_pool_filt_conf_filt_no_edges_trn_tst_fitjar_marginal_reannot2_marginals_tile_pos.csv --result-dir /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/gated_atten_reannotate_marg --pretrained-weights-path /hdd/hanoch/runmodels/img_quality/results/backup_models/saved_state_mobilenet_v2_256_win_n_lyrs___1609764290 --fine-tune-pretrained-model-plan freeze_pretrained_add_gated_atten --classify-image-all-tiles --gpu-id 3 --nlayers-finetune 3 --hsv-hue-jitter 0.1 --hsv-sat-jitter 0.2 --gradient-clip-value 5 --balancing-sampling --lr 0.005 --epochs 30 --npoch-decay-step 410  --lr-decay-base 0.70 --batch-size 4 --positional-embeddings raster_bitmap_8_8 --gpu-id 3
mobilenet_v2_2FC_w256_fusion_3cls --tensorboard-log-dir /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/gated_atten_reannotate_marg --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/annotations_quality_tile_pool_filt_conf_filt_no_edges_trn_tst_fitjar_marginal_reannot2_marginals_tile_pos.csv --result-dir /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/gated_atten_reannotate_marg --pretrained-weights-path /hdd/hanoch/runmodels/img_quality/results/backup_models/saved_state_mobilenet_v2_256_win_n_lyrs___1609764290 --fine-tune-pretrained-model-plan freeze_pretrained_add_gated_atten --classify-image-all-tiles --gpu-id 3 --nlayers-finetune 3 --hsv-hue-jitter 0.1 --hsv-sat-jitter 0.2 --gradient-clip-value 5 --balancing-sampling --lr 0.005 --epochs 30 --npoch-decay-step 410  --lr-decay-base 0.70 --batch-size 4 --positional-embeddings raster_bitmap_8_8_mlp --gpu-id 3
# new dataset
mobilenet_v2_2FC_w256_fusion_3cls --tensorboard-log-dir /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/raster_bitmap_8_8_mlp_155marg_fix_gradtrue --database-root /hdd/hanoch/data/database/blind_quality/quality_based_all_annotations_24_6/png/tiles --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/stratified_train_val_test_set_penn_dependant_data_list_7818.csv --result-dir /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/test --pretrained-weights-path /hdd/hanoch/runmodels/img_quality/results/backup_models/saved_state_mobilenet_v2_256_win_n_lyrs___1609764290 --fine-tune-pretrained-model-plan freeze_pretrained_add_gated_atten --classify-image-all-tiles --gpu-id 1 --nlayers-finetune 3 --hsv-hue-jitter 0.1 --hsv-sat-jitter 0.2 --gradient-clip-value 5 --lr 0.005 --epochs 30 --npoch-decay-step 1700 --balancing-sampling --lr-decay-base 0.70 --batch-size 32
SAN
mobilenet_v2_2FC_w256_fusion_3cls --tensorboard-log-dir /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/raster_bitmap_8_8_mlp_155marg_fix_gradtrue --database-root /hdd/hanoch/data/database/blind_quality/quality_based_all_annotations_24_6/png/tiles --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/stratified_train_val_test_set_penn_dependant_data_list_7818.csv --result-dir /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/test --pretrained-weights-path /hdd/hanoch/runmodels/img_quality/results/backup_models/saved_state_mobilenet_v2_256_win_n_lyrs___1609764290 --fine-tune-pretrained-model-plan transformer_san --classify-image-all-tiles --gpu-id 1 --hsv-hue-jitter 0.1 --hsv-sat-jitter 0.2 --gradient-clip-value 5 --lr 0.005 --epochs 30 --npoch-decay-step 1700 --balancing-sampling --lr-decay-base 0.70 --batch-size 32 --transformer-chunk-len 32 
transformer_san
mobilenet_v2_2FC_w256_fusion_3cls --tensorboard-log-dir /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/transformer --database-root /hdd/hanoch/data/database/blind_quality/quality_based_all_annotations_24_6/png/tiles --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/stratified_train_val_test_set_penn_dependant_data_list_7818.csv --result-dir /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/transformer --pretrained-weights-path /hdd/hanoch/runmodels/img_quality/results/backup_models/saved_state_mobilenet_v2_256_win_n_lyrs___1609764290 --fine-tune-pretrained-model-plan freeze_pretrained_add_vit --classify-image-all-tiles --gpu-id 0 --nlayers-finetune 3 --hsv-hue-jitter 0.1 --hsv-sat-jitter 0.2 --gradient-clip-value 5 --balancing-sampling --lr 0.001 --epochs 30 --npoch-decay-step 4000  --lr-decay-base 0.70 --batch-size 48 --lr-decay-policy warmup_linear_over_cosine_anneal --lr-decay-hyperparam 4000 --dropout 0.1

Overfitting test : no balance batch;  no regularization + subdir for every class (--dedicated-subdir-per-class) --batch-size 8
mobilenet_v2_2FC_w256_fusion_3cls --tensorboard-log-dir /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/transformer --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/annotations_quality_3class_tile_pos_overfit.csv --result-dir /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/transformer --pretrained-weights-path /hdd/hanoch/runmodels/img_quality/results/backup_models/saved_state_mobilenet_v2_256_win_n_lyrs___1609764290 --fine-tune-pretrained-model-plan freeze_pretrained_add_vit --classify-image-all-tiles --gpu-id 0 --nlayers-finetune 3 --hsv-hue-jitter 0.1 --hsv-sat-jitter 0.2 --gradient-clip-value 5 --lr 0.001 --epochs 30 --npoch-decay-step 4000  --lr-decay-base 0.70 --batch-size 8 --lr-decay-hyperparam 4000 --dedicated-subdir-per-class
"""