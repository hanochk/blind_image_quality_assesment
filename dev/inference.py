from __future__ import print_function, division
import os
from argparse import ArgumentParser

from dev.configuration import add_clargs, CLARG_SETS
from dev.evaluation import evaluate_model_on_dataset, grad_cam_model_on_dataset
from dev.dataset import class_labels
from dev.models import initialize_model
from dev.dataset import prepare_test_dataloader
from dev.utillities.files_manipulations.training_manipulation import filter_low_conf_high_loss_examples_from_dataframe

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
import time
import pickle


def calc_monte_carlo_dropout(bayes_uncertian_dict, test_df, all_targets,
                             result_dir, monte_carlo_n, unique_run_name):
    from PIL import Image

    accs = []
    for y_p in bayes_uncertian_dict['all_mc_predictions']:
        acc = accuracy_score(all_targets, y_p.argmax(axis=1))
        accs.append(acc)

    plt.figure()
    plt.hist(accs)
    # MC ensamble prediction avg over all ensamble softmax =>avg of ensamble softmax
    mc_ensemble_pred = bayes_uncertian_dict['all_mc_predictions'].mean(axis=0).argmax(axis=1)
    ensemble_acc = accuracy_score(all_targets, mc_ensemble_pred)

    plt.axvline(x=ensemble_acc, color="b")
    plt.title(
        "MC-ensemble accuracy(mean(softmax)): {:.1%} vs. histogram N={}".format(ensemble_acc, monte_carlo_n))
    plt.savefig(os.path.join(result_dir,
                             unique_run_name + '_acc_hist_mc_dropout_n_' + str(monte_carlo_n) + '.png'),
                format="png")

    # plt.show()  # cause exception in running
    print("Highest std {:.3f} out of {} examples".format(bayes_uncertian_dict['all_unct'][:, 1].max(),
                                                         bayes_uncertian_dict['all_unct'][:, 1].shape))

    print("MC accuracy: {:.1%}".format(sum(accs) / len(accs)))

    # go over the ensamble predictions find the most uncertian by mean
    max_means = []
    preds = []
    max_vars = []
    for idx in range(bayes_uncertian_dict['all_mc_predictions'].shape[1]):
        px = np.array([p[idx] for p in bayes_uncertian_dict['all_mc_predictions']])
        preds.append(px.mean(axis=0).argmax())
        max_means.append(px.mean(axis=0).max())
        max_vars.append(px.std(axis=0)[px.mean(axis=0).argmax()])

    print("Uncertian examples by the model {} mean of ensamble ".format((np.array(max_means)).argsort()[:10]))
    # dataloader shuffle = False hence index are aligned
    for ex in (np.array(max_means)).argsort()[:4]:
        plt.figure()
        file_low_conf = test_df['full_file_name'].iloc[ex]
        print("fname {}, mean {}".format(file_low_conf, max_means[ex].__format__('.3f')))
        img = Image.open(file_low_conf)
        img = img.convert('RGB')
        plt.imshow(img)
        plt.savefig(os.path.join(result_dir,
                                 unique_run_name + '_worst_tile_by_avg_' + str(
                                     max_means[ex].__format__('.3f')) + '_in_ensamble_no_' + str(
                                     ex) + ' _mc_dropout_n_' + str(
                                     monte_carlo_n) + '.png'), format="png")
        # plt.show()
    # std based : the greater the std (sorted in acsending that -large is the minimal)

    print("Uncertian examples by the model {} std of ensamble per image ".format((-np.array(max_vars)).argsort()[:10]))
    for ex in (-np.array(max_vars)).argsort()[:4]:
        plt.figure()
        file_low_conf = test_df['full_file_name'].iloc[ex]
        print("fname {}, std {}".format(file_low_conf, max_vars[ex].__format__('.3f')))
        img = Image.open(file_low_conf)
        img = img.convert('RGB')
        plt.imshow(img)
        plt.savefig(os.path.join(result_dir,
                                 unique_run_name + '_worst_tile_by_std_' + str(
                                     max_vars[ex].__format__('.3f')) + '_in_ensamble_no_' + str(
                                     ex) + ' _mc_dropout_n_' + str(
                                     monte_carlo_n) + '.png'), format="png")
        # plt.show()


def set_padding_mode_for_torch_compatibility(model):
    """
    Pytorch 1.1 fails to load pre-1.1 models because it expects the newly added padding_mode attribute.
    This function fixes the problem by adding the missing attribute with the default value.
    """
    modules = model.modules()
    for layer in modules:
        if isinstance(layer, nn.Conv2d):
            setattr(layer, 'padding_mode', 'zeros')


def load_model(model_path, *args, **kwargs):
    chkpoint = torch.load(model_path, *args, **kwargs)

    if torch.__version__ == '1.1.0':
        set_padding_mode_for_torch_compatibility(chkpoint)

    return chkpoint


def roc_plot(labels, predictions, positive_label, save_dir, thresholds_every=5, unique_id=''):
    # roc_auc_score assumes positive label is 0 namely FINGER_IDX=0 or equivalently positive_label = 1
    # os.environ['DISPLAY'] = str('localhost:10.0')
    # qt bug ???
    os.environ['QT_XKB_CONFIG_ROOT'] = '/usr/share/X11/xkb/'
    assert positive_label == 1

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels, predictions,
                                            pos_label=positive_label)

    auc = sklearn.metrics.roc_auc_score(labels, predictions) # TODO consider replace with metrics.auc(fpr, tpr) since it has the label built in implicit
    print("AUC: {}".format(auc))
    granularity_percentage = 1. / labels.shape[0] *100
    lw = 2
    n_labels = len(labels)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f) support = %3d' % (auc, n_labels))

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("ROC model {} (gran={:.2e}[%])".format(unique_id, granularity_percentage))
    plt.legend(loc="lower right")

    # plot some thresholds
    thresholdsLength = len(thresholds) #- 1
    thresholds_every = int(thresholdsLength/thresholds_every)
    thresholds = thresholds[1:] #  `thresholds[0]` represents no instances being predicted and is arbitrarily set to `max(y_score) + 1`. https://github.com/scikit-learn/scikit-learn/commit/4d9a67f77787ffe9955187865f9b95e19286f069
    thresholdsLength = thresholdsLength - 1 #changed the threshold vector henc echange the len

    colorMap = plt.get_cmap('jet', thresholdsLength)
    for i in range(0, thresholdsLength, thresholds_every):
        threshold_value_with_max_four_decimals = thresholds[i].__format__('.3f')
        plt.text(fpr[i] - 0.03, tpr[i] + 0.005, threshold_value_with_max_four_decimals, fontdict={'size': 15},
                 color=colorMap(i / thresholdsLength))

    filename = unique_id + 'roc_curve.png'
    plt.savefig(os.path.join(save_dir, filename), format="png")

def p_r_plot_multi_class(all_targets, all_predictions, save_dir, thresholds_every_in=5, unique_id=None):
    # Precision recall  assumes positive label is 0 namely FINGER_IDX=0 or equivalently positive_label = 1
    # os.environ['DISPLAY'] = str('localhost:10.0')
    # qt bug ???
    os.environ['QT_XKB_CONFIG_ROOT'] = '/usr/share/X11/xkb/'

    all_targets_one_hot = label_binarize(all_targets, classes=[0, 1, 2])
    precision = dict()
    recall = dict()
    thresholds_ap = dict()
    average_precision = dict()
    for i in range(all_predictions.shape[1]):
        precision[i], recall[i], thresholds_ap[i] = precision_recall_curve(all_targets_one_hot[:, i],
                                                            all_predictions[:, i])
        average_precision[i] = average_precision_score(all_targets_one_hot[:, i], all_predictions[:, i])

        granularity_percentage = 1. / all_targets.shape[0] *100
        lw = 2
        n_labels = all_targets.shape[0]

        plt.figure()
        plt.plot(recall[i], precision[i], color='darkorange',
                 lw=lw, label='AP (area = %0.3f) support = %3d' % (average_precision[i], n_labels))
        plt.plot([1, 0], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.grid()
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title("PR (Macro) 1vs.all :{} model {} (gran={:.2e}[%])".format(i, unique_id, granularity_percentage))
        plt.legend(loc="lower right")

        # plot some thresholds
        thresholdsLength = len(thresholds_ap[i]) #- 1
        thresholds_every = max(int(thresholdsLength/thresholds_every_in), 1)

        thresholds = thresholds_ap[i][1:] #  `thresholds[0]` represents no instances being predicted and is arbitrarily set to `max(y_score) + 1`. https://github.com/scikit-learn/scikit-learn/commit/4d9a67f77787ffe9955187865f9b95e19286f069
        thresholdsLength = thresholdsLength - 1 #changed the threshold vector henc echange the len

        colorMap = plt.get_cmap('jet', thresholdsLength)
        precision_cls = precision[i]
        recall_cls = recall[i]
        for ind in range(0, thresholdsLength, thresholds_every):
            threshold_value_with_max_four_decimals = thresholds[ind].__format__('.3f')
            plt.text(recall_cls[ind] - 0.03, precision_cls[ind] + 0.005, threshold_value_with_max_four_decimals, fontdict={'size': 15},
                     color=colorMap(ind / thresholdsLength))

        filename = unique_id + 'p_r_curve_class_' + str(i) + '.png'
        plt.savefig(os.path.join(save_dir, filename), format="png")

    if 0:
        # A "micro-average": quantifying score on all classes jointly
        precision["micro"], recall["micro"], thresholds_micro = precision_recall_curve(all_targets_one_hot.ravel(),
                                                                        all_predictions.ravel())

        average_precision["micro"] = average_precision_score(all_targets_one_hot, all_predictions,
                                                             average="micro")

        plt.step(recall['micro'], precision['micro'], where='post')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.plot([1, 0], color='navy', lw=lw, linestyle='--')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title(
            'Average precision score, micro-averaged over all classes: AP={0:0.2f}'.format(average_precision["micro"]))

        # plot some thresholds
        thresholdsLength = len(thresholds_micro)  # - 1
        thresholds_every = max(int(thresholdsLength / thresholds_every_in), 1)

        thresholds = thresholds_micro[1:]  # `thresholds[0]` represents no instances being predicted and is arbitrarily set to `max(y_score) + 1`. https://github.com/scikit-learn/scikit-learn/commit/4d9a67f77787ffe9955187865f9b95e19286f069
        thresholdsLength = thresholdsLength - 1  # changed the threshold vector henc echange the len

        colorMap = plt.get_cmap('jet', thresholdsLength)
        for ind in range(0, thresholdsLength, thresholds_every):
            threshold_value_with_max_four_decimals = thresholds[i].__format__('.3f')
            plt.text(recall["micro"][ind] - 0.03, precision["micro"][ind] + 0.005, threshold_value_with_max_four_decimals, fontdict={'size': 15},
                     color=colorMap(ind / thresholdsLength))

        filename = unique_id + 'p_r_micro_curve.png'
        plt.savefig(os.path.join(save_dir, filename), format="png")


def p_r_plot(labels, predictions, positive_label, save_dir, thresholds_every=5, unique_id=None):
    # Precision recall  assumes positive label is 0 namely FINGER_IDX=0 or equivalently positive_label = 1
    # os.environ['DISPLAY'] = str('localhost:10.0')
    # qt bug ???
    os.environ['QT_XKB_CONFIG_ROOT'] = '/usr/share/X11/xkb/'
    assert positive_label == 1

    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(labels, predictions,
                                                                pos_label=positive_label)

    ap = sklearn.metrics.average_precision_score(labels, predictions,
                                                                pos_label=positive_label)

    print("AP : {}".format(ap))
    # auc = sklearn.metrics.roc_auc_score(labels, predictions)
    granularity_percentage = 1. / labels.shape[0] *100
    lw = 2
    n_labels = len(labels)

    plt.figure()
    plt.plot(recall, precision, color='darkorange',
             lw=lw, label='AP (area = %0.3f) support = %3d' % (ap, n_labels))
    plt.plot([1, 0], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid()
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title("PR model {} (gran={:.2e}[%])".format(unique_id, granularity_percentage))
    plt.legend(loc="lower right")

    # plot some thresholds
    thresholdsLength = len(thresholds) #- 1
    thresholds_every = max(int(thresholdsLength/thresholds_every), 1)

    thresholds = thresholds[1:] #  `thresholds[0]` represents no instances being predicted and is arbitrarily set to `max(y_score) + 1`. https://github.com/scikit-learn/scikit-learn/commit/4d9a67f77787ffe9955187865f9b95e19286f069
    thresholdsLength = thresholdsLength - 1 #changed the threshold vector henc echange the len

    colorMap = plt.get_cmap('jet', thresholdsLength)
    for i in range(0, thresholdsLength, thresholds_every):
        threshold_value_with_max_four_decimals = thresholds[i].__format__('.3f')
        plt.text(recall[i] - 0.03, precision[i] + 0.005, threshold_value_with_max_four_decimals, fontdict={'size': 15},
                 color=colorMap(i / thresholdsLength))

    filename = unique_id + 'p_r_curve.png'
    plt.savefig(os.path.join(save_dir, filename), format="png")


def plot_tsne(all_targets, all_features, path):
    from sklearn.manifold import TSNE
    n_components = 2
    perplexity = 30
    tsne_data = TSNE(n_components=2, perplexity=perplexity).fit_transform(all_features)
    tsne_data.shape
    plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=all_targets, s=1)
    plt.title("TSNE of classes {} N={} p={}".format(np.unique(all_targets), all_targets.shape[0], perplexity))
    plt.savefig(os.path.join(path, 'tsne_' 'p_' + str(perplexity) + '.png'))


def prepare_clargs_parser():
    parser = ArgumentParser()

    parser.add_argument("--model-path", type=str, required=True, metavar='PATH',
                                        help="path to the neural network model to use")
    parser.add_argument("--filter-by-train-val-test", default='test', choices=['train', 'val', 'test'],
                                                     help="Print the statistics for the test or training set only,"
                                                          " not for both. Optional.")
    parser.add_argument("--result-dir", type=str, required=False, default=None, metavar='PATH',
                                        help="if specified, the resulting csv and plots will be saved here")

    parser.add_argument("--confidence-threshold", type=float, required=False, default=None, metavar='FLOAT',
                                                  help="If specified, a confidence threshold will be used for "
                                                       "evaluation, instead of printing the whole FAR-FRR curve. "
                                                       "This enables higher-detailed analysis.")
    parser.add_argument("--synthetic-test-flip", choices=['none', 'horizontal', 'vertical', 'inversion'],
                                                 required=False, default='none',
                                                 help="Measure robustness of model by observing average change in "
                                                      "class probability after flipping test images.")
    parser.add_argument("--crossvalidation-fold", default=None, type=str,
                                                  help="If a cross-validation set is inspected, the user can set a "
                                                       "train-test separation to a given fold. If not given, "
                                                       "everything is shown as training set.")

    parser.add_argument('--num-workers', type=int, default=12, metavar='INT',
                        help="number of worker processes for the data loader")

    parser.add_argument('--gpu-id', type=int, default=0, metavar='INT',
                        help="cuda device id ")

    parser.add_argument('--single-channel-input-grey', action='store_true',
                        help='Reduction to grey level based image')

    parser.add_argument('--monte-carlo-dropout-bayes', action='store_true',
                        help=" Monte carlo bayes approximation of model uncertianty"
                             " "
                             ".")
    parser.add_argument('--monte-carlo-dropout-iter', type=int, default=100, metavar='INT',
                        help="number of times running inference ")

    parser.add_argument('--plot-roc', action='store_true',
                        help='due to bug in localhost only on demand rather than all the time')

    parser.add_argument('--tta', type=int, default=0, metavar='INT',
                        help="TTA ")

    parser.add_argument('--hue-norm-preprocess', action='store_true',
                            help='Preprocessing the Hue mean offset to 0.45')

    parser.add_argument('--hue-norm-preprocess-type', type=str, default='weighted_hue_by_sv', metavar='STRING',
                        help=".")

    parser.add_argument('--extract-fetures', action='store_true',
                        help='yield the features out of the model')

    parser.add_argument('--pre-load-images', action='store_true',
                        help='load all data in the init')

    parser.add_argument('--grad-cam-plot', action='store_true',
                        help='load all data in the init')


    return parser



def main(args: list = None):

    parser = prepare_clargs_parser()
    add_clargs(parser, CLARG_SETS.COMMON)
    add_clargs(parser, CLARG_SETS.SETUP)
    add_clargs(parser, CLARG_SETS.ARCHITECTURE)

    args = parser.parse_args(args)
    start_time = time.time()

    model_name = args.run_config_name  #"mobilenet_v2_256_win" #"mobilenet_v2" #"resnet" #"squeezenet"
    num_classes = [3 if model_name == 'mobilenet_v2_2FC_w256_fusion_3cls' else 2][0]
    if model_name == 'mobilenet_v2_2FC_w256_fusion_3cls' or model_name == 'mobilenet_v2_2FC_w256_fusion_2cls':
        args.classify_image_all_tiles = 1
    else:
        args.classify_image_all_tiles = 0

    feature_extract = True
    device = torch.device("cuda:" + str(args.gpu_id) + "" if torch.cuda.is_available() else "cpu")
    # import os  TODO try this option
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"  $ or device id

    len_handcrafted_features = 0
    if args.handcrafted_features is not None:
        len_handcrafted_features = len(args.handcrafted_features)
        if len_handcrafted_features == 0 :
            raise ValueError("hand crafted feature was defines but w/o a type!! ")

    pooling_method = ['avg_pooling' if args.fine_tune_pretrained_model_plan == 'freeze_pretrained_add_nn_avg_pooling' else
                      'lp_mean_pooling' if args.fine_tune_pretrained_model_plan == 'freeze_pretrained_add_nn_lp' else
                       'gated_attention' if args.fine_tune_pretrained_model_plan == 'freeze_pretrained_add_gated_atten' else None][0]

    args.batch_size = 128 #64*2

    if pooling_method is not None:
        args.batch_size = 32 # limited by the fact that each example contains many tiles

    loading_model_without_predefiend_pattern = False # THE OPTION TO SKIP THE SETUP MODEL PATTERN WILL BE OMITTED

    if loading_model_without_predefiend_pattern:
        model = torch.load(args.model_path)
    else:
        model, input_size = initialize_model(model_name, num_classes, feature_extract,
                                             pretrained_type={'source': 'imagenet', 'path': None},
                                             n_layers_finetune=0,
                                             dropout=0.0, dropblock_group='4', replace_relu=False,
                                             len_handcrafted_features=len_handcrafted_features,
                                             pooling_method=pooling_method,
                                             device=device,  pooling_at_classifier=args.pooling_at_classifier,
                                             fc_sequential_type=args.fc_sequential_type,
                                             positional_embeddings=args.positional_embeddings,
                                             debug=True)

        # validate args are not conflicting : only method named "_custom_forward_impl()" supports HCF by convention
        if not hasattr(model, '_custom_forward_impl') and len_handcrafted_features>0:
            raise ValueError("You asked for hand crafted feature but model doesn't support !!!!!")
        # checkpoint = load_model(args.model_path, device)
        checkpoint = torch.load(args.model_path, map_location=device)
        #validate swiches in args are aligned
        position_embed_ref = [key for key in checkpoint if key.startswith('position') == True]
        if position_embed_ref ==[] and args.positional_embeddings is not None or position_embed_ref !=[] and args.positional_embeddings == None:
            raise Exception("Positional embeddings requirements and model are conflicting")
        model.load_state_dict(checkpoint) # for state dict the model pattern is needed before hand

    args.input_size = input_size
    args.loss_smooth = False

    if torch.cuda.is_available():
        model = model.to(device)

    # args.num_workers = 12
    test_data_filter = dict()
    test_data_filter = {}
        # test_data_filter = {'lessthan': 25, 'greater': 150}
    args.test_data_filter = test_data_filter
    if (test_data_filter):
        print("!!!!!!!!!!  filtering the testset")

    args.gamma_aug_corr = {}


    csv_path = args.dataset_split_csv
    if os.path.splitext(csv_path)[1] == '.csv':
        dataframe = pd.read_csv(csv_path)
    else:# hk@@ TODO : pickle
        raise
    if args.filter_by_train_val_test == 'test':
        test_df = dataframe.loc[dataframe['train_or_test'] == args.filter_by_train_val_test]
    elif args.filter_by_train_val_test == 'train':
        test_df = dataframe.loc[dataframe['train_or_test'] == args.filter_by_train_val_test]
        test_df = test_df.loc[test_df['val'] == 0]
    elif args.filter_by_train_val_test == 'val': # val==1 is for sure validation else it may be train or test
        test_df = dataframe.loc[dataframe['val'] == 1]

    assert len(test_df) != 0

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)


    if 'in_outline_tiles_id' in test_df:
        test_df = test_df.drop(['in_outline_tiles_id'], axis=1)

    kwargs = dict()
    monte_carlo_dropout_bayes = args.monte_carlo_dropout_bayes if 'monte_carlo_dropout_bayes' in args else None
    kwargs['monte_carlo_dropout_bayes'] = monte_carlo_dropout_bayes
    kwargs['extract_features'] = args.extract_fetures
    kwargs['hue_norm_preprocess'] = args.hue_norm_preprocess
    kwargs['num_classes'] = num_classes

    if kwargs['monte_carlo_dropout_bayes']:
        kwargs['monte_carlo_n'] = args.monte_carlo_dropout_iter
        #Iteartions are accomodated by dropout rate in variety of models if FC [1024 to 2] and dropout=0.2 => 205/1024 are toggled wjich is 1024!/205!/819! too huge
        args.batch_size = 64 # not chunking the GPU mem 10.9G

    unique_run_name = args.model_path.split('___')[-1] + '_' + args.filter_by_train_val_test
    # Test Time Augmentation
    if args.tta:
        all_targets_tta = []
        all_predictions_cls2_tta = []

        tta_dict_permute = {0: [0], 1: [0, 1], 2: [0, 1, 2], 3: [0, 1, 2, 3], 4: [0, 1, 2, 3, 4],
                            5: [0, 1, 2, 3, 4, 5], 6: [0, 1, 2, 3, 4, 5, 6], 7: [0, 1, 2, 3, 4, 5, 6, 7],
                            8: [0, 1, 2, 3, 4, 5, 6, 7, 8], 9: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}
        # for ind in test_df.index:
        #     df_test_one_entry = pd.DataFrame(test_df.loc[ind])
        #     test_dataloader = prepare_test_dataloader(args, df_test_one_entry,
        #                                               test_filter_percentile40=test_data_filter)
        for tta_iter in tta_dict_permute[args.tta]: #[::-1] temp patch to start with the augmented version
            print("Running TTA permutation {}".format(tta_iter))
            test_dataloader = prepare_test_dataloader(args, test_df, test_filter_percentile40=test_data_filter,
                                                      tta=tta_iter,pooling_method=pooling_method, **kwargs)

            return_dict = evaluate_model_on_dataset(model,
                                            test_dataloader, device, loss_fn=None, max_number_of_batches=None,
                                            do_softmax=True,
                                            positional_embeddings=args.positional_embeddings, **kwargs)

            all_targets = return_dict['all_targets']
            all_predictions = return_dict['all_predictions']
            if args.get_image_name_item:
                image_names = return_dict['all_tile_id']

            all_targets_tta.append(all_targets)
            all_predictions_cls2_tta.append(all_predictions[:, class_labels['good']])

        o = [np.vstack(pp.reshape(1, -1)) for pp in all_predictions_cls2_tta]
        all_predictions_cls2_tta_f = np.concatenate(o)

        # auc = sklearn.metrics.roc_auc_score(y_true=all_targets, y_score=np.mean(
        #     [all_predictions_cls2_tta_f[4, :], all_predictions_cls2_tta_f[0, :]], axis=0))
        tta_res = dict()
        tta_res['all_predictions'] = all_predictions_cls2_tta_f
        tta_res['all_targets'] = all_targets
        with open(os.path.join(args.result_dir, unique_run_name + '_tta.pkl'), 'wb') as f:
            pickle.dump(tta_res, f)
        ap = sklearn.metrics.average_precision_score(y_true=all_targets, y_score=np.mean(
                                                        [all_predictions_cls2_tta_f[4, :], all_predictions_cls2_tta_f[0, :]], axis=0),
                                                        pos_label=1)

        print(ap)
    else:
        # _, _, test_dataloader = prepare_dataloaders(args, test_filter_percentile40=test_data_filter)
        test_dataloader = prepare_test_dataloader(args, test_df, test_filter_percentile40=test_data_filter,
                                                  positional_embeddings=args.positional_embeddings,
                                                  pooling_method=pooling_method, **kwargs)

        return_dict = evaluate_model_on_dataset(model,
                                        test_dataloader, device, loss_fn=None, max_number_of_batches=None,
                                        do_softmax=True, positional_embeddings=args.positional_embeddings,
                                         **kwargs)

        all_targets = return_dict['all_targets']
        all_predictions = return_dict['all_predictions']
        # loss = return_dict['loss']
        all_features = return_dict['all_features']
        if args.get_image_name_item:
            image_names = return_dict['all_tile_id']
        if args.classify_image_all_tiles:
            all_atten_weithgs = return_dict['all_atten_weithgs']

        filter_low_conf_examples = False
        if filter_low_conf_examples:
            det_good_cls_image_names_below_th, \
            det_bad_cls_image_names_below_th, \
            res_df = filter_low_conf_high_loss_examples_from_dataframe(all_predictions,
                                                                       all_targets, image_names, test_df, th=0.2) #test_df

        if args.grad_cam_plot:
            # # find last layer
            # for idx, p in enumerate(model_ft.children()):
            #     print(idx, p)
            # # for seq in range(idx):
            global_norm_of_heatmap = None #0.008 # None
            kwargs_grad_cam = {'result_dir': args.result_dir, 'input_size': args.input_size, 'path': args.result_dir,
                               'global_norm_of_heatmap': global_norm_of_heatmap,
                               'normalize_rgb_mean': test_dataloader.dataset.normalize_rgb_mean,
                               'normalize_rgb_std': test_dataloader.dataset.normalize_rgb_std}

            grad_cam_model_on_dataset(model, test_df, class_label='good', device=device,
                                      all_targets=all_targets, all_predictions=all_predictions,
                                      **kwargs_grad_cam)


        if args.extract_fetures:
            plot_tsne(all_targets, all_features, args.result_dir) # not debugged yet

            output = dict()
            output['all_targets'] = all_targets
            output['all_predictions'] = all_predictions
            output['all_features'] = all_features
            with open(os.path.join(args.result_dir, args.dataset_split_csv.split('/')[-1].split('.')[0] + '_cnn_features.pkl'), 'wb') as f:
                pickle.dump(output, f)

            # [np.percentile(all_features[x, :], 50) for x in range(11)]
        # TODO add --arg for control the dropout in inference
        if kwargs['monte_carlo_dropout_bayes']:
            bayes_uncertian_dict = return_dict['return_dict']
            calc_monte_carlo_dropout(bayes_uncertian_dict, test_df, all_targets,
                                args.result_dir, monte_carlo_n=kwargs['monte_carlo_n'],
                                unique_run_name=unique_run_name)

    thresh = ''
    if all_predictions.shape[1] == 3:  # multi-class
        acc = -1
        # auc = sklearn.metrics.roc_auc_score(y_true=all_targets, y_score=all_predictions, multi_class='ovr')
        # print("AUC 1vs.all {}", auc)
        conf_mat = sklearn.metrics.confusion_matrix(y_true=all_targets, y_pred=all_predictions.argmax(axis=1))
        print("[bad, good, marginal] ")
        print(conf_mat)
        conf_mat_recall = conf_mat / conf_mat.sum(axis=1).reshape(-1, 1)
        conf_mat_precision = conf_mat/conf_mat.sum(axis=0).reshape(1, -1)
        if 1:
            prob_given_cls0 = all_predictions[all_targets == 0, :][np.where(all_predictions[all_targets == 0, :].argmax(axis=1) != 0)[0]]
#Macro precision/recall non sensitive to skewed data

        all_targets_one_hot = label_binarize(all_targets, classes=[0, 1, 2])
        precision = dict()
        recall = dict()
        average_precision = dict()
        roc_one_vs_all = dict()
        for i in range(all_predictions.shape[1]):
            # precision[i], recall[i], _ = precision_recall_curve(all_targets_one_hot[:, i],
            #                                                     all_predictions[:, i])
            average_precision[i] = average_precision_score(all_targets_one_hot[:, i], all_predictions[:, i])
            average_precision[i] = average_precision[i].__format__('.4f')
            roc_one_vs_all[i] = sklearn.metrics.roc_auc_score(y_true=all_targets_one_hot[:, i], y_score=all_predictions[:, i],
                                                              multi_class='ovr')

        # A "micro-average": quantifying score on all classes jointly
        average_precision["micro"] = average_precision_score(all_targets_one_hot, all_predictions,
                                                             average="micro")
        print('Average precision score, micro-averaged over all classes: {0:0.2f}'
              .format(average_precision["micro"]))

        # average_precision["micro"] = average_precision["micro"].__format__('.4f')
        ap = average_precision[1] # class good vs. all
        print("One vs. all AP", ap)
        roc = roc_one_vs_all[1]
        print("One vs. all ROC", roc)

#threshold base of class good vs. all
        if hasattr(args, 'confidence_threshold') and args.confidence_threshold is not None:
            dets = all_predictions[:, class_labels['good']] > args.confidence_threshold #0.98393804
            good_det = dets.astype('int') == all_targets_one_hot[:, class_labels['good']]
            thresh = args.confidence_threshold
            acc = np.sum(good_det.astype('int')) / all_targets.shape[0]
            print("Multi-class acc = {} @confidence threshold={}".format(acc, args.confidence_threshold))

        result_dict = dict(conf_mat=conf_mat,
                           conf_mat_recall=conf_mat_recall,
                           conf_mat_precision=conf_mat_precision,
                           ap_macro_micro=average_precision,
                           acc_class_good_vs_all=acc)
        if args.plot_roc:
            roc_plot(all_targets_one_hot[:, 1], all_predictions[:, 1], positive_label=1, save_dir=args.result_dir,
                     unique_id='multi_class_' + unique_run_name)
            p_r_plot_multi_class(all_targets, all_predictions,
                                 save_dir=args.result_dir, unique_id=unique_run_name)

    elif all_predictions.shape[1] == 2:
        if hasattr(args, 'confidence_threshold') and args.confidence_threshold is not None:
            dets = all_predictions[:, class_labels['good']] > args.confidence_threshold
            good_det = all_targets == dets.astype('int')
            thresh = args.confidence_threshold
        else: # who ever has the greater confidence i.e threshold=0.5
            good_det = all_targets == all_predictions.argmax(axis=1)

        acc = np.sum(good_det.astype('int')) / all_targets.shape[0]

        if args.get_image_name_item:
            df_image_names_results = pd.DataFrame(columns=['file_name', 'is_tp'])
            df_image_names_results['file_name'] = image_names
            df_image_names_results['is_tp'] = dets
            df_image_names_results.to_csv(os.path.join(args.result_dir, 'inference-file_names' + unique_run_name + '_th_' + str(thresh) + '.csv'), index=False)
        # see if there are examples from more than one class ?
        ap = -1
        recall_at_th = -1
        if np.unique(all_targets).shape[0]>1:
            auc = sklearn.metrics.roc_auc_score(y_true=all_targets, y_score=all_predictions[:, 1])
            fpr, tpr, thresholds = sklearn.metrics.roc_curve(all_targets, all_predictions[:, 1],
                                                             pos_label=1)
            if hasattr(args, 'confidence_threshold') and args.confidence_threshold is not None:

                if args.confidence_threshold < min(thresholds):
                    print("Warning!!! this threshold below the minimal confidence for that class(softmax output)")
                else:
                    ind_th = np.where(thresholds<args.confidence_threshold)[0][0]
                    recall_at_th = tpr[ind_th]

                    print("***********  Acc {} @ threshold {} recall {} support {} model {}".format(acc.__format__('.3f'),
                                                                                          args.confidence_threshold,
                                                                                          recall_at_th.__format__('.3f'),
                                                                                          len(test_dataloader.dataset),
                                                                                          unique_run_name))

            #  @ th=0.5     FN of the "good" class
            fn = np.sum([all_predictions.argmax(axis=1) == 0][0].astype('int') * [all_targets == 1][0].astype('int'))
            fp = np.sum([all_predictions.argmax(axis=1) == 1][0].astype('int') * [all_targets == 0][0].astype('int'))

            tp = np.sum([all_predictions.argmax(axis=1) == 1][0].astype('int') * [all_targets == 1][0].astype('int'))

            acc_op5 = tp/all_predictions.shape[0]

            precision = [tp / (tp + fp) if (tp + fp) else -1][0]
            recall = [tp / (tp + fn) if (tp + fn) else -1][0]


            if args.plot_roc: # patch till it resolved TODO: add switch to arg parse for printing ROC
                roc_plot(all_targets, all_predictions[:, 1], positive_label=1, save_dir=args.result_dir, unique_id=unique_run_name)
                p_r_plot(all_targets, all_predictions[:, 1], positive_label=1, save_dir=args.result_dir, unique_id=unique_run_name)

            ap = sklearn.metrics.average_precision_score(y_true=all_targets, y_score=all_predictions[:, 1],
                                                         pos_label=1)
            print('********* Test:  Acc: {:.4f} [th=0.5] auc: {:.3f} ap: {:.3f}  precision {:.3f}: recall {:.3f} support : {:d} *******'.format(
                    acc_op5, auc, ap, precision, recall, len(test_dataloader.dataset)))

            result_dict = dict(recall_at_th=recall_at_th.__format__('.3f'),
                               acc=acc.__format__('.4f'),
                               auc_test=auc.__format__('.4f'),
                               ap_test=ap.__format__('.4f'),
                               precision_test=precision.__format__('.4f'),
                               recall_test=recall.__format__('.4f'))
        else:
            print("***********  Acc {} @ threshold {} support {} model {}".format(acc.__format__('.3f'),
                                                                                  args.confidence_threshold,
                                                                                  len(test_dataloader.dataset),
                                                                                  unique_run_name))

            result_dict = dict(acc=acc.__format__('.4f'))

    args_dict = vars(args)
    args_dict.update(result_dict)
    args_dict.update()

    df_result = pd.DataFrame.from_dict(list(args_dict.items()))
    df_result = df_result.transpose()
    df_result.to_csv(os.path.join(args.result_dir, 'inference-' + unique_run_name + '_th_' + str(thresh) + '.csv'), index=False)
    # print("Finished 1 in %.2fs" % (time.time() - start_time))

    arguments_str = '\n'.join(["{}: {}".format(key, args_dict[key]) for key in sorted(args_dict)])
    print(arguments_str + '\n')
    return acc, ap

if __name__ == "__main__":
    import sys

    main(sys.argv[1:])  # to be called by outside myModule.main(['arg1', 'arg2', 'arg3'])
    # main()

"""
Single channel infernce
--model-path /hdd/hanoch/runmodels/img_quality/results/saved_state_mobilenet_v2_256_win_n_lyrs___1593516086 --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/EileenQuality_needtoRunblindFinder_4_outline/tiles --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/EileenQuality_needtoRunblindFinder_4_outline/eileen_best_images_merged_list.csv --result-dir /hdd/hanoch/runmodels/img_quality/results --confidence-threshold 0.18 --single-channel-input-grey

To limit leakage to other GPUs or run with GPU#0
CUDA_VISIBLE_DEVICES=3 nohup python -u inference.py --model-path /hdd/hanoch/runmodels/img_quality/results/saved_state_mobilenet_v2_256_win_3_lyrsgamma_aug_corr___1592752657 --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/EileenQuality_needtoRunblindFinder_4_outline/tiles --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/EileenQuality_needtoRunblindFinder_4_outline/eileen_best_images_merged_list.csv --result-dir /hdd/hanoch/runmodels/img_quality/results --confidence-threshold 0.18 --gpu-id 0 --monte-carlo-dropout-bayes --monte-carlo-dropout-iter 128 > ./inference.log </dev/null 2>&1 & tail -f ./inference.log


# runing over the val set  + monte carlo  
CUDA_VISIBLE_DEVICES=3 nohup python -u inference.py --model-path /hdd/hanoch/runmodels/img_quality/results/saved_state_mobilenet_v2_256_win_n_lyrs___1593440243  --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/train_test_val_quality_merged_tile_tile.csv --result-dir /hdd/hanoch/runmodels/img_quality/results --confidence-threshold 0.18 --gpu-id 0 --monte-carlo-dropout-bayes --monte-carlo-dropout-iter 128 --filter-by-train-val-test val > ./inference.log </dev/null 2>&1 & tail -f ./inference.log

#last setup
--model-path /hdd/hanoch/runmodels/img_quality/results/saved_state_mobilenet_v2_256_win_n_lyrs___1595238994  --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/train_test_val_quality_merged_tile_tile.csv --result-dir /hdd/hanoch/runmodels/img_quality/results --confidence-threshold 0.18 --gpu-id 0

Eileen set
CUDA_VISIBLE_DEVICES=3 nohup python -u inference.py mobilenet_v2_256_win_n_lyrs --model-path /hdd/hanoch/runmodels/img_quality/results/saved_state_mobilenet_v2_256_win_n_lyrs___1595238994  --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/EileenQuality_needtoRunblindFinder_4_outline/tiles  --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/EileenQuality_needtoRunblindFinder_4_outline/eileen_best_images_merged_list.csv --result-dir /hdd/hanoch/runmodels/img_quality/results --confidence-threshold 0.18 --gpu-id 0



Eileen small good set
CUDA_VISIBLE_DEVICES=3 nohup python -u inference.py mobilenet_v2_256_win_n_lyrs --model-path /hdd/hanoch/runmodels/img_quality/results/saved_state_mobilenet_v2_256_win_n_lyrs___1596031838  --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/EileenQuality_needtoRunblindFinder_4_outline/tiles  --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/EileenQuality_needtoRunblindFinder_4_outline/eileen_best_images_merged_list.csv --result-dir /hdd/hanoch/runmodels/img_quality/results --confidence-threshold 0.18 --gpu-id 0  > ./inference.log </dev/null 2>&1 & tail -f ./inference.log

test set: just change the attribute of the "--filter-by-train-val-test" to val
CUDA_VISIBLE_DEVICES=0 nohup python -u inference.py ConvNet11 --model-path /hdd/hanoch/runmodels/img_quality/results/saved_state_ConvNet11___1596733309 --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data  --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/train_test_val_quality_tile_filtered_eileen_good_bad_val.csv --result-dir /hdd/hanoch/runmodels/img_quality/results --filter-by-train-val-test test --confidence-threshold 0.073 --gpu-id 0 --plot-roc > ./inference.log </dev/null 2>&1 & tail -f ./inference.log

val set (eileen all set -Eillen bad+good july set) : just change the attribute of the "--filter-by-train-val-test" to test
CUDA_VISIBLE_DEVICES=0 nohup python -u inference.py ConvNet11 --model-path /hdd/hanoch/runmodels/img_quality/results/saved_state_ConvNet11___1596733309 --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data  --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/train_test_val_quality_tile_filtered_eileen_good_bad_val.csv --result-dir /hdd/hanoch/runmodels/img_quality/results --filter-by-train-val-test val --confidence-threshold 0.073 --gpu-id 0 --plot-roc > ./inference.log </dev/null 2>&1 & tail -f ./inference.log

CUDA_VISIBLE_DEVICES=0 nohup python -u inference.py mobilenet_v2_256_win_n_lyrs --model-path /hdd/hanoch/runmodels/img_quality/results/saved_state_mobilenet_v2_256_win_n_lyrs___1596521630  --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data  --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/train_test_val_quality_tile_filtered_eileen_good_bad_val.csv --result-dir /hdd/hanoch/runmodels/img_quality/results --filter-by-train-val-test val --confidence-threshold 0.21 --gpu-id 0 --plot-roc > ./inference.log </dev/null 2>&1 & tail -f ./inference.log

CUDA_VISIBLE_DEVICES=0 nohup python -u inference.py mobilenet_v2_256_win_n_lyrs --model-path /hdd/hanoch/runmodels/img_quality/results/saved_state_mobilenet_v2_256_win_n_lyrs___1593440243  --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data  --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/train_test_val_quality_tile_filtered_eileen_good_bad_val.csv --result-dir /hdd/hanoch/runmodels/img_quality/results --filter-by-train-val-test test --confidence-threshold 0.21 --gpu-id 0 --plot-roc > ./inference.log </dev/null 2>&1 & tail -f ./inference.log

domain shift - extract features:tSNE + extract features
mobilenet_v2_256_win_n_lyrs --model-path /hdd/hanoch/runmodels/img_quality/results/saved_state_mobilenet_v2_256_win_n_lyrs___1593440243  --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_domain_shift  --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_domain_shift/blind_low_conf.csv --result-dir /hdd/hanoch/runmodels/img_quality/results --filter-by-train-val-test test --confidence-threshold 0.21 --gpu-id 0 --plot-roc --extract-fetures
mobilenet_v2_256_win_n_lyrs --model-path /hdd/hanoch/runmodels/img_quality/results/saved_state_mobilenet_v2_256_win_n_lyrs___1593440243  --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data  --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/train_test_val_quality_tile_filtered_eileen_good_bad_val.csv --result-dir /hdd/hanoch/runmodels/img_quality/results --filter-by-train-val-test test --confidence-threshold 0.21 --gpu-id 0 --extract-fetures
mobilenet_v2_256_win_n_lyrs --model-path /hdd/hanoch/runmodels/img_quality/results/saved_state_mobilenet_v2_256_win_n_lyrs___1593440243  --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_domain_shift  --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_domain_shift/blind_low_conf.csv --result-dir /hdd/hanoch/runmodels/img_quality/results --filter-by-train-val-test test --confidence-threshold 0.21 --gpu-id 0 --plot-roc --extract-fetures

--
CUDA_VISIBLE_DEVICES=0 nohup python -u inference.py mobilenet_v2_256_win_n_lyrs --model-path /hdd/hanoch/runmodels/img_quality/results/saved_state_mobilenet_v2_256_win_n_lyrs___1602505062  --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data  --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/train_test_val_quality_tile_filtered_eileen_good_bad_val.csv --result-dir /hdd/hanoch/runmodels/img_quality/results --filter-by-train-val-test test --confidence-threshold 0.21 --gpu-id 0 --plot-roc --hue-norm-preprocess

mobilenet_v2_256_win_n_lyrs --model-path /hdd/hanoch/runmodels/img_quality/results/saved_state_mobilenet_v2_256_win_n_lyrs___1593440243  --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data  --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/train_test_val_quality_tile_filtered_eileen_good_bad_val.csv --result-dir /hdd/hanoch/runmodels/img_quality/results --filter-by-train-val-test test --confidence-threshold 0.21 --gpu-id 0 --plot-roc --hue-norm-preprocess

Tile 64
CUDA_VISIBLE_DEVICES=0 nohup python -u inference.py mobilenet_v2_64_win_n_lyrs --model-path /hdd/hanoch/runmodels/img_quality/results/tile_64/saved_state_mobilenet_v2_64_win_n_lyrs___1603311582 --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data_64 --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data_64/post_train_eileenVal_test_tile_64.csv --result-dir /hdd/hanoch/runmodels/img_quality/results --filter-by-train-val-test test --gpu-id 0 --plot-roc & tail -f nohup.out

single channel
CUDA_VISIBLE_DEVICES=0 nohup python -u inference.py mobilenet_v2_256_win_n_lyrs --model-path /hdd/hanoch/runmodels/img_quality/results/saved_state_mobilenet_v2_256_win_n_lyrs___1603714813  --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data  --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/file_quality_tile_eileen_good_bad_val_bad_9_2020_avg_pool_filt_weighted_hue_byimage.csv --result-dir /hdd/hanoch/runmodels/img_quality/results --filter-by-train-val-test train --confidence-threshold 0.21 --gpu-id 0 --plot-roc
squeezenet:
CUDA_VISIBLE_DEVICES=0 nohup python -u inference.py squeezenet --model-path /hdd/hanoch/runmodels/img_quality/results/squeezenet/saved_state_squeezenet___1604250479 --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data  --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/file_quality_tile_eileen_good_bad_val_bad_9_2020_avg_pool_filt_weighted_hue_byimage.csv --result-dir /hdd/hanoch/runmodels/img_quality/results --filter-by-train-val-test train --confidence-threshold 0.21 --gpu-id 0 --plot-roc
hue_norm
CUDA_VISIBLE_DEVICES=0 nohup python -u inference.py mobilenet_v2_256_win_n_lyrs --model-path /hdd/hanoch/runmodels/img_quality/results/hue_weighted_norm/reg/saved_state_mobilenet_v2_256_win_n_lyrs___1604507681 --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data  --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/file_quality_tile_eileen_good_bad_val_bad_9_2020_avg_pool_filt_weighted_hue_byimage.csv --result-dir /hdd/hanoch/runmodels/img_quality/results --filter-by-train-val-test train --confidence-threshold 0.21 --hue-norm-preprocess --gpu-id 0 --plot-roc

CUDA_VISIBLE_DEVICES=0 nohup python -u inference.py mobilenet_v2_256_win_n_lyrs --model-path /hdd/hanoch/runmodels/img_quality/results/saved_state_mobilenet_v2_256_win_n_lyrs___1593440243                    --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data  --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/train_test_val_quality_tile_filtered_eileen_good_bad_val.csv --result-dir /hdd/hanoch/runmodels/img_quality/results --filter-by-train-val-test test --confidence-threshold 0.21 --gpu-id 0 --plot-roc > ./inference.log </dev/null 2>&1 & tail -f ./inference.log

Cages/tiles:
CUDA_VISIBLE_DEVICES=0 nohup python -u inference.py mobilenet_v2_256_win_n_lyrs --model-path /hdd/hanoch/runmodels/img_quality/results/saved_state_mobilenet_v2_256_win_n_lyrs___1593440243  --database-root /hdd/hanoch/data/cages_holdout/annotated_images_eileen/tile_data  --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/Cages_n5_tiles_eileen_blind_test_tiles.csv --result-dir /hdd/hanoch/runmodels/img_quality/results/cages_tiles --filter-by-train-val-test test --confidence-threshold 0.21 --gpu-id 0 --plot-roc
TODO
CUDA_VISIBLE_DEVICES=0 nohup python -u inference.py mobilenet_v2_256_win_n_lyrs --model-path /hdd/hanoch/runmodels/img_quality/results/hue_weighted_norm/reg/saved_state_mobilenet_v2_256_win_n_lyrs___1604507681 --database-root /hdd/hanoch/data/cages_holdout/annotated_images_eileen/tile_data  --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/Cages_n5_tiles_eileen_blind_test_tiles.csv --result-dir /hdd/hanoch/runmodels/img_quality/results/cages_tiles --confidence-threshold 0.21 --hue-norm-preprocess --gpu-id 0 --plot-roc

single channel

CUDA_VISIBLE_DEVICES=0 nohup python -u inference.py mobilenet_v2_256_win_n_lyrs  --model-path /hdd/hanoch/runmodels/img_quality/results/saved_state_mobilenet_v2_256_win_n_lyrs___1603714813  --database-root /hdd/hanoch/data/cages_holdout/annotated_images_eileen/tile_data  --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/Cages_n5_tiles_eileen_blind_test_tiles.csv --result-dir /hdd/hanoch/runmodels/img_quality/results/cages_tiles --single-channel-input-grey --confidence-threshold 0.21 --gpu-id 0 --plot-roc 

Vanilla
1604591734
grad cam
CUDA_VISIBLE_DEVICES=0 nohup python -u inference.py mobilenet_v2_256_win_n_lyrs  --model-path /hdd/hanoch/runmodels/img_quality/results/vanilla/saved_state_mobilenet_v2_256_win_n_lyrs___1604591734 --database-root /hdd/hanoch/data/cages_holdout/annotated_images_eileen/tile_data  --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/Cages_n5_tiles_eileen_blind_test_tiles.csv --result-dir /hdd/hanoch/runmodels/img_quality/results/cages_tiles --confidence-threshold 0.21 --gpu-id 2 --grad-cam-plot
CUDA_VISIBLE_DEVICES=2 nohup python -u inference.py mobilenet_v2_256_win_n_lyrs  --model-path /hdd/hanoch/runmodels/img_quality/results/vanilla/saved_state_mobilenet_v2_256_win_n_lyrs___1604591734 --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data  --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/train_test_val_quality_tile_filtered_eileen_good_bad_val.csv --result-dir /hdd/hanoch/runmodels/img_quality/results/cages_tiles/gradcam_matt --confidence-threshold 0.21 --gpu-id 2 --plot-roc --grad-cam-plot & tail -f nohup.out


CUDA_VISIBLE_DEVICES=2 nohup python -u inference.py mobilenet_v2_256_win_n_lyrs  --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/clip_grad_new_norm/sgd_filt_list/saved_state_mobilenet_v2_256_win_n_lyrs___1606226252 --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data  --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/train_test_val_quality_tile_filtered_eileen_good_bad_val.csv --result-dir /hdd/hanoch/runmodels/img_quality/results/cages_tiles/gradcam_matt --confidence-threshold 0.21 --gpu-id 2 --plot-roc --grad-cam-plot & tail -f nohup.out

What is the train-set loss with fast overfitted model 
CUDA_VISIBLE_DEVICES=0 nohup python -u inference.py mobilenet_v2_256_win_n_lyrs --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/high_lr_clip_grad_longtrain/saved_state_mobilenet_v2_256_win_n_lyrs___1605909841  --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data  --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/file_quality_tile_eileen_good_bad_val_bad_9_2020_avg_pool_filt_weighted_hue_byimage.csv --result-dir /hdd/hanoch/runmodels/img_quality/results --filter-by-train-val-test train --confidence-threshold 0.21 --gpu-id 0 --plot-roc
                                                    mobilenet_v2_256_win_n_lyrs --model-path --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/clip_grad_new_norm/sgd/saved_state_mobilenet_v2_256_win_n_lyrs___1606136726 --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data  --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/file_quality_tile_eileen_good_bad_val_bad_9_2020_avg_pool_filt_weighted_hue_byimage.csv --result-dir /hdd/hanoch/runmodels/img_quality/results --filter-by-train-val-test train --confidence-threshold 0.21 --gpu-id 0 --plot-roc


cages
CUDA_VISIBLE_DEVICES=0 nohup python -u inference.py mobilenet_v2_256_win_n_lyrs --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/clip_grad_new_norm/saved_state_mobilenet_v2_256_win_n_lyrs___1606063570 --database-root /hdd/hanoch/data/cages_holdout/annotated_images_eileen/tile_data  --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/Cages_n5_tiles_eileen_blind_test_tiles.csv --result-dir /hdd/hanoch/runmodels/img_quality/results/cages_tiles --confidence-threshold 0.21 --gpu-id 0 --plot-roc

new filtered csv
CUDA_VISIBLE_DEVICES=0 nohup python -u inference.py mobilenet_v2_256_win_n_lyrs --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/clip_grad_new_norm/sgd_filt_list/saved_state_mobilenet_v2_256_win_n_lyrs___1606226252 --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data  --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/file_quality_tile_eileen_good_bad_val_bad_9_2020_avg_pool_filt_weighted_hue_byimagefilt_by_conf.csv --result-dir /hdd/hanoch/runmodels/img_quality/results --filter-by-train-val-test val --confidence-threshold 0.21 --gpu-id 0 --plot-roc
Fitjar model
CUDA_VISIBLE_DEVICES=0 nohup python -u ./src/CNN/inference.py mobilenet_v2_256_win_n_lyrs --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/clip_grad_new_norm/fitjar/saved_state_mobilenet_v2_256_win_n_lyrs___1609764290 --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data  --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/file_quality_tile_eileen_good_bad_val_bad_9_2020_avg_pool_filt_weighted_hue_byimagefilt_by_conf.csv --result-dir /hdd/hanoch/runmodels/img_quality/results --filter-by-train-val-test val --confidence-threshold 0.21 --gpu-id 0 --plot-roc & tail -f nohup.out
                                                                                                        /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/clip_grad_new_norm/fitjar/saved_state_mobilenet_v2_256_win_n_lyrs___1609764290
Cages_n5_no_marginals_tiles_eileen_blind_test_tiles.csv
CUDA_VISIBLE_DEVICES=0 nohup python -u ./src/CNN/inference.py mobilenet_v2_256_win_n_lyrs --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/clip_grad_new_norm/fitjar/saved_state_mobilenet_v2_256_win_n_lyrs___1609764290 --database-root /hdd/hanoch/data/cages_holdout/annotated_images_eileen/tile_data  --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/Cages_n5_no_marginals_tiles_eileen_blind_test_tiles.csv --result-dir /hdd/hanoch/runmodels/img_quality/results/cages_tiles --confidence-threshold 0.21 --gpu-id 0 --plot-roc & tail -f nohup.out

EfficientNetB0
nohup python -u ./src/CNN/inference.py efficientnet_b0_w256 --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/clip_grad_new_norm/fitjar/saved_state_efficientnet_b0_w256___1609842424 --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data  --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/file_quality_tile_eileen_good_bad_val_bad_9_20_avg_pool_filt_conf_filt_no_edges_trn_tst_fitjar.csv --result-dir /hdd/hanoch/runmodels/img_quality/results/plot_roc --filter-by-train-val-test test --confidence-threshold 0.21 --gpu-id 3 --plot-roc & tail -f nohup.out

HCF
mobilenet_v2_2FC_w256_nlyrs --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/hcf_and_balanced_sampling/saved_state_mobilenet_v2_2FC_w256_nlyrs___1610488946 --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/file_quality_tile_eileen_good_bad_val_bad_9_20_avg_pool_filt_conf_filt_no_edges_trn_tst_fitjar_ntiles.csv --result-dir /hdd/hanoch/runmodels/img_quality/results/plot_roc --filter-by-train-val-test test --confidence-threshold 0.21 --gpu-id 3 --plot-roc --handcrafted-features n_tiles
Cages tiles w/o marginals

Cages_n5_no_marginals_tiles_eileen_blind_test_tiles.csv
nohup python -u ./src/CNN/inference.py mobilenet_v2_2FC_w256_nlyrs --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/hcf_and_balanced_sampling/saved_state_mobilenet_v2_2FC_w256_nlyrs___1610488946 --database-root /hdd/hanoch/data/cages_holdout/annotated_images_eileen/tile_data --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/Cages_n5_no_marginals_tiles_eileen_blind_test_tiles_ntiles.csv --result-dir /hdd/hanoch/runmodels/img_quality/results/plot_roc --filter-by-train-val-test test --confidence-threshold 0.21 --gpu-id 3 --plot-roc --handcrafted-features n_tiles

gradcam Fitjar model
nohup python -u ./src/CNN/inference.py mobilenet_v2_256_win_n_lyrs --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/clip_grad_new_norm/fitjar/saved_state_mobilenet_v2_256_win_n_lyrs___1609764290 --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data  --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/file_quality_tile_eileen_good_bad_val_bad_9_20_avg_pool_filt_conf_filt_no_edges_trn_tst_fitjar.csv --result-dir /hdd/hanoch/runmodels/img_quality/gradcam/gradcam_matt_18thlayer --filter-by-train-val-test test --confidence-threshold 0.21 --gpu-id 3 --plot-roc --grad-cam-plot & tail -f nohup.out

Filter train-set with latest model according to percentile 
nohup python -u ./src/CNN/inference.py mobilenet_v2_256_win_n_lyrs --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/clip_grad_new_norm/fitjar/saved_state_mobilenet_v2_256_win_n_lyrs___1609764290 --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data  --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/file_quality_tile_eileen_good_bad_val_bad_9_20_avg_pool_filt_conf_filt_no_edges_trn_tst_fitjar.csv --result-dir /hdd/hanoch/runmodels/img_quality/filter_by_conf --filter-by-train-val-test train --confidence-threshold 0.21 --gpu-id 3 --plot-roc & tail -f nohup.out

nohup python -u ./src/CNN/inference.py mobilenet_v2_2FC_w256_fusion_3cls --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/embedding_fusion/saved_state_mobilenet_v2_2FC_w256_fusion_3cls___1616008382 --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data  --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/file_quality_tile_eileen_good_bad_val_bad_9_20_avg_pool_filt_conf_filt_no_edges_trn_tst_fitjar_hcf_marginal.csv --result-dir /hdd/hanoch/runmodels/img_quality/filter_by_conf --filter-by-train-val-test train --confidence-threshold 0.85 --gpu-id 2 --plot-roc  --fine-tune-pretrained-model-plan freeze_pretrained_add_gated_atten --classify-image-all-tiles

Gated attention Fusion 
mobilenet_v2_2FC_w256_fusion_3cls --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/embedding_fusion_ext/saved_state_mobilenet_v2_2FC_w256_fusion_3cls___1616529154 --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data  --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/file_quality_tile_eileen_good_bad_val_bad_9_20_avg_pool_filt_conf_filt_no_edges_trn_tst_fitjar_hcf_marginal_ext.csv --result-dir /hdd/hanoch/runmodels/img_quality/filter_by_conf --filter-by-train-val-test test --confidence-threshold 0.85 --gpu-id 0 --plot-roc  --fine-tune-pretrained-model-plan freeze_pretrained_add_gated_atten --classify-image-all-tiles --get-image-name-item

# gated attention
mobilenet_v2_2FC_w256_fusion_3cls --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/embedding_fusion_nonlinear_fix_bug/saved_state_mobilenet_v2_2FC_w256_fusion_3cls___1616951762 --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data  --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/file_quality_tile_eileen_good_bad_val_bad_9_20_avg_pool_filt_conf_filt_no_edges_trn_tst_fitjar_hcf_marginal_ext.csv --result-dir /hdd/hanoch/runmodels/img_quality/filter_by_conf --filter-by-train-val-test val --confidence-threshold 0.85 --gpu-id 0 --plot-roc  --fine-tune-pretrained-model-plan freeze_pretrained_add_gated_atten --classify-image-all-tiles --get-image-name-item

Filter train-set with latest model according to percentile 
nohup python -u ./src/CNN/inference.py mobilenet_v2_256_win_n_lyrs --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/clip_grad_new_norm/fitjar/saved_state_mobilenet_v2_256_win_n_lyrs___1609764290 --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data  --dataset-split-csv /hdd/hanoch/debug/gated_fusion_conf_mat/act_0_det_1/annotations.csv --result-dir /hdd/hanoch/runmodels/img_quality/filter_by_conf --filter-by-train-val-test train --confidence-threshold 0.21 --gpu-id 3 --plot-roc & tail -f nohup.out


Gated attention at classifier input 
mobilenet_v2_2FC_w256_fusion_3cls --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/gated_atten_fix_pooling_point/saved_state_mobilenet_v2_2FC_w256_fusion_3cls___1619733655 --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/file_quality_tile_eileen_good_bad_val_bad_9_20_avg_pool_filt_conf_filt_no_edges_trn_tst_fitjar_hcf_marginal_ext.csv --result-dir /hdd/hanoch/runmodels/img_quality/filter_by_conf --filter-by-train-val-test val --confidence-threshold 0.85 --gpu-id 0 --plot-roc  --fine-tune-pretrained-model-plan freeze_pretrained_add_gated_atten --classify-image-all-tiles --get-image-name-item --pooling-at-classifier

Gated attention with best model  
mobilenet_v2_2FC_w256_fusion_3cls --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/embedding_fusion_nonlinear_fix_bug/saved_state_mobilenet_v2_2FC_w256_fusion_3cls___1616951762 --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data  --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/file_quality_tile_eileen_good_bad_val_bad_9_20_avg_pool_filt_conf_filt_no_edges_trn_tst_fitjar_hcf_marginal_ext.csv --result-dir /hdd/hanoch/runmodels/img_quality/filter_by_conf --filter-by-train-val-test val --confidence-threshold 0.85 --gpu-id 0 --plot-roc  --fine-tune-pretrained-model-plan freeze_pretrained_add_gated_atten --classify-image-all-tiles --get-image-name-item

--database-root /hdd/hanoch/data/database/png_blind_q/png_with_q_grade/gated_attention
One image over gated atten
mobilenet_v2_2FC_w256_fusion_3cls --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/embedding_fusion_nonlinear_fix_bug/saved_state_mobilenet_v2_2FC_w256_fusion_3cls___1616951762 --database-root /hdd/hanoch/data/database/png_blind_q/png_with_q_grade/gated_attention --dataset-split-csv /hdd/hanoch/data/database/png_blind_q/png_with_q_grade/gated_attention/test/unknown-tested/3959fd23-77e5-54c7-939b-a3f566e30fb6.csv --result-dir /hdd/hanoch/runmodels/img_quality/filter_by_conf --filter-by-train-val-test test --confidence-threshold 0.85 --gpu-id 0 --plot-roc  --fine-tune-pretrained-model-plan freeze_pretrained_add_gated_atten --classify-image-all-tiles --get-image-name-item

avg_pool
/hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/MLP_avg_pooling/saved_state_mobilenet_v2_2FC_w256_fusion_3cls___1628146568
--pretrained-weights-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/new_db_7818_bin_class_penn_regularization/saved_state_mobilenet_v2_256_win_n_lyrs___1625844658
mobilenet_v2_2FC_w256_fusion_3cls --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/MLP_avg_pooling/saved_state_mobilenet_v2_2FC_w256_fusion_3cls___1628146568 --database-root /hdd/hanoch/data/database/blind_quality/quality_based_all_annotations_24_6/png/tiles --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/stratified_train_val_test_test_set_penn_dependant_label_concensus_data_list_new_7818.csv --result-dir /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/MLP_avg_pooling --filter-by-train-val-test val --confidence-threshold 0.85 --gpu-id 0 --plot-roc  --fine-tune-pretrained-model-plan freeze_pretrained_add_nn_avg_pooling --classify-image-all-tiles

Bin cls bect model over holdout

mobilenet_v2_256_win_n_lyrs --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/new_db_7818_bin_class_penn_regularization/saved_state_mobilenet_v2_256_win_n_lyrs___1625844658 --database-root /hdd/hanoch/data/database/blind_quality/quality_based_all_annotations_24_6/png/tiles --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/stratified_train_val_test_test_set_penn_dependant_label_concensus_data_list_new_7818.csv --result-dir /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/MLP_avg_pooling --filter-by-train-val-test val --confidence-threshold 0.85 --gpu-id 0 --plot-roc

--model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/avg_pool_concencus-replace-val-with-test_MLP_fusion_2cls/saved_state_mobilenet_v2_2FC_w256_fusion_2cls___1629051510
mobilenet_v2_2FC_w256_fusion_2cls --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/avg_pool_concencus-replace-val-with-test_MLP_fusion_2cls/saved_state_mobilenet_v2_2FC_w256_fusion_2cls___1629051510 --database-root /hdd/hanoch/data/database/blind_quality/quality_based_all_annotations_24_6/png/tiles --dataset-split-csv /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/tile_data/stratified_train_val_test_test_set_penn_dependant_label_concensus_data_list_new_7818.csv --result-dir /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/avg_pool_concencus-replace-val-with-test_MLP_fusion_2cls --filter-by-train-val-test val --confidence-threshold 0.85 --gpu-id 0 --plot-roc

"""