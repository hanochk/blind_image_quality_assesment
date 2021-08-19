import numpy as np
import torch
torch.multiprocessing.set_sharing_strategy('file_system') # due to RuntimeError: received 0 items of ancdata https://discuss.pytorch.org/t/runtimeerror-received-0-items-of-ancdata/4999/2

import sklearn.metrics
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize

import torch.nn as nn
import copy
import tqdm
from dev.models import dropout_test_time_on, dropout_test_time_off
# from models import dropout_test_time_on, dropout_test_time_off
import time
import collections
import torch.nn.functional as F
import os
import itertools
os.environ['DISPLAY'] = str('localhost:10.0')
# for gradcam
from dev.utillities.gradcam import GradCam, show_cam_on_image, GuidedBackpropReLUModel, deprocess_image
from dev.utillities.meters_utils import AverageMeter
from dev.dataset import class_labels
# from utillities.gradcam import GradCam, show_cam_on_image, preprocess_image, GuidedBackpropReLUModel, deprocess_image
# from dataset import class_labels
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
import pandas as pd
import itertools as it
import io
from copy import deepcopy
# import tensorflow as tf
# uSE gARBAGE COLLECTOR TO MONITOR TENSORS
# import gc
# DEVICE=2
# print('Start ******************************')
# for obj in gc.get_objects():
#     try:
#         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
#             if obj.get_device() != DEVICE:
#                 print(type(obj), obj.size(), obj.get_device())
#     except:
#         pass
from dev.utils_tensors import tensor2array
# from utils_tensors import tensor2array
# https://discuss.pytorch.org/t/visualize-feature-map/29597/3
def plot_conv_feature_map(model, preprocessed_img, device, file_name, n_layer_conv_to_show=3):
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
#MobileNet features sequential part has 0:18 blocks that could be addressed inside each subblock they may be several conv layers
    n_layer_conv_to_show = min(18, n_layer_conv_to_show)
    inside_block_conv_layer_to_show = 1
# count the conv2d inside
#     conv_layer = 0
#     for layer in model.features[n_layer_conv_to_show]:
#         if isinstance(layer, nn.Conv2d):
#             print(layer)
#             conv_layer = + 1
    #TODO consider replace the last conv layer conv_layer with inside_block_conv_layer_to_show
    model.features[n_layer_conv_to_show].register_forward_hook(get_activation('conv'+str(inside_block_conv_layer_to_show)))
    # model.features[0].register_forward_hook(get_activation('conv'+str(n_layer_conv_to_show)))
    output = model(to_device(preprocessed_img, device))

    act = activation['conv'+str(inside_block_conv_layer_to_show)].squeeze()
    # fig, axarr = plt.subplots(int(np.sqrt(act.size(0)))+1, int(np.sqrt(act.size(0)))+1)
    fig = plt.figure()
    for idx in range(act.size(0)):
        axarr = plt.subplot(int(np.sqrt(act.size(0))) + 1, int(np.sqrt(act.size(0))) + 1, idx+1)
        axarr.imshow(act[idx].cpu())
        # axarr.imshow(act[idx].cpu(), cmap='gray')
    fig.suptitle("Conv layer{} feature-maps ".format(n_layer_conv_to_show))
    fig.savefig(file_name)
    return


def plot_conv_filter_response(model, n_layer_conv_to_show=2):
    conv_idx = 0
    dim = 3
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            conv_idx += 1
            filters = layer.weight.cpu()
            f_min, f_max = filters.min(), filters.max()
            filters = (filters - f_min) / (f_max - f_min)
            print(layer.weight.shape)
            n_filters, ix = 6, 1
            fig = plt.figure()
            for i in range(n_filters):
                # get the filter
                f = filters[i, :, :, :]
# plot each channel separately
                for j in range(min(dim, f.shape[0])):
                # specify subplot and turn of axis
                    ax = plt.subplot(n_filters, min(dim, f.shape[0]), ix)
                    ax.set_xticks([])
                    ax.set_yticks([])
                        # plot filter channel in grayscale
                    plt.imshow(f[j, :, :], cmap='gray')
                    ix += 1
                # show the figure
            fig.suptitle("Conv layer{} filters {} dim{}".format(conv_idx, n_filters, dim))
            plt.show()
            if (conv_idx == n_layer_conv_to_show):
                break

def find_clipped_gradient_within_layer(model, gradient_clip_value):
    margin_from_sum_abs = 1 / 3
    # find if excess gradient value w/o clipping using the clipping API with clip=INF=100 :just check total norm with dummy high clip val
    total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
    if total_grad_norm > gradient_clip_value:
        max_grad_temp = -100.0
        name_grad_temp = 'None'

        for name, param in model.named_parameters():
            # not_none_grad = [p is not None for p in param.grad]
            if param.grad is not None:
                # print(param.grad)
                norm_layer = torch.unsqueeze(torch.norm(param.grad.detach(), float(2)), 0)
                not_none_grad = [i for i in norm_layer if i is not None]
                for u in not_none_grad:
                    if (u>gradient_clip_value*margin_from_sum_abs).any():
                        # print(name, u[u > gradient_clip_value/2])
                        if (u[u > gradient_clip_value*margin_from_sum_abs] > max_grad_temp):
                            max_grad_temp = u[u > gradient_clip_value *margin_from_sum_abs]
                            name_grad_temp = name

        print("layer {} with max gradient {}".format(name_grad_temp, max_grad_temp))



class WeightHistogram(object):
    """
    Plot a histogram of the network weights and their gradients.

    Parameters
    ----------
    max_num_batches : int, default 10
        Maximum number of batches of weights to accumulate. If this
        is set to a high number, it will take time and memory to
        accumulate the weights and gradients of that batch.
    bins : int, default 1000
        Number of bins to use in the histograms.

    """

    def __init__(self, max_num_batches=10, bins=1000, every_n_epochs=25,
                 debug=False, **kwargs):
        super(WeightHistogram, self).__init__(every_n_epochs=every_n_epochs,
                                              **kwargs)
        self.bins = bins
        self.max_num_batches = max_num_batches

        self.params = collections.OrderedDict()
        self.gradients = collections.OrderedDict()

    def on_epoch_begin(self, state):
        model = state['model']
        step = state['epoch']
        if step % self.every_n_epochs == 0:
            for tag, _ in model.named_parameters():
                for prefix in ['', 'val_']:
                    self.params[prefix + tag] = []
                    self.gradients[prefix + tag] = []

    def on_batch_end(self, state):
        self._epoch_check(state, callback=self._accumulate_weights)

    def on_epoch_end(self, state):
        self._epoch_check(state, callback=self._plot_weight_histogram)

    def _accumulate_weights(self, state):
        """
        Save weights.

        """
        model = state['model']
        prefix = '' if model.training else 'val_'

        for tag, value in model.named_parameters():
            if len(self.params[prefix + tag]) < self.max_num_batches:
                if value.grad is not None:
                    self.params[prefix + tag].append(tensor2array(value))
                    self.gradients[prefix + tag].append(tensor2array(value.grad))

    def _plot_weight_histogram(self, state):
        """
        Plot weight and gradient histograms.

        Parameters
        ----------
        state : dict
            Container.

        """
        step = state['epoch']

        for tag, values in self.params.items():
            tag = tag.replace('.', '/')
            if len(values):
                self.logger.add_histogram(tag,
                                          np.concatenate(values, axis=0),
                                          step=step)

        for tag, gradients in self.gradients.items():
            if len(gradients):
                self.logger.add_histogram(tag + '/grad',
                                          np.concatenate(gradients, axis=0),
                                          step=step)

        state['model'].debug = False
        self.params = collections.OrderedDict()
        self.gradients = collections.OrderedDict()


def to_device(tensor_or_list, device):
    if isinstance(tensor_or_list, (list, tuple)):
        tensor_or_list = [tensor.to(device) for tensor in tensor_or_list]
    else:
        tensor_or_list = tensor_or_list.to(device)

    return tensor_or_list

def grad_cam_model_on_dataset(model, test_df, class_label, device, all_targets, all_predictions, **kwargs):

    test_df['likelihood_good'] = all_predictions[:, class_labels[class_label]]
    test_df['actual_class'] = all_targets
    df = test_df[all_targets == 1]
    det_good_cls = all_predictions[:, class_labels[class_label]][all_targets == 1]
    k = np.argsort(det_good_cls)
    sr = np.sort(det_good_cls)
    df_low_conf_good_cls = df.iloc[k]

    df_low_conf_good_cls["good_cls_prob"] = sr
    target_layer_names = ["18"] #TODO parameter calculatee from last layer feat ext of param
    n_layer_conv_to_show = 3
    print("Grad cam over layer {} ************  ".format(target_layer_names[0]))
    # model.features tells the name of the module and target_layer_names the inside subunit
    grad_cam = GradCam(model=model, feature_module=model.features, \
                       target_layer_names=target_layer_names, device=device, is_mobilenet=True,
                       global_norm_of_heatmap=kwargs['global_norm_of_heatmap']) # mobileNet written with sequential() hard to disect therefore inspect only seq level "features()"

    df_low_conf_good_cls.reset_index()
    for idx, row in df_low_conf_good_cls.iterrows():
        # print(idx)
        fname = row.full_file_name.split('/')[-1].split('.png_')[0]
        # print(fname)
        tile_no = row.file_name.split('_')[-1]
        img = Image.open(row.full_file_name)
        img = img.convert('RGB')
        img_base = copy.deepcopy(img)
        img_base = np.asarray(img)/255
        transform_op = [transforms.CenterCrop(kwargs['input_size']),
                        transforms.ToTensor(),
                        transforms.Normalize(kwargs['normalize_rgb_mean'], kwargs['normalize_rgb_std'])]
        transformations = transforms.Compose(transform_op)
        # from torchvision.utils import save_image
        # save_image(image, 'img1.png')
        img = transformations(img)
        img = np.asarray(img)
        #plot filters response
        preprocessed_img = torch.from_numpy(img)
        preprocessed_img.unsqueeze_(0)
        if 0:
            plot_conv_feature_map(model, preprocessed_img, device, n_layer_conv_to_show=n_layer_conv_to_show)
            plot_conv_filter_response(model, n_layer_conv_to_show=n_layer_conv_to_show)


        input = preprocessed_img.requires_grad_(True)

        # file_full_path = subprocess.getoutput('find ' + path_data + ' -iname ' + '"*' + fname + '.png' + '*"')
        # cages_id = file_full_path.split('/hdd/annotator_uploads')[1].split('/')[1]
        # df_low_conf_good_cls.loc[idx, 'cage_id'] = cages_id
        if row.good_cls_prob < 0.05 or row.good_cls_prob > 0.8:
            target_index = class_labels[class_label]
            # np.seterr(all='ignore')
            mask = grad_cam(input, target_index)
            show_cam_on_image(img_base, mask,
                              file_name=os.path.join(kwargs['path'],
                                        fname + '_t' + str(tile_no) + '_prob_' + str(row.good_cls_prob.__format__('.3e')) + '_layer_' + str(target_layer_names[0]) + '_grad_cam.png'),
                                        norm_by_max=False)

            if 0:
                plot_conv_feature_map(model, preprocessed_img, device, file_name=os.path.join(kwargs['path'],
                                      fname + '_t' + str(tile_no) + '_prob_' + str(row.good_cls_prob.__format__('.3e')) + '_layer_' + str(target_layer_names[0]) + '_featuremap_conv' + str(n_layer_conv_to_show) + '.png'),
                                            n_layer_conv_to_show=n_layer_conv_to_show)
# Guided backpropagation
            gb_model = GuidedBackpropReLUModel(model=model, device=device)
            # print(model._modules.items())
            gb = gb_model(input, index=target_index) # guided backpropagation was declared as more input correlated than XAI by google "Sanity checks for Sailency maps"
            gb = gb.transpose((1, 2, 0))


            cam_mask = cv2.merge([mask, mask, mask])
            cam_gb = deprocess_image(cam_mask * gb)
            gb = deprocess_image(gb)
            file_name_gb = os.path.join(kwargs['path'],
                                        fname + '_t' + str(tile_no) + '_prob_' + str(
                                            row.good_cls_prob.__format__('.3e')) + '_layer_' + str(
                                            target_layer_names[0]) + '_gb.png')
            file_name_cam_gb = os.path.join(kwargs['path'],
                                        fname + '_t' + str(tile_no) + '_prob_' + str(
                                            row.good_cls_prob.__format__('.3e')) + '_layer_' + str(
                                            target_layer_names[0]) + '_cam_gb.png')
            cv2.imwrite(file_name_gb, gb)
            cv2.imwrite(file_name_cam_gb, cam_gb)
        else:
            continue


    return


def check_model_layers_were_modified(model, model_org):
    for n, m in zip(model.pool_method.parameters(), model_org.pool_method.parameters()): # debug that all params were trained
        print(n - m)
        print(n.shape)
    for n, m in zip(model.positional_embeddings_mlp.parameters(), model_org.positional_embeddings_mlp.parameters()):
        print(n - m)
        print(n.shape)

    return


def evaluate_model_on_dataset(model, dataloader, device, loss_fn=None, max_number_of_batches=None,
                              do_softmax=True, **kwargs):
    """
    Generic evaluation code. Get a dataset and a model, do inference on the dataset, and return the results.

    A loss function can be given to also return the validation loss.

    :param max_number_of_batches: Controls how many batches are evaluated. Evaluates the whole dataset by default.
    """

    # device = next(model.parameters()).device  # Get device
    missing_label = kwargs['missing_label'] if 'missing_label' in kwargs else False
    conf_mat = None
    model_was_training_mode = model.training
    model.eval()
    all_unct = []
    all_mean = []
    all_mc_predictions = []
    if 'monte_carlo_dropout_bayes' in kwargs:
        if kwargs['monte_carlo_dropout_bayes']:
            model.apply(dropout_test_time_on) # switch dropout to enable as in training for ensabmling way of inference

    with torch.no_grad():
        # TODO: NORMALIZATION! In case we add something else than hardcoded normalization
        # dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,
        #                                          shuffle=False)
        # Check if model in cuda : next(model.parameters()).is_cuda  => should be True
        total_val_loss = 0.0
        n_samples = 0

        all_targets = list()
        all_predictions = list()
        all_features = list()
        batch_counter = 0
        all_tile_id = list()
        all_atten_weithgs = list()
        all_tile_index_pos = list()

        for read_list in tqdm.tqdm(dataloader):
            if kwargs['positional_embeddings'] is not None:
                inputs, targets, pos_n_rows_m_cols, tile_index_pos = read_list
                pos_n_rows_m_cols = to_device(pos_n_rows_m_cols, device)
                tile_index_pos = to_device(tile_index_pos, device)
                inputs = to_device(inputs, device)
                inputs = [inputs, tile_index_pos, pos_n_rows_m_cols]
            else:
                if len(read_list) == 2:
                    inputs, targets = read_list
                else:
                    inputs, targets, tile_id = read_list
                inputs = to_device(inputs, device)

            if missing_label is False:
                targets = targets.to(device)

            if kwargs['extract_features']:
                if hasattr(model, '_custom_forward_impl'): # model that can combines Hand HCF should have this method implemented to combine HCF
                    x = model.features(inputs) # running model double time, didn;t find a way to resue features
                    features = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)  # copied rfom mobilenet.py
                    predictions = model.forward(inputs)
                else:
                    x = model.features(inputs[0])
                    features = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1) # copied rfom mobilenet.py
                    predictions = model.forward(inputs[0])

                # predictions, features = model.forward_return_multiple_features(inputs)
            else:
                if hasattr(model, '_custom_forward_impl') or hasattr(model, '_custom_forward_impl_embeddings_pooling'): # model that can combines Hand HCF should have this method implemented to combine HCF
                    predictions = model.forward(inputs)
                    if hasattr(model, 'pool_method'):
                        atten_weithgs = model.attention_final_weights_acc # this accumulates under setting debug=True in the model
                else:
                    predictions = model.forward(inputs[0])

                features = None

            if loss_fn is not None and missing_label is False:
                val_loss = loss_fn(predictions, targets)
                n_samples_in_batch = len(targets)
                n_samples += n_samples_in_batch
                # Hidden assumption: The loss has a "mean" reduction (see pytorch loss arguments)
                total_val_loss += val_loss.item() * n_samples_in_batch

            if do_softmax:
                predictions = torch.nn.functional.softmax(predictions, dim=1)
                # print(predictions[:,1])
            else:
                predictions = predictions # logits


            if 'monte_carlo_dropout_bayes' in kwargs:
                if kwargs['monte_carlo_dropout_bayes']:
                    mc_predictions = []
                    # accs_mc = []
                    for i in range(kwargs['monte_carlo_n']):
                        mc_predictions.append(torch.unsqueeze(torch.nn.functional.softmax(model(inputs), dim=1), dim=0))
                        #Ensamble accuracy : go over all MC iterations per all data/batch=>acc(batch) = >mean(acc(batch)) calc hist later
                        # acc = accuracy_score(targets.cpu().numpy(), mc_predictions[i].squeeze().cpu().numpy().argmax(axis=1))
                        # accs_mc.append(acc)
                    output_mean = torch.cat(mc_predictions, 0).mean(dim=0).cpu().numpy()
                    # More precise to calculate over the numpy (CPU) float 64 vs. RTC2080Ti (half precision - float)
                    # output_mean - torch.cat(mc_predictions, 0).cpu().numpy().mean(axis=0) # Since the computation is made with different resolution the deltas are : 5.9604645e-08 to  0.0000000e+00
                    output_variance = torch.cat(mc_predictions, 0).var(dim=0).sqrt().cpu().numpy()
                    all_mean.append(output_mean)
                    all_unct.append(output_variance)
                    all_mc_predictions.append(torch.cat(mc_predictions, 0).cpu().numpy())

            all_targets.append(targets.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())
            if hasattr(model, 'pool_method'):
                all_atten_weithgs.append(atten_weithgs)

            if len(read_list) == 3:
                all_tile_id.append(tile_id)

            if kwargs['positional_embeddings']:
                all_tile_index_pos.append(tile_index_pos)

            if features is not None:
                all_features.append(features.cpu().numpy())

            batch_counter += 1
            if batch_counter == max_number_of_batches:
                break

        all_targets = np.concatenate(all_targets)
        all_predictions = np.concatenate(all_predictions)
        all_features = np.concatenate(all_features) if len(all_features) else None
        all_tile_id = np.concatenate(all_tile_id) if len(all_tile_id) else None
        all_tile_index_pos = np.concatenate(all_tile_index_pos) if len(all_tile_index_pos) else None
        all_atten_weithgs = list(itertools.chain.from_iterable(all_atten_weithgs))
        all_mean = np.concatenate(all_mean) if len(all_mean) else None
        all_unct = np.concatenate(all_unct) if len(all_unct) else None
        all_mc_predictions = np.concatenate(all_mc_predictions, axis=1) if len(all_mc_predictions) else None

        if all_predictions.shape[1] == 3:  # multi-class
            ap = -1
            auc = -1
            if np.unique(all_targets).size == all_predictions.shape[1]:
                auc = sklearn.metrics.roc_auc_score(y_true=all_targets, y_score=all_predictions, multi_class='ovr')
            conf_mat = sklearn.metrics.confusion_matrix(y_true=all_targets, y_pred=all_predictions.argmax(axis=1))
            # self.attention_final_weights for extract the in-accurte decisions
            extract_off_diag_conf_mat_ind = True
            if extract_off_diag_conf_mat_ind and (all_tile_id is not None or all_tile_index_pos is not None):
                if 0: # partach way
                    ind = extract_off_diag_indeces(all_targets=all_targets, all_predictions=all_predictions, all_tile_id=all_tile_index_pos)
    #     We dont have at this stage the filenemes esp/ when run over validation filtering of the csv. Hence extract filenames and csv and run twice otherwise you dont need that file name
    #     Validation way, filter out according to val flag in the csv and take the relevant lines, then create ad-hoc csv
                    q = 1 + ((len(ind['act_2_det_1']['ids']) - 1) // 5)
                    fig, ax = plt.subplots(5, q)
                    plt.title('FP - act_2_det_1')
                    for i, ex in enumerate(ind['act_2_det_1']['ids']):
                        ax[i % 5][i // 5].axhline(1 / len(all_atten_weithgs[ex][0]), color='red')#avg attention per tile
                        ax[i % 5][i // 5].plot(all_atten_weithgs[ex][0])
                        # ax[i%5][i//5].title('Attention+pos_embed 8x8')
                        ax[i % 5][i // 5].set_xlabel('tile index')
                        ax[i % 5][i // 5].set_ylabel('attention')
                        ax[i % 5][i // 5].grid()
                else:
                    ind = extract_off_diag_indeces(all_targets, all_predictions, all_tile_id)

    loss = total_val_loss / n_samples if loss_fn is not None else None
    # restore last trainig state
    model.train(mode=model_was_training_mode)



    if ('monte_carlo_dropout_bayes' in kwargs) and kwargs['monte_carlo_dropout_bayes']:
        bayes_uncertian_dict = {'all_mean': all_mean, 'all_unct': all_unct, 'all_mc_predictions': all_mc_predictions}
    else:
        bayes_uncertian_dict = None

    # restore dropout status as for inference
    if ('monte_carlo_dropout_bayes' in kwargs) and kwargs['monte_carlo_dropout_bayes']:
        model.apply(dropout_test_time_off)

    return_dict = dict()
    return_dict = {'all_targets': all_targets, 'all_predictions': all_predictions,
                   'loss': loss, 'all_features': all_features,
                   'bayes_uncertian_dict': bayes_uncertian_dict}

    if hasattr(model, 'pool_method'):
        return_dict.update({'all_atten_weithgs': all_atten_weithgs})
    if len(read_list) == 3 and (kwargs['positional_embeddings'] is None):
        return_dict.update({'all_tile_id': all_tile_id})
    return return_dict
    # if len(read_list) == 2:
    #     return all_targets, all_predictions, loss, all_features, bayes_uncertian_dict
    # else: # with tile number
    #     return all_targets, all_predictions, loss, all_features, all_tile_id, bayes_uncertian_dict


def extract_off_diag_indeces(all_targets, all_predictions, all_tile_id=None):
    cls1 = np.where(all_targets == 1)
    ind_act_1_det_0 = cls1[0][all_predictions[all_targets == 1, :].argmax(axis=1) == 0]
    list_all_metrices = dict()
    for cls_id_target in np.arange(np.unique(all_targets).size):
        for cls_id_det in np.arange(np.unique(all_targets).size):
            cls1 = np.where(all_targets == cls_id_target)
            globals()[f'act_{cls_id_target}_det_{cls_id_det}'] = cls1[0][all_predictions[all_targets == cls_id_target, :].argmax(axis=1) == cls_id_det]
            if all_tile_id is not None:
                tiles = all_tile_id[globals()[f'act_{cls_id_target}_det_{cls_id_det}']]
                list_all_metrices.update({f'act_{cls_id_target}_det_{cls_id_det}': {'ids' :globals()[f'act_{cls_id_target}_det_{cls_id_det}'],
                                                                                'tiles_id': tiles}})
    return list_all_metrices
# examples all_predictions[ind['act_0_det_1']['ids']] : extract llr of the cross A[0,1] in the conf mat
# ind['act_0_det_1']
######################################################################
# Training the model
# ------------------
#
# Now, let's write a general function to train a model. Here, we will
# illustrate:
#
# -  Scheduling the learning rate
# -  Saving the best model
#
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.

#TODO add         roc_plot()

def train_eval_model(model, args, dataloaders, device, dataset_sizes, criterion_dict,
                    optimizer, lr_scheduler_dict, tb_logger, phases=['train', 'val'], num_epochs=25,
                    monte_carlo_n=100, select_best_model_in_val=True,
                     gradient_clip_value=1000, debug=False, **kwargs):

    since = time.time()
    criterion = criterion_dict['criterion_prime']

    result_dict = dict()
    predictions_dict = dict()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    idx_running_batch = 0
    # Each epoch has a training and validation phase
    best_ap = -1
    best_auc = -1
    best_epoch = -1
    precision = -1
    recall = -1
    best_recall = -1
    best_precision = -1
    conf_mat_test = {}
    conf_mat_recall = {}
    conf_mat_precision = {}
    conf_mat = {}
    if args.loss_smooth:
        label_sm_loss = LabelSmoothingLoss(smoothing=args.loss_smooth_weight, reduction="mean")

    if criterion_dict['secondary_loss_type'] == 'margin':
        prec_meter = AverageMeter()
        sm_meter = AverageMeter()
        dist_ap_meter = AverageMeter()
        dist_an_meter = AverageMeter()
        embeddings_norm_meter = AverageMeter()
        primary_task_loss_weight = criterion_dict['margin_multi_task_weight'][0]
        secondary_task_loss_weight = criterion_dict['margin_multi_task_weight'][1]

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)


        for phase in phases:
            if phase == 'train':
                if hasattr(model, 'open_layers_for_training'): # for that specified layers open the gradient + module to train while other layers to eval
                    open_specified_layers(model, model.open_layers_for_training)
                else:
                    model.train()  # Set model to training mode over all layers
                # TODO : fix that with respect to param.requires_grad==True
                #open_specified_layers(model, args.open_layers)
            # two_stepped_transfer_learning(self.epoch, fixbase_epoch, open_layers) # https://github.com/KaiyangZhou/deep-person-reid/blob/93b8c9f3db59938f39aa95c7383a1c076a65a417/torchreid/engine/engine.py#L237
            else:
                model.eval()   # Set model to evaluate mode
                if args.monte_carlo_dropout_bayes:
                    model.apply(dropout_test_time_on)
            #not working properly
            # if phase == 'train' and (lr_scheduler_dict['lr_decay_policy'] == 'sinus_anneal'):
            #     lr_scheduler_dict['exp_lr_scheduler'].step(lr_scheduler_dict['exp_lr_scheduler'].last_epoch+1)
            #     tb_logger.add_scalar('lr/train', optimizer.param_groups[0]['lr'], idx_running_batch)
            #     print(optimizer.param_groups[0]['lr'])

            running_loss = 0.0
            running_corrects = 0
            iters = len(dataloaders[phase])

            all_targets = []
            all_predictions = []
            idx_batch = 0
            debug_ = False
            if debug_:
                model_org = copy.deepcopy(model)
            # Iterate over data.
            for read_list in tqdm.tqdm(dataloaders[phase]):
                if args.positional_embeddings is not None:
                    inputs, labels_tmp, pos_n_rows_m_cols, tile_index_pos = read_list
                    # print("  0:{} 1:{} 2:{}".format((labels_tmp==0).type(torch.ShortTensor).sum(), (labels_tmp==1).type(torch.ShortTensor).sum(), (labels_tmp==2).type(torch.ShortTensor).sum()))
                    pos_n_rows_m_cols = to_device(pos_n_rows_m_cols, device)
                    tile_index_pos = to_device(tile_index_pos, device)
                    inputs = to_device(inputs, device)
                    inputs = [inputs, tile_index_pos, pos_n_rows_m_cols]
                else:
                    inputs, labels_tmp = read_list
                    inputs = to_device(inputs, device)

                labels = deepcopy(labels_tmp)
                labels = labels.to(device)
                del labels_tmp # https://github.com/pytorch/pytorch/issues/973#issuecomment-459398189

                # zero the parameter gradients
                optimizer.zero_grad()
                idx_running_batch += 1
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    if hasattr(model, '_custom_forward_impl') or hasattr(model, '_custom_forward_impl_embeddings_pooling'):  # model that can combines Hand HCF should have this method implemented to combine HCF
                        outputs = model(inputs)
                        n_out_classes = model.custom_classifier[-1].out_features
                        if criterion_dict['secondary_loss_type'] == 'margin': # Margin/metric learning pair/triplet loss
                            embeddings = model.embeddings
                    else:
                        outputs = model(inputs[0])
                        n_out_classes = model.classifier[-1].out_features # direct from the model

                    _, preds = torch.max(outputs, 1)

                    del inputs #https://github.com/pytorch/pytorch/issues/973#issuecomment-459398189

                    loss = 0
                    loss_batch_summary = {}
                    if args.loss_smooth:
                        loss += label_sm_loss(outputs, labels.long())
                        # one_hot = torch.nn.functional.one_hot(labels.long(), num_classes=n_out_classes)
                        # y_smo = one_hot.float() * (1 - args.loss_smooth_weight) + 1/n_out_classes * args.loss_smooth_weight
                        # loss += F.binary_cross_entropy_with_logits(outputs, y_smo.type_as(outputs))#pos_weight=torch.tensor(pos_weight))
                    else:
                        loss += criterion(outputs, labels.long())
                    loss_batch_summary['loss_ce'] = loss.item()

# Multi-task loss of combined margin loss and NN output based loss like CE/FL
                    if criterion_dict['secondary_loss_type'] == 'margin' and (phase == 'train'):  # Margin/metric learning pair/triplet loss
                        if secondary_task_loss_weight >= 0: # ==0 for statistics collect
                            online_hnm_indices = criterion_dict['miner'](embeddings, labels)

                            embeddings_cpu = embeddings.detach().cpu().numpy()
# check if margin loss collapse all embeddings towards zero
                            embeddings_zero_elements_ratio = [torch.tensor(embeddings_cpu[i, :] == 0).type(torch.int32).sum() /
                                embeddings_cpu[i, :].shape[0] for i in np.arange(embeddings_cpu.shape[0])]

                            embeddings_norm = np.linalg.norm(embeddings_cpu, axis=1).mean()
                            embeddings_norm_meter.update(embeddings_norm)

                            embeddings_zero_elements_ratio_mean = np.array(embeddings_zero_elements_ratio).mean()
                            if any([online_hnm_indices[0][:].nelement() == 0 for i in range(len(online_hnm_indices))]):
                                print("No HNM were found in this batch!!!!")
                                print("embeddings_zero_elements_ratio_mean {}".format(embeddings_zero_elements_ratio_mean))

#distance analysis
                            online_hnm_indices_cpu = [online_hnm_indices[i].cpu().numpy() for i in range(len(online_hnm_indices))]

                            dist_ap = np.linalg.norm(
                                embeddings_cpu[online_hnm_indices_cpu[0][:], :] - embeddings_cpu[online_hnm_indices_cpu[1][:], :], axis=1)

                            cosine_sim = 1 - np.linalg.norm(embeddings_cpu[online_hnm_indices_cpu[0][:], :] - embeddings_cpu[
                                                                                  online_hnm_indices_cpu[1][:], :],
                                                            axis=1) / (np.linalg.norm(embeddings_cpu[online_hnm_indices_cpu[0][:], :],
                                                            axis=1) + np.linalg.norm(embeddings_cpu[online_hnm_indices_cpu[1][:], :], axis=1))
                            if any(np.linalg.norm(embeddings_cpu[online_hnm_indices_cpu[1][:], :], axis=1) == 0):
                                print("positive embedding is zero ")

                            if len(online_hnm_indices) == 4: # 4 for Contrastsive loss a,p,a,n 3:for triplet : a,p,n #https://kevinmusgrave.github.io/pytorch-metric-learning/miners/
                                negative_ind = 3
                                ancor_ind_for_neg = 2

                            elif len(online_hnm_indices) == 3:
                                negative_ind = 2
                                ancor_ind_for_neg = 0
                            else:
                                raise ValueError("Tuple miner yield 3,4 outputs ")

                            dist_an = np.linalg.norm(embeddings_cpu[online_hnm_indices_cpu[ancor_ind_for_neg][:], :] - embeddings_cpu[
                                                                    online_hnm_indices_cpu[negative_ind][:], :], axis=1)

                            if any(np.linalg.norm(embeddings_cpu[online_hnm_indices_cpu[negative_ind][:], :], axis=1) == 0):
                                print("Negative embedding is zero ")

                            min_dim = min(dist_an.shape[0], dist_ap.shape[0]) # consider to pull only min amount  min(dist_an.shape[0], dist_ap.shape[0], )
                            if min_dim > 0:
                                num_pos_pairs = criterion_dict['miner'].num_pos_pairs
                                prec = (dist_an[: min_dim] > dist_ap[: min_dim]).mean()
                                # the proportion of triplets that satisfy margin
                                sm = (dist_an[: min_dim] > dist_ap[: min_dim] + criterion_dict['margin']).mean()
                                d_ap = dist_ap[: min_dim].mean()
                                d_an = dist_an[: min_dim].mean()
                                if np.isinf(np.array(d_an.item())):
                                    print('#########  Inf d_an !!!!!!!!!!!!')
                                prec_meter.update(prec)
                                sm_meter.update(sm)
                                dist_ap_meter.update(d_ap)
                                dist_an_meter.update(d_an)
                            else:
                                print("One of the Pairs list is empty Anchor-p/n : len an {} len ap {}, no statistic was collected".format(dist_an.shape[0], dist_ap.shape[0]))

                            secondary_loss = criterion_dict['criterion_second'](embeddings, labels, online_hnm_indices)
                            loss_batch_summary['loss_margin'] = secondary_loss.item()
                            #Multi-task learning
                            loss = loss * primary_task_loss_weight + secondary_loss * secondary_task_loss_weight
                            # print(loss_batch_summary['loss_ce'], loss_batch_summary['loss_margin'])
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        if debug:
                            find_clipped_gradient_within_layer(model, gradient_clip_value)
    # actual clipping if gradient_clip_value=100 no clipping is made assuming 1000 is too high
                        total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_value)

                        optimizer.step()
                        if debug_:
                            check_model_layers_were_modified(model, model_org)
                        if debug and (total_grad_norm > gradient_clip_value):
                            print("Gradeint {} was clipped to {}".format(total_grad_norm, gradient_clip_value))

                # statistics
                n_samples_in_batch = len(labels)
                # Hidden assumption: The loss has a "mean" reduction (see pytorch loss arguments)
                running_loss += loss.item() * n_samples_in_batch
                running_corrects += torch.sum(preds == labels.data)
                predictions = torch.nn.functional.softmax(outputs, dim=1)

                all_targets.append(labels.data.cpu().numpy())
                # print("cls0: {} cls1: {}".format([all_targets[-1] == 0][0].sum(), [all_targets[-1] == 1][0].sum())) # class distrivution
                all_predictions.append(predictions.detach().cpu().numpy())
                # if tb_logger is not None:
                #     if phase == 'train':
                #         for tag, parameters in model.named_parameters():
                #             if isinstance(parameters, torch.Tensor):
                #                 parameters = [parameters]
                #             parameters = list(filter(lambda p: p.grad is not None, parameters))
                #
                #             print(tag)
                #             norm_type = 2
                #             total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]),
                #                                     norm_type)
                #             # tb_logger.add_histogram(tag, parm.grad.data.cpu().numpy(), idx_running_batch)
                #             tb_logger.add_histogram(tag, total_norm, idx_running_batch)
                # warm_restart : is based on batch level
                # for param_group, lr in zip(optimizer.param_groups, lr_scheduler_dict['exp_lr_scheduler'].get_lr()):
                #     print(param_group['lr'])

# Batch level learning rate optimization
                if phase == 'train' and lr_scheduler_dict['lr_decay_policy'] == 'cosine_anneal_warm_restart':
                    lr_scheduler_dict['exp_lr_scheduler'].step(epoch + idx_batch/iters)
                    tb_logger.add_scalar('lr/train', optimizer.param_groups[0]['lr'], idx_running_batch)

                if phase == 'train' and lr_scheduler_dict['lr_decay_policy'] == 'cyclic_lr':
                    lr_scheduler_dict['exp_lr_scheduler'].step()
                    tb_logger.add_scalar('lr/train', optimizer.param_groups[0]['lr'], idx_running_batch)
                    # print(lr_scheduler_dict['exp_lr_scheduler'].get_last_lr())
                    # print(epoch, optimizer.param_groups[0]['lr'])

                if phase == 'train' and (lr_scheduler_dict['lr_decay_policy'] == 'warmup_over_cosine_anneal' or
                                         lr_scheduler_dict['lr_decay_policy'] == 'warmup_linear_over_cosine_anneal'):
                    lr_scheduler_dict['exp_lr_scheduler'].step(lr_scheduler_dict['exp_lr_scheduler'].last_epoch+1)
                    lr_scheduler_dict['warmup_scheduler'].dampen()
                    tb_logger.add_scalar('lr/train', optimizer.param_groups[0]['lr'], idx_running_batch)
                    # print(epoch, optimizer.param_groups[0]['lr'])
# batch level logging
                if tb_logger is not None:
                    if phase == 'train':
                        if criterion_dict['secondary_loss_type'] == 'margin':
                            tb_logger.add_scalar('CE loss batch/train', loss_batch_summary['loss_ce'], idx_running_batch)
                            tb_logger.add_scalar('Margin loss batch/train', loss_batch_summary['loss_margin'], idx_running_batch)
                            tb_logger.add_scalar('Margin precision batch/train', prec, idx_running_batch)
                            tb_logger.add_scalar('Margin similarity ratio batch/train', sm, idx_running_batch)
                            tb_logger.add_scalar('Margin dist_ap batch/train', d_ap, idx_running_batch)
                            tb_logger.add_scalar('Margin dist_an batch /train', d_an, idx_running_batch)
                            tb_logger.add_scalar('Embeddings_zero_elements_ratio_mean /train', embeddings_zero_elements_ratio_mean, idx_running_batch)
                            tb_logger.add_scalar('embeddings norm /train', embeddings_norm_meter.avg, idx_running_batch)



# Epoch level learning rate optimization
            if phase == 'train':
                if (lr_scheduler_dict['lr_decay_policy'] == 'cosine_anneal' or
                                    lr_scheduler_dict['lr_decay_policy'] == 'StepLR'):
                    lr_scheduler_dict['exp_lr_scheduler'].step()

                tb_logger.add_scalar('lr/train', optimizer.param_groups[0]['lr'], idx_running_batch)
                print(epoch, optimizer.param_groups[0]['lr'])



            elif phase == 'val' and lr_scheduler_dict['lr_decay_policy'] == 'reduced_platue':
                lr_scheduler_dict['exp_lr_scheduler'].step(running_loss)
                tb_logger.add_scalar('lr/train', optimizer.param_groups[0]['lr'], idx_running_batch)
                print("lr : {}".format(optimizer.param_groups[0]['lr']))

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            all_targets = np.concatenate(all_targets)
            all_predictions = np.concatenate(all_predictions)

            predictions_dict.update({phase: all_predictions})
            # binary class
            assert(n_out_classes == all_predictions.shape[1])

            if all_predictions.shape[1] == 2:
                auc = sklearn.metrics.roc_auc_score(y_true=all_targets, y_score=all_predictions[:, 1])
                ap = sklearn.metrics.average_precision_score(y_true=all_targets, y_score=all_predictions[:, 1], pos_label=1)
                                                             # FN of the "good" class
                fn = np.sum([all_predictions.argmax(axis=1) == 0][0].astype('int') * [all_targets == 1][0].astype('int'))

                fp = np.sum([all_predictions.argmax(axis=1) == 1][0].astype('int') * [all_targets == 0][0].astype('int'))
                precision = running_corrects.detach().cpu().numpy()/(running_corrects.detach().cpu().numpy() + fp)
                recall = running_corrects.detach().cpu().numpy()/(running_corrects.detach().cpu().numpy() + fn)

            elif all_predictions.shape[1] == 3: # multi-class
                ap = -1
                all_targets_one_hot = label_binarize(all_targets, classes=[0, 1, 2])

                average_precision = dict()
                for cls_id in range(all_predictions.shape[1]):
                    average_precision[cls_id] = average_precision_score(all_targets_one_hot[:, cls_id], all_predictions[:, cls_id])
                # class godd vs. all 1 vs. all
                ap = average_precision[class_labels['good']]

                auc = sklearn.metrics.roc_auc_score(y_true=all_targets, y_score=all_predictions, multi_class='ovr')
                conf_mat = sklearn.metrics.confusion_matrix(y_true=all_targets, y_pred=all_predictions.argmax(axis=1))
                print(conf_mat)
                conf_mat_recall = conf_mat / conf_mat.sum(axis=1).reshape(-1, 1)
                conf_mat_precision = conf_mat / conf_mat.sum(axis=0).reshape(1, -1)

            else:
                raise ValueError('Not valid option uniclass')

            if tb_logger is not None:
                if phase == 'train':
                    tb_logger.add_scalar('auc/train', auc, idx_running_batch)
                    tb_logger.add_scalar('ap/train', ap, idx_running_batch)
                    tb_logger.add_scalar('acc/train', epoch_acc, idx_running_batch)
                    tb_logger.add_scalar('Loss/train', epoch_loss, idx_running_batch)
                    total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 100) # dont worry the clipping occurs if |sum(grad)|^2>1000 => no clipping just monitoring
                    tb_logger.add_scalar('Grad norm', total_grad_norm, idx_running_batch)
                    if (all_predictions.shape[1] == 3) and (epoch == num_epochs-1) and 0:
                        if 1:
                            df_cm = pd.DataFrame(conf_mat, index=range(3), columns=range(3))
                            plt.figure(figsize=(10, 7))
                            fig = sns.heatmap(df_cm, annot=True, cmap='Spectral').get_figure()

                            # fig = plt.figure()
                            # plt.imshow(conf_mat)
                            tb_logger.add_figure('conf_mat/train', fig, idx_running_batch)
                            plt.close()
                        else:
                            figure = plot_confusion_matrix(conf_mat, class_names=np.arange(3))
                            cm_image = plot_to_image(figure)

                            # Log the confusion matrix as an image summary.
                            tb_logger.add_images("Confusion Matrix", cm_image, idx_running_batch)


                    if criterion_dict['secondary_loss_type'] == 'margin':
                        tb_logger.add_scalar('CE loss/train', loss_batch_summary['loss_ce'], idx_running_batch)
                        tb_logger.add_scalar('Margin loss/train', loss_batch_summary['loss_margin'], idx_running_batch)
                        tb_logger.add_scalar('Margin precision /train', prec_meter.avg, idx_running_batch)
                        tb_logger.add_scalar('Margin similarity ratio /train', sm_meter.avg, idx_running_batch)
                        tb_logger.add_scalar('Margin dist_ap /train', dist_ap_meter.avg, idx_running_batch)
                        tb_logger.add_scalar('Margin dist_an/train', dist_an_meter.avg, idx_running_batch)


                elif phase == 'val':
                    tb_logger.add_scalar('auc/val', auc, idx_running_batch)
                    tb_logger.add_scalar('ap/val', ap, idx_running_batch)
                    tb_logger.add_scalar('acc/val', epoch_acc, idx_running_batch)
                    tb_logger.add_scalar('Loss/val', epoch_loss, idx_running_batch)
                    tb_logger.add_scalar('precision/val', precision, idx_running_batch)
                    tb_logger.add_scalar('recall/val', recall, idx_running_batch)
                    tb_logger.add_scalar('CE loss_batch/val', loss_batch_summary['loss_ce'], idx_running_batch)
                    if all_predictions.shape[1] == 3 and (epoch == num_epochs-1) and 0:
                        # fig = plt.figure()
                        # plt.imshow(conf_mat)
                        df_cm = pd.DataFrame(conf_mat, index=range(3), columns=range(3))
                        plt.figure(figsize=(10, 7))
                        fig = sns.heatmap(df_cm, annot=True, cmap='Spectral').get_figure()
                        tb_logger.add_figure('conf_mat/val', fig, idx_running_batch)
                        plt.close()
                elif phase == 'test':
                    tb_logger.add_scalar('auc/test', auc, idx_running_batch)
                    tb_logger.add_scalar('ap/test', ap, idx_running_batch)
                    tb_logger.add_scalar('acc/test', epoch_acc, idx_running_batch)
                    tb_logger.add_scalar('Loss/test', epoch_loss, idx_running_batch)
                    tb_logger.add_scalar('precision/test', precision, idx_running_batch)
                    tb_logger.add_scalar('recall/test', recall, idx_running_batch)
                    if all_predictions.shape[1] == 3 and (epoch == num_epochs-1) and 0: # it fails with error of localhost

                        # fig = plt.figure()
                        # plt.imshow(conf_mat)
                        df_cm = pd.DataFrame(conf_mat, index=range(3), columns=range(3))
                        plt.figure(figsize=(10, 7))
                        fig = sns.heatmap(df_cm, annot=True, cmap='Spectral').get_figure()

                        tb_logger.add_figure('conf_mat/val', fig, idx_running_batch)
                        plt.close()
                        tb_logger.close()
                    print('TB logger on test')
                else:
                    raise Exception("Invalid option")
            print('{} Loss: {:.4f} Acc: {:.4f} auc: {:.3f} ap: {:.3f} precision {:.3f}: recall {:.3f} support : {:d} '.format(
                phase, epoch_loss, epoch_acc, auc, ap, precision, recall, dataset_sizes[phase]))

            # deep copy the model
            if phase == 'val':
                if ap > best_ap and (all_predictions.shape[1] == 2):
                    best_ap = ap
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_auc = auc
                    best_epoch = epoch + 1 # epoch starts from #0
                    best_recall = recall
                    best_precision = precision
                    result_dict = dict(best_val_acc=best_acc.__format__('.4f'),
                                       best_val_auc=best_auc.__format__('.4f'),
                                       best_val_ap=best_ap.__format__('.4f'),
                                       precision_val=best_precision.__format__('.4f'),
                                       recall_val=best_recall.__format__('.4f'),
                                       best_epoch_val=best_epoch,
                                       auc_last=auc.__format__('.4f'),
                                       conf_mat_recall=conf_mat_recall,
                                       conf_mat_precision=conf_mat_precision)

                elif (all_predictions.shape[1] == 3): # for multiclass is AUC => ap@best_AUC
                    if auc > best_auc:
                        best_epoch = epoch + 1  # epoch starts from #0
                        best_model_wts = copy.deepcopy(model.state_dict())
                        best_auc = auc
                    if ap > best_ap:
                        best_ap = ap

                    result_dict = dict(best_val_auc=best_auc.__format__('.4f'),
                                       best_epoch_val=best_epoch,
                                       auc_last=auc.__format__('.4f'),
                                       conf_mat_val=conf_mat,
                                       conf_mat_recall=conf_mat_recall,
                                       conf_mat_precision=conf_mat_precision,
                                       best_val_ap=best_ap.__format__('.4f'))

            elif phase == 'test':
                result_dict = dict(acc_test=epoch_acc.__format__('.4f'),
                                   auc_test=auc.__format__('.4f'),
                                   ap_test=ap.__format__('.4f'),
                                   precision_test=precision.__format__('.4f'),
                                   recall_test=recall.__format__('.4f'),
                                   conf_mat_test=conf_mat,
                                   conf_mat_recall=conf_mat_recall,
                                   conf_mat_precision=conf_mat_precision)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val AP: {:4f}'.format(best_ap))

    # load best model weights
    if select_best_model_in_val:
        model.load_state_dict(best_model_wts)
    return model, result_dict, predictions_dict

#https://github.com/KaiyangZhou/deep-person-reid/blob/93b8c9f3db59938f39aa95c7383a1c076a65a417/torchreid/engine/engine.py#L476
def two_stepped_transfer_learning(
    self, epoch, fixbase_epoch, open_layers, model=None
):
    """Two-stepped transfer learning.
    The idea is to freeze base layers for a certain number of epochs
    and then open all layers for training.
    Reference: https://arxiv.org/abs/1611.05244
    """
    model = self.model if model is None else model
    if model is None:
        return

    if (epoch + 1) <= fixbase_epoch and open_layers is not None:
        print(
            '* Only train {} (epoch: {}/{})'.format(
                open_layers, epoch + 1, fixbase_epoch
            )
        )
        open_specified_layers(model, open_layers)
    else:
        open_all_layers(model)

def open_all_layers(model):
    r"""Opens all layers in model for training.
    Examples::
        >>> from torchreid.utils import open_all_layers
        >>> open_all_layers(model)
    """
    model.train()
    for p in model.parameters():
        p.requires_grad = True
# https://github.com/KaiyangZhou/deep-person-reid/blob/93b8c9f3db59938f39aa95c7383a1c076a65a417/torchreid/utils/torchtools.py#L183
def open_specified_layers(model, open_layers):
    r"""Opens specified layers in model for training while keeping
    other layers frozen.
    Args:
        model (nn.Module): neural net model.
        open_layers (str or list): layers open for training.
    Examples::
        >>> from torchreid.utils import open_specified_layers
        >>> # Only model.classifier will be updated.
        >>> open_layers = 'classifier'
        >>> open_specified_layers(model, open_layers)
        >>> # Only model.fc and model.classifier will be updated.
        >>> open_layers = ['fc', 'classifier']
        >>> open_specified_layers(model, open_layers)
    """
    if isinstance(model, nn.DataParallel):
        model = model.module

    if isinstance(open_layers, str):
        open_layers = [open_layers]

    for layer in open_layers:
        assert hasattr(
            model, layer
        ), '"{}" is not an attribute of the model, please provide the correct name'.format(
            layer
        )

    for name, module in model.named_children():
        if name in open_layers:
            module.train()
            for p in module.parameters():
                p.requires_grad = True
        else:
            module.eval()
            for p in module.parameters():
                p.requires_grad = False
    if 0: #debug
        for name, module in model.named_children():
            print('=========================')
            for p in module.parameters():
                print(name, p.requires_grad)

def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i, j in it.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure

def plot_to_image(figure):
    """
    Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    """

    buf = io.BytesIO()

    # Use plt.savefig to save the plot to a PNG in memory.
    plt.savefig(buf, format='png')

    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)

    # Use tf.image.decode_png to convert the PNG buffer
    # to a TF image. Make sure you use 4 channels.
    image = tf.image.decode_png(buf.getvalue(), channels=4)

    # Use tf.expand_dims to add the batch dimension
    image = tf.expand_dims(image, 0)

    return image
"""
Guilaume : giom

Add that practice for gradint NaN of norm_grad >100*traget 

# gradient clipping
# by default self.grad_norm is float('inf') and the line below takes no effect.
total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
if (not math.isnan(total_norm)) and total_norm < 100*self.grad_norm:
    self.state['max_grad_norm'] = max(total_norm, self.state['max_grad_norm'])
    self.optimizer.step()
else:
    print("Gradient is NaN (or larger than 100 x grad_norm_target), "
          "optimization step skipped for this batch")
    print("Loss value = {}".format(loss.item()))
    if math.isnan(loss.item()):
        ValueError("Loss value is NaN.")


Moran klien
"""
if 0:
    class EMA(nn.Module):
        #Exponential moving average
        def __init__(self, mu):
            super(EMA, self).__init__()
            self.mu = mu
            self.flag_first_time_passed = False
        def forward(self, x, last_average):
            if self.flag_first_time_passed==False:
                new_average = x
                self.flag_first_time_passed = True
            else:
                new_average = self.mu * x + (1 - self.mu) * last_average
            return new_average


    gradient_clip_value = 10 # initial value
    ema_mu = 0.01
    ema_object = EMA(ema_mu)

    # back propagation
    loss.backward()
    total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_value)
    # (*). Update norm moving average: (*)#
    gradient_clip_value = ema_object(total_grad_norm, gradient_clip_value)


    def set_optimizer_learning_rate(optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    # lr scheduler
    init_learning_rate = hp.train.adam
    hypergrad_lr = 0  # for AdamHR
    flag_change_optimizer_betas = False
    flag_use_learning_rate_scheduler = True
    minimum_learning_rate = 0.00025
    learning_rate_lowering_factor = 0.8
    learning_rate_lowering_patience_before_lowering = 450
    learning_rate_lowering_tolerance_factor_from_best_so_far = 0.03
    initial_spike_learning_rate_reduction_multiple = 0.95
    minimum_learning_rate_counter = 0
    Generator_lr_previous = init_learning_rate
    learning_rate = init_learning_rate * (
            initial_spike_learning_rate_reduction_multiple ** minimum_learning_rate_counter)
    set_optimizer_learning_rate(optimizer, learning_rate)
    if hp.train.lr_schedular:
        lr_schedualer = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       mode='min',
                                                       factor=learning_rate_lowering_factor,
                                                       patience=learning_rate_lowering_patience_before_lowering,
                                                       threshold=learning_rate_lowering_tolerance_factor_from_best_so_far,
                                                       threshold_mode='rel')



    ### Get Generator Learning Rate: ###
    if type(optimizer.param_groups[0]['lr']) == torch.Tensor:
        Generator_lr = optimizer.param_groups[0]['lr'].cpu().item()
    elif type(optimizer.param_groups[0]['lr']) == float:
        Generator_lr = optimizer.param_groups[0]['lr']

    if hp.train.lr_schedular:
        ### Update Learning-Rate Scheduler: ###
        lr_schedualer.step(loss)

        ### If Learning Rate changed do some stuff: ###
        if Generator_lr < Generator_lr_previous:
            print('CHANGED LEEARNING RATE!!!!')
            if flag_change_optimizer_betas:
                for i, param_group in enumerate(optimizer.param_groups):
                    param_group['betas'] = tuple(
                        [1 - (1 - b) * 0.1 for b in param_group['betas']])  # beta_new = 0.1*beta_old + 0.9
        Generator_lr_previous = Generator_lr

        ### If Learning Rate falls below minimum then spike it up: ###
        if Generator_lr < minimum_learning_rate:
            minimum_learning_rate_counter += 1
            # If we're below the minimum learning rate than it's time to jump the LR up to try and get to a lower local minima:

            ### Initialize Learning Rate (later to be subject to lr scheduling): ###
            learning_rate = init_learning_rate * (
                    initial_spike_learning_rate_reduction_multiple ** minimum_learning_rate_counter)  # TODO: make hyper-parameter  and instead of adjusting the initial LR at the start of each meta_epoch simply adjust it after each "meta-epoch" as i define it which would be dynamic
            set_optimizer_learning_rate(optimizer, learning_rate)

            ### Initialize Learning Rate Scheduler: ###
            lr_schedualer = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           factor=learning_rate_lowering_factor,
                                                           patience=learning_rate_lowering_patience_before_lowering,
                                                           threshold=learning_rate_lowering_tolerance_factor_from_best_so_far,
                                                           threshold_mode='rel')

    # back propagation
    loss.backward()


    """
Profiling 
x = torch.randn((1, 1), requires_grad=True)
>>> with torch.autograd.profiler.profile() as prof:
>>>     for _ in range(100):  # any normal python code, really!
>>>         y = x ** 2
>>          y.backward()
>>> # NOTE: some columns were removed for brevity
>>> print(prof.key_averages().table(sort_by="self_cpu_time_total"))
    
Virtual increased mini-batch

count = 0
for inputs, targets in training_data_loader:
    if count == 0:
        optimizer.step()
        optimizer.zero_grad()
        count = batch_multiplier
    
    outputs = model(inputs)
    loss = loss_function(outputs, targets) / batch_multiplier
    loss.backward()
    
    count -= 1    
    """


class EntropyLoss(torch.nn.Module):
    def forward(self, input):
        super(EntropyLoss, self).__init__()
        probabilities = torch.nn.functional.softmax(input, -1)

        # Entropy has a minus sign. We want to MAXIMIZE the entropy, hence the other minus.
        return - (- probabilities * torch.log(probabilities)).sum(-1).mean()

#https://stackoverflow.com/questions/55681502/label-smoothing-in-pytorch
class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, smoothing: float = 0.1,
                 reduction="mean", weight=None):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        self.weight = weight

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
         if self.reduction == 'sum' else loss

    def linear_combination(self, x, y):
        return self.smoothing * x + (1 - self.smoothing) * y

    def forward(self, preds, target):
        assert 0 <= self.smoothing < 1

        if self.weight is not None:
            self.weight = self.weight.to(preds.device)

        n = preds.size(-1)
        log_preds = F.log_softmax(preds, dim=-1)
        loss = self.reduce_loss(-log_preds.sum(dim=-1))
        nll = F.nll_loss(
            log_preds, target, reduction=self.reduction, weight=self.weight
        )
        return self.linear_combination(loss / n, nll)
"""
@ per sample weight https://discuss.pytorch.org/t/per-class-and-per-sample-weighting/25530/4
# https://gist.github.com/ptrblck/4dfd97f487c469d01a4aa8d738c893ea
batch_size = 10
nb_classes = 2

model = nn.Linear(10, nb_classes)
weight = torch.empty(nb_classes).uniform_(0, 1)
criterion = nn.CrossEntropyLoss(weight=weight, reduction='none')

# This would be returned from your DataLoader
x = torch.randn(batch_size, 10)
target = torch.empty(batch_size, dtype=torch.long).random_(nb_classes)
sample_weight = torch.empty(batch_size).uniform_(0, 1)

output = model(x)
loss = criterion(output, target)

sample_weight = sample_weight.view(-1, 1, 1)
loss =(loss * sample_weight / sample_weight.sum()).sum() # good idea to normalize the sample weights so that the range of the loss will approx. have the same range and won’t depend on the current sample distribution in your batch.

# loss = loss * sample_weight
loss.mean().backward()

Debug all param were trained 
                        # for n, m in zip(model.pool_method.parameters(), model_org.pool_method.parameters()): # debug that all params were trained
                        #     print(n - m)
                        #     print(n.shape)
                        #     print(m.shape)

"""
