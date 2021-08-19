import numpy as np
import os
import time
import tensorflow as tf
import json
from argparse import ArgumentParser
from skimage import io
import glob
import pickle
from collections import namedtuple
# from skimage import measure
from tensorflow.python.client import timeline
from quality_utils import get_gpu_total_memory

from svm_utils.blindSvmPredictors import blindSvmPredictors as blindSvmPredictors
from quality_utils import load_pb, tile_mapping



outline_743ba24b6d775d4e83633d62022d5ee7 = [[1969.3649647521975, 2533.129149951523], [1641.1382679748535, 2560.0156765944666], [1148.7983699798583, 2569.511827866808], [246.17495384216318, 2546.0588014451723], [141.29558647155773, 2523.290001958394], [0.005078430175672111, 2303.336362331034], [-40.65945190429693, 2116.4084490220325], [-52.72158874511729, 1058.5161758532627], [-47.34852195739745, 977.1396886153188], [0.005078430175672111, 911.2737191838328], [410.2883022308349, 779.1834738751968], [656.4583248138429, 735.7381548984445], [820.5716732025148, 684.5696685571465], [1066.7416957855226, 656.3232261602825], [1230.8548970031738, 605.8726775903497], [1477.0249195861816, 576.8316015888463], [1641.1382679748535, 529.8478224500477], [2133.478313140869, 454.4265237712175], [2656.2980653381346, 407.5054556208547], [2945.7502020263673, 326.12911559180384], [3200.215077667236, 173.04614609093983], [3364.3284260559085, 103.66615317365245], [3610.4984486389158, 114.45524015358023], [4020.781525268555, 204.57519959374304], [4677.234918823242, 243.92678646389527], [4923.40494140625, 239.9578874985948], [5497.801660766601, 175.71960679747212], [5743.9716833496095, 105.91977411722974], [5908.085031738281, 78.80301276556884], [6072.198380126953, 90.50818068689568], [6154.255054321289, 145.16640104664316], [6202.684327697754, 326.12911559180384], [6207.426177978516, 814.3870085572166], [6200.540340270996, 1628.1504088477263], [6187.931315917969, 1709.5268960856706], [6154.255054321289, 1764.674880432568], [5908.085031738281, 1823.510741967949], [5333.68831237793, 1833.113766896639], [4923.40494140625, 1870.0146201154312], [4841.348267211914, 1912.6616252652175], [4669.779823608398, 1953.6557689639303], [4349.008222045898, 2079.5464589510034], [4266.951547851562, 2141.5826418211136], [4020.781525268555, 2268.6345155565], [3938.7248510742183, 2286.8722253154506], [3774.611502685547, 2290.336344986511], [3364.3284260559085, 2223.630401611328], [3200.215077667236, 2227.252918051301], [2920.0450172424316, 2279.1611290801343], [2215.534987335205, 2491.3477330928226], [1969.3649647521975, 2533.129149951523]]
global_good_quality_image_class_label = 3
global_bad_quality_image_class_label = 1
# 4/2/21 since it is based on TF1.12 model then it is run under venv under DL1 of blind_svm_quality
path_to_ref_model_out_vector = '/hdd/hanoch/runmodels/img_quality/export_onnx_var_batch_tf1_12/saved_state_mobilenet_v2_256_win_n_lyrs___1609764290ref_model_out_vectors.npz'

here = os.path.dirname(__file__)
model_dir = os.path.abspath(os.path.join(here, 'model/'))

Point = namedtuple('Point', ['x', 'y'])


class blindQualityNode(object):
    def __init__(self, model_metadata_dictionary=f'{model_dir}/model_meta_data.json',
                 model_path=model_dir, **kwargs):

        self.debug = kwargs['debug_is_active'] if 'debug_is_active' in kwargs else False
        self.fusion_method = kwargs['fusion_method'] if 'fusion_method' in kwargs else 'soft_fusion'

        with open(model_metadata_dictionary, 'r') as f:
                    model_metadata = json.load(f)
        print("model meta parameters file were loaded")
        self.model_metadata = model_metadata

        for k, v in model_metadata.items():
            print(k, v)

        # TODO add a missing : with tf.device(config['device']):  like in finder.py config['device'] is the GPU no.
        if tf.__version__ == '1.12.0' or tf.__version__ == '1.15.0':  # tf.__version__ == '1.12.0' was added to resolve remove!!!
            raise EnvironmentError("TF should be >2.4 but it is {}".format(tf.__version__))
            self.tf_graph, y = load_pb(model_path)
        elif tf.__version__ >= '2.0':
            memory_fraction = 0.12 #0.13 # GPU mem fraction #0.05->15xx, 0.06->1647(B=32) 116tiles, B=4,@1470mS,(B=96@1020mS ) 0.07(comp 771MB) 1757M de-facto 300M model, B=48 (B=96@800mS ) 0.09:B=64, (116,96tiles=>B=2 ~800mS)
            #0.14 with tf.zeros() init vs. 0.13 in pytorch
            total_memory = get_gpu_total_memory()
            physical_devices = tf.config.list_physical_devices('GPU')

            tf.config.experimental.set_virtual_device_configuration(
                physical_devices[kwargs['gpu_id']],
                [tf.config.experimental.VirtualDeviceConfiguration(
                    memory_limit=int(memory_fraction * total_memory[kwargs['gpu_id']]))])

            # TODO table of amount of GPU mem to batch size
            self.max_batch_size = 16
            if int(memory_fraction * total_memory[kwargs['gpu_id']]) <= 670: # 0.06the real GPU meme will be computed upon batch size allocation kitchen-....
                self.max_batch_size = 32
            elif int(memory_fraction * total_memory[kwargs['gpu_id']]) > 770 and int(memory_fraction * total_memory[kwargs['gpu_id']])<880:  #0.07
                self.max_batch_size = 48
            elif int(memory_fraction * total_memory[kwargs['gpu_id']])>=880:
                self.max_batch_size = 64
                if self.debug:
                    print('GPU mem to Bach size was not optimized at asked mem {} [MB] !!!!'.format(int(memory_fraction * total_memory[kwargs['gpu_id']])))

            tf.keras.backend.clear_session()
            tf.config.experimental.set_visible_devices(physical_devices[kwargs['gpu_id']], 'GPU')
            self.tf_graph = tf.saved_model.load(model_path)
            if 0: # cause exception
                tf.config.experimental.enable_mlir_graph_optimization()
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            gpu_devices = tf.config.list_physical_devices('GPU')
            if self.debug:
                print(tf.config.experimental.get_virtual_device_configuration(gpu_devices[kwargs['gpu_id']]))
                print(tf.config.get_logical_device_configuration(gpu_devices[kwargs['gpu_id']]))
                print(len(physical_devices), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

        else:
            raise
        if tf.__version__ == '1.12.0' or tf.__version__ == '1.15.0':
            self.config = tf.ConfigProto(log_device_placement=False,
                                        gpu_options=tf.GPUOptions(visible_device_list=repr(kwargs['gpu_id']).encode("utf-8"),
                                                          per_process_gpu_memory_fraction=0.05)) # was 0.18 0.15

            self.sess = tf.Session(graph=self.tf_graph, config=self.config)
            self.output_tensor = self.tf_graph.get_tensor_by_name('test_output:0')  # from some reason it doesn't find the node name!!
            self.input_tensor = self.tf_graph.get_tensor_by_name('test_input:0')


        # TODO implement the Normalize as simple stand alone
        self.tile_conf_th = self.model_metadata['confidence_threshold']
        self.images_hard_fusion_th = self.model_metadata['voting_confidence_threshold']
        self.images_soft_fusion_th = self.model_metadata['average_pooling_threshold']
        self.average_pooling_threshold_loose_for_secondary_use = self.model_metadata['average_pooling_threshold_loose_for_secondary_use']
        self.hard_fusion_voting_method = 'vote_4_good_class'

        self.chrome_trace = None
        if self.debug:
            self.chrome_trace = False
            self.resulat_stat_acc = dict()
            self.img_dummy_cnt = 1
        # Do not touch settings!!!!
        self.to_scale = False # to_scale=False remain in int8 in tile creation saving un-necessary time spending
        self.tile_size = 256
        self.remove_tiles_on_edges = True
        self.class_labels = {'bad': 0, 'good': 1} # output of the CNN model
        # dummy run of 1st batch to make the TF compute all needed buffers a penalty that takes many sec
        # import torch
        # _ = self._run_cnn_cls(torch.zeros(self.max_batch_size, 3, self.tile_size, self.tile_size))
        if self.debug:
            print("Running dummy batch !")

        # _ = self._run_cnn_cls(np.random.uniform([self.max_batch_size, 3, self.tile_size, self.tile_size], minval=-1, maxval=1, dtype='float32'))
        _ = self._run_cnn_cls(2 * (np.random.rand(self.max_batch_size, 3, self.tile_size, self.tile_size) - 0.5))
            # _ = self._run_cnn_cls(tf.zeros([self.max_batch_size, 3, self.tile_size, self.tile_size], dtype='float32'))

    def handle_model_modeltest(self, dummy_input):
        # for op in self.tf_graph.get_operations():
        #     print(op.values())

        batch_input = tf.concat(self.max_batch_size * [dummy_input], 0)
        # batch_input = dummy_input.cpu()

        # with tf.Session(graph=self.tf_graph, config=config) as sess:  #Device memory is reserved
        for i in range(10):
            tock0 = time.perf_counter()
            if tf.__version__ == '1.12.0' or tf.__version__ == '1.15.0':
                output = self.sess.run(self.output_tensor, feed_dict={self.input_tensor: batch_input})
            elif tf.__version__ >= '2.0':
                infer = self.tf_graph.signatures['serving_default'] # no softmax no normalizing input
                predictions = infer(tf.cast(batch_input, tf.float32))['output_0']
    # bit exactness test

        with open(path_to_ref_model_out_vector, 'rb') as f:
            dummy_output = np.load(f)
        diff_tf_pytorch = dummy_output[0, :] - predictions
        diff_tf_pytorch = np.sum(np.abs(diff_tf_pytorch))
        print("Bit exactness test pyTorch TF model output error {} Batch:{}".format(diff_tf_pytorch,
                                                                                    batch_input.shape[0]))

        return

    # no outline based model check bit extactness only
    def create_dummy_input(self, image_file):

        image = io.imread(image_file)
        batch_images_tens = (image / 255.0 - np.array(self.model_metadata['normalize_rgb_mean'])[None, :]) / np.array(
            self.model_metadata['normalize_rgb_std'])[None, :]

        batch_images_tens = tf.transpose(batch_images_tens[None, :], perm=[0, 3, 1, 2])

        return batch_images_tens

    def _average_pooling(self, sm_confidence_tile, images_soft_fusion_th):
        soft_score = np.mean(sm_confidence_tile)
        if soft_score >= images_soft_fusion_th:
            predict_all_img_label = global_good_quality_image_class_label
        else:
            predict_all_img_label = global_bad_quality_image_class_label
        return predict_all_img_label, soft_score

    def _voting_over_tiles(self, good_cls_ratio, image_fusion_voting_th, confidence_threshold,
                          image_fusion_voting_method='vote_4_good_class'):

        if image_fusion_voting_method == 'vote_4_good_class':
            if good_cls_ratio >= image_fusion_voting_th:  # (good_cls_ratio>0.5) n_good_tiles should be > bad_tiles 'concensus_and_vote'
                th_of_class = confidence_threshold
                acc = good_cls_ratio
                label = global_good_quality_image_class_label
            else:
                th_of_class = 1 - confidence_threshold  # Threshold for class good => for class bad it is 1-th
                label = global_bad_quality_image_class_label
                acc = 1 - good_cls_ratio

        elif image_fusion_voting_method == 'concensus_and_vote':
            if good_cls_ratio >= image_fusion_voting_th and good_cls_ratio > 0.5:  # (good_cls_ratio>0.5) n_good_tiles should be > bad_tiles 'concensus_and_vote'
                th_of_class = confidence_threshold
                acc = good_cls_ratio
                label = global_good_quality_image_class_label
            else:  # bad quality
                th_of_class = 1 - confidence_threshold  # Threshold for class good => for class bad it is 1-th
                acc = 1 - good_cls_ratio
                label = global_bad_quality_image_class_label
        else:
            raise

        return label, acc, th_of_class

    def handle_image(self, image, contour):
        if contour is None:
            raise ValueError("Contour must have a value") # TODO (Erlend)how does exception is going to be treated in the production system
        if image.size == 0:
            print('Image size is 0!!!!')
            return -1
        tiles = self._tiles_create(image, contour)
        if tiles.size == 0:
            print("Warning contour gave no Tiles!!!!")
            self.nested_record_stat = {'tile_good_class_pred' :-2, 'error_msg' : 'no tiles has been extracted'}
            return -2
        # print("Created batch of %d tiles" % (tiles.shape[0]))

        # over single CPU is it efficient to convert to tensor and qpply transformations() over all ?
        batch_images_tens = (tiles/255.0 - np.array(self.model_metadata['normalize_rgb_mean'])[None, :]) / np.array(
                                                    self.model_metadata['normalize_rgb_std'])[None, :]

        batch_images_tens = batch_images_tens.transpose(0, 3, 1, 2)# tf.transpose(batch_images_tens, perm=[0, 3, 1, 2])

        #for profiler
        if self.chrome_trace: # Add to pycharm or shell definition of variable :LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/cuda-9.0/lib64:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()


        # output = self.sess.run(self.output_tensor, feed_dict={self.input_tensor: batch_images_tens})
        if tf.__version__ == '1.12.0' or tf.__version__ == '1.15.0':
            if self.chrome_trace:
                output = self.sess.run(self.output_tensor, feed_dict={self.input_tensor: batch_images_tens},
                                       options=options, run_metadata=run_metadata)
            else:
                output = self.sess.run(self.output_tensor, feed_dict={self.input_tensor: batch_images_tens})
                # TODO find a way doing so in TF f = tf.nn.softmax(output)
                predictions = tf.nn.softmax(np.array(output), dim=1).detach().cpu().numpy()
        elif tf.__version__ >= '2.0':
            predictions = self._run_cnn_cls(batch_images_tens)
            if predictions is None:
                return -1
        # Write metadata
        if self.chrome_trace:
            show_memory = False # heavy duty over the execution time
            fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            self.chrome_trace = fetched_timeline.generate_chrome_trace_format(show_memory=show_memory)

        dets = predictions[:, self.class_labels['good']] > self.tile_conf_th
        good_cls_ratio = np.sum(dets.astype('int')) / dets.shape[0]
        # pseudo label for the entire image based on decision

        if self.fusion_method == 'hard_fusion':
            predict_all_img_label, soft_score, _ = self._voting_over_tiles(good_cls_ratio=good_cls_ratio,
                                                                        image_fusion_voting_th=self.images_hard_fusion_th,
                                                                        confidence_threshold=self.tile_conf_th,
                                                                        image_fusion_voting_method=self.hard_fusion_voting_method)
        elif self.fusion_method == 'soft_fusion':
            predict_all_img_label, soft_score = self._average_pooling(sm_confidence_tile=predictions[:, self.class_labels['good']],
                                                                      images_soft_fusion_th=self.images_soft_fusion_th)

        if self.debug:
            self.nested_record_stat = {'tile_good_class_pred': predictions[:, self.class_labels['good']],
                                  'pred_label': predict_all_img_label,
                                  'image_fusion_voting_th': self.images_hard_fusion_th,
                                  'confidence_threshold': self.tile_conf_th,
                                  'soft_score': soft_score}

            # self.resulat_stat_acc = {self.img_dummy_cnt: nested_record_stat}

            self.img_dummy_cnt += 1

        return predict_all_img_label, soft_score  #TODO convert to old classes 1, 3

    def _run_cnn_cls(self, batch_images_tens):
        if self.debug:
            self.model_run_time = list()

        tock0 = time.perf_counter()
        actual_minibatch_size = 0
        actual_batch_size = batch_images_tens.shape[0]
        if actual_batch_size == 0:
            return None

        if int(actual_batch_size / self.max_batch_size) == (actual_batch_size / self.max_batch_size):
            minibatch_of_batches_size = int(actual_batch_size / self.max_batch_size)
        else:
            minibatch_of_batches_size = int(actual_batch_size / self.max_batch_size + 1)

        # print(actual_batch_size, minibatch_of_batches_size)
        padding_to_batch = self.max_batch_size - actual_batch_size % self.max_batch_size

        # batch_images_tens = tf.concat([tf.cast(batch_images_tens, tf.float32), tf.zeros(
        #     [padding_to_batch, batch_images_tens.shape[1], batch_images_tens.shape[2], batch_images_tens.shape[3]], dtype='float32')], 0)

        batch_images_tens = np.concatenate((batch_images_tens.astype(np.float32), np.zeros(
                                            (padding_to_batch, batch_images_tens.shape[1], batch_images_tens.shape[2], batch_images_tens.shape[3]),
                                            dtype=np.float32)))

        all_predictions = list()
        infer = self.tf_graph.signatures['serving_default']
        for batch in range(minibatch_of_batches_size):
            actual_minibatch_size += (batch + 1) * self.max_batch_size
            mini_batch_images_tens = batch_images_tens[
                                     batch * self.max_batch_size: min((batch + 1) * self.max_batch_size,
                                                                      batch_images_tens.shape[0]), :, :, :]
            all_predictions_batch = infer(tf.cast(mini_batch_images_tens, tf.float32))['output_0']
            all_predictions.append(all_predictions_batch)
        # slice back to the original batch size
        all_predictions = np.concatenate(all_predictions)
        all_predictions = all_predictions[:actual_batch_size, :]
        predictions = tf.nn.softmax(np.array(all_predictions))
        predictions = predictions.numpy()
        # print(predictions.shape[0])

        if self.debug:
            tock1 = time.perf_counter()
            self.model_run_time += [tock1 - tock0]
            print("*********** Finished batch of %d tiles process %.4e" % (actual_batch_size, self.model_run_time[-1]))

        return predictions

    def _tiles_create(self, image, contour):
        self.pred = blindSvmPredictors(cutout=image, outline=contour, tile_size=self.tile_size,
                                                   to_scale=self.to_scale)
        dim_tiles = self.pred.tile_map.shape
        tot_n_tiles = dim_tiles[0] * dim_tiles[1]
        in_outline_n_tiles = np.where(self.pred.tile_map)[0].shape
        if self.debug:
            print("In blind tiles ratio {}".format(in_outline_n_tiles[0] / tot_n_tiles))
        # the tiles within the blind
            if self.pred.cache['tiles'].shape[3] != self.tile_size and self.pred.cache['tiles'].shape[2] != self.tile_size:
                print('Error not the tile dim specified')
                raise ValueError('Error not the tile dim specified')

        if self.remove_tiles_on_edges:
            dim_all_tiles = self.pred.tile_map.shape
            tile_map = list()
            # ref_edges = 'center'
            ref_edges = 'topleft'
            tile_map.append(
                tile_mapping(dim_all_tiles, cutout_shape=image.shape[:2], outline=contour, ref_edges=ref_edges))
            ref_edges = 'topright'
            tile_map.append(
                tile_mapping(dim_all_tiles, cutout_shape=image.shape[:2], outline=contour, ref_edges=ref_edges))
            ref_edges = 'bottomleft'
            tile_map.append(
                tile_mapping(dim_all_tiles, cutout_shape=image.shape[:2], outline=contour, ref_edges=ref_edges))
            ref_edges = 'bottomright'
            tile_map.append(
                tile_mapping(dim_all_tiles, cutout_shape=image.shape[:2], outline=contour, ref_edges=ref_edges))

            # and oiperation between 4 permutations of intersection between tile 4 edges and the outline
            temp = np.ones_like(self.pred.tile_map).astype('bool')
            for tmap in tile_map:
                temp = temp & tmap
            self.pred.cache['tile_map'] = temp

            all_tiles_id = np.where(self.pred.tile_map.ravel())[0]
            remove_tiles_on_edges_ratio = all_tiles_id.shape[0] / in_outline_n_tiles[0]
            if self.debug:
                print("Removed tiles on edges ratio {} total {}".format(remove_tiles_on_edges_ratio, all_tiles_id.shape[0]))

        tiles = self.pred.masked_tile_array

        # Tiles are ordered row-wise by pred.tile_map telling which is relevent in outline by TRue/False
        # saving all the tiles
        assert (np.where(self.pred.tile_map.reshape((-1)))[0].shape[0] == tiles.shape[0])

        return tiles


# def comapre_test_vectors(result_dir, pkl_name):
#
#     with open(os.path.join(result_dir, '1609764290_stat_collect_inference_best_qual.pkl'), 'rb') as f:
#         reference_vectors = pickle.load(f)
#     # with open(os.path.join(result_dir, pkl_file_name), 'rb') as f:
#     #     calc_vectors = pickle.load(f)
#     calc_vectors = pkl_name
#
#     for cutout in calc_vectors.keys():
#         llr_tf_model = calc_vectors[cutout]['tile_good_class_pred']
#         if reference_vectors.get('cutout', None) is not None:
#             llr_ref_model = reference_vectors[cutout]['tile_good_class_pred']
#             if llr_ref_model.size == llr_tf_model.size:
#                 err = llr_tf_model - llr_ref_model
#                 print("sum of abs error {} cutout id{}".format(np.sum(np.abs(err)), cutout))
#         # else:
#         #     print('skip cu id : {}'.format(cutout))
#
def comapre_test_vectors(result_dir, pkl_name, ref_vectors_filename):

    with open(os.path.join(result_dir, ref_vectors_filename), 'rb') as f:
        reference_vectors = pickle.load(f)
    # with open(os.path.join(result_dir, pkl_file_name), 'rb') as f:
    #     calc_vectors = pickle.load(f)
    calc_vectors = pkl_name
    for cutout in calc_vectors.keys():
        llr_tf_model = calc_vectors[cutout]['tile_good_class_pred']
        if reference_vectors.get(cutout, None) is not None:
            llr_ref_model = reference_vectors[cutout]['tile_good_class_pred']
            if llr_ref_model.size == llr_tf_model.size:
                err = llr_tf_model - llr_ref_model
                print("sum of abs error {} cutout id{}".format(np.sum(np.abs(err)), cutout))
                assert (np.isclose(np.sum(np.abs(err)), 0, atol=1e-3).all())
        else:
            print('skip cu id : {}'.format(cutout))

def main(args: list = None):
    parser = ArgumentParser()

    parser.add_argument("--model-path", type=str, required=False, metavar='PATH', default="./model",
                                        help="full path to the neural network model to use")

    parser.add_argument('--gpu-id', type=int, default=0, metavar='INT',
                        help="cuda device id ")

    parser.add_argument("--metadata-json-path", type=str, required=False, default=None, metavar='PATH',
                                        help="training metadata necessary[currently not mandatory] for inference mainly with HCF,")

    parser.add_argument("--process-folder-non-recursively", action='store_true', help="If true.")

    parser.add_argument('--database-root', type=str, required=False, metavar='PATH',
                             help="path to the database")

    parser.add_argument("--outline-file-format", type=str, required=False, default='media_id_cutout_id',
                        choices=['media_id_cutout_id', 'misc', 'other', 'fname_n_cutout_id'], metavar='STRING',
                                        help="")

    parser.add_argument("--outline-pickle-path", type=str, required=True, default=None, metavar='PATH',
                                        help="outline-pickle-path")

    parser.add_argument("--ref-vectors", type=str, required=False, default=None, metavar='PATH',
                                        help="ref vectors -path")


    parser.add_argument('--result-dir', type=str, default=None, metavar='PATH',
                            help="log files and debug ")


    parser.add_argument("--run-modeltest", action='store_true', help="If true.")
    parser.add_argument("--debug", action='store_true', help="If true.")

    args = parser.parse_args(args)
    creating_unitest_test_vec = False
    modeltest = args.run_modeltest
    debug_is_active = args.debug
    resulat_stat_acc = dict()
    pkl_file_name = 'tf_model_output.pkl'

    if debug_is_active and args.result_dir is None:
        raise

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    if args.metadata_json_path:
        blind_quality = blindQualityNode(model_metadata_dictionary=args.metadata_json_path,
                                       model_path=args.model_path, gpu_id=args.gpu_id,
                                       debug_is_active=True)
    else:
        blind_quality = blindQualityNode(gpu_id=args.gpu_id,
                                       debug_is_active=True)

    if modeltest:
        image_file = '/hdd/hanoch/runmodels/img_quality/export_onnx_models/data/fdff7fa9-000a-5aaf-9d9f-ba4dbee17dea_b6722802-5155-5aa6-80c0-67b248e45ef0_cls_good_tile_82.png'
        dummy_input = blind_quality.create_dummy_input(image_file=image_file)

        blind_quality.handle_model_modeltest(dummy_input)
    else:
        #file/outline loader
        try:
            with open(args.outline_pickle_path, 'rb') as f:
                cutout_2_outline_n_meta = pickle.load(f)
                print('direct')
        except:
            import sys
            sys.path.append('/home/hanoch/GIT')
            sys.path.append('/home/hanoch/GIT/Finders')
            sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
            from Finders.finders.blind import default as blindfinder
            with open(args.outline_pickle_path, 'rb') as f:
                cutout_2_outline_n_meta = pickle.load(f)
                print('via_Finders')
        if args.process_folder_non_recursively:
            filenames = [os.path.join(args.database_root, x) for x in os.listdir(args.database_root)
                         if x.endswith('png')]
            print('Process folder NON - recursively')
        else:
            filenames = glob.glob(args.database_root + '/**/*.png', recursive=True)
            print('Process folder recursively')
        for idx, file in enumerate(filenames):
            unique_test_vec = False
            if unique_test_vec:
                print("Warnings unique_test_vec!!")
                creating_unitest_test_vec = True
                file = '/hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/holdout/60407163-19df-5e6a-9b63-8b55315a8e19_743ba24b-6d77-5d4e-8363-3d62022d5ee7.png'
                args.outline_file_format = 'media_id_cutout_id'

            print(idx, file)
            if file.split('.')[-1] == 'png':
                image = io.imread(file)
                if args.outline_file_format == 'media_id_cutout_id':
                    cutout_id = file.split('.')[-2].split('_')[-1]
                    print(cutout_id)
                    outline = cutout_2_outline_n_meta.get(cutout_id, None)
                    if outline == None:
                        print("Outline for cutout_id {} was not found {}!!!".format(cutout_id, file))
                        continue
                    if isinstance(outline, dict):
                        outline = outline['outline']
                elif args.outline_file_format == 'fname_n_cutout_id':
                    if 'blind_' in os.path.split(file)[-1]:
                        cutout_id = 'blind' + os.path.split(file)[-1].split('blind')[-1]
                        media_id = cutout_id  # format is blind_xyz_cutout_uuid hdence keep the namre for filipping the order
                        cutout_uuid = os.path.split(file)[-1].split('_blind')[0]  # this is the reali cutout id
                    elif 'cutout_' in os.path.split(file)[-1]:
                        cutout_id = os.path.split(file)[-1].split('cutout_')[-1].split('.png')[0]
                    outline_dict = cutout_2_outline_n_meta.get(cutout_id, None)
                    if outline_dict:
                        if 'contour' in outline_dict:
                            outline = outline_dict.get('contour', None)
                            if outline is not None:
                                if isinstance(outline, dict):
                                    # outline = outline['outline']
                                    if 'contour' in outline.keys():
                                        outline = outline_dict['contour']
                                    elif 'outline' in outline.keys():
                                        outline = outline['outline']
                        else:
                            outline = outline_dict.get('outline', None)
                            if outline == None:
                                print("Strange !!! the nested dict : Outline for cutout_id {} was not found!!!".format(cutout_id))
                                continue
                    else:
                        print("Outline {} is missing".format(cutout_id))
                        continue

                elif args.outline_file_format == 'misc':
                    outline = cutout_2_outline_n_meta
                else:
                    cutout_id = file.split('/')[-1]

                    outline_dict = cutout_2_outline_n_meta.get(cutout_id, None)

                    if outline_dict == None:
                        print("Outline for cutout_id {} was not found!!!".format(cutout_id))
                        continue

                    outline = outline_dict.get('contour', None)

                    if outline is not None:
                        if isinstance(outline, dict):
                            # outline = outline['outline']
                            if 'contour' in outline.keys():
                                outline = outline_dict['contour']
                            elif 'outline' in outline.keys():
                                outline = outline['outline']
                if unique_test_vec:
                    outline = [Point(x, y) for x, y in outline_743ba24b6d775d4e83633d62022d5ee7]

#TODO add monitoring for counting Acceptance rate of images as class=3/good out of all as a EMA
                blind_quality_category, soft_score = blind_quality.handle_image(image=image, contour=outline)
                resulat_stat_acc.update({cutout_id: blind_quality.nested_record_stat})
                if idx %10 == 0:
                    with open(os.path.join(args.result_dir, pkl_file_name), 'wb') as f:
                        pickle.dump(resulat_stat_acc, f)

                if debug_is_active:
                    if blind_quality.chrome_trace:
                        with open(os.path.join(args.result_dir, 'timeline.json'), 'w') as f:
                            f.write(blind_quality.chrome_trace)

                    if creating_unitest_test_vec:
                        # file = '/hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/holdout/60407163-19df-5e6a-9b63-8b55315a8e19_743ba24b-6d77-5d4e-8363-3d62022d5ee7.png'

                        d = dict()
                        d.update({'cutout_id': '743ba24b-6d77-5d4e-8363-3d62022d5ee7'})
                        d.update({'blind_quality_category': blind_quality_category})
                        d.update({'soft_score': str(soft_score)})

                        d.update({'outline': outline})
                        d.update({'tile_good_class_pred': blind_quality.nested_record_stat['tile_good_class_pred'].tolist()})
                        with open(os.path.join(os.path.split(args.model_path)[0], d['cutout_id'] + 'unittest.json'), 'w') as f:
                            json.dump(d, f)

                        # with open(os.path.join(os.path.split(args.model_path)[0], cutout_id + 'unittest.pkl'), 'wb') as f:
                        #     pickle.dump(d, f)

                    comapre_test_vectors(result_dir=args.result_dir,
                                         pkl_name=resulat_stat_acc,
                                         ref_vectors_filename=args.ref_vectors) #'1609764290_stat_collect_inference_best_qual.pkl'

        with open(os.path.join(args.result_dir, pkl_file_name), 'wb') as f:
            pickle.dump(resulat_stat_acc, f)


if __name__ == '__main__':
    main()


"""
--model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/clip_grad_new_norm/fitjar/saved_state_mobilenet_v2_256_win_n_lyrs___1609764290 --onnx-out-model-path /hdd/hanoch/runmodels/img_quality/export_onnx_var_batch_tf1_12 --metadata-json-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/clip_grad_new_norm/fitjar/mobilenet_v2_256_win_n_lyrs_1609764290.json --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/holdout --outline-pickle-path /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/holdout/quality_holdout_outlines_merged.pkl --gpu-id 2
--model-path /hdd/hanoch/runmodels/img_quality/export_onnx_var_batch_tf1_12/saved_state_mobilenet_v2_256_win_n_lyrs___1609764290_pb/saved_state_mobilenet_v2_256_win_n_lyrs___1609764290.pb --metadata-json-path /hdd/hanoch/runmodels/img_quality/export_onnx_var_batch_tf1_12/saved_state_mobilenet_v2_256_win_n_lyrs___1609764290.json --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/holdout --outline-pickle-path /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/holdout/quality_holdout_outlines_merged.pkl --gpu-id 2

--model-path /hdd/hanoch/runmodels/img_quality/export_onnx_var_batch_tf1_12/saved_state_mobilenet_v2_256_win_n_lyrs___1609764290_pb/saved_state_mobilenet_v2_256_win_n_lyrs___1609764290.pb --metadata-json-path /hdd/hanoch/runmodels/img_quality/export_onnx_var_batch_tf1_12/saved_state_mobilenet_v2_256_win_n_lyrs___1609764290.json --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/holdout --outline-pickle-path /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/holdout/quality_holdout_outlines_merged.pkl --result-dir /hdd/hanoch/runmodels/img_quality/export_onnx_var_batch_tf1_12/saved_outputs --gpu-id 2

Optimized model no gain actually maybe need bazel install and execute
--model-path /hdd/hanoch/runmodels/img_quality/export_onnx_var_batch_tf1_12/optimize_for_inference/optimized.pb --metadata-json-path /hdd/hanoch/runmodels/img_quality/export_onnx_var_batch_tf1_12/saved_state_mobilenet_v2_256_win_n_lyrs___1609764290.json --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/holdout --outline-pickle-path /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/holdout/quality_holdout_outlines_merged.pkl --result-dir /hdd/hanoch/runmodels/img_quality/export_onnx_var_batch_tf1_12/saved_outputs --gpu-id 2
Optimized quantization
--model-path /hdd/hanoch/runmodels/img_quality/export_onnx_var_batch_tf1_12/optimize_for_inference_quant/optimized.pb --metadata-json-path /hdd/hanoch/runmodels/img_quality/export_onnx_var_batch_tf1_12/saved_state_mobilenet_v2_256_win_n_lyrs___1609764290.json --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/holdout --outline-pickle-path /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/holdout/quality_holdout_outlines_merged.pkl --result-dir /hdd/hanoch/runmodels/img_quality/export_onnx_var_batch_tf1_12/saved_outputs --gpu-id 2
/hdd/hanoch/runmodels/img_quality/export_onnx_var_batch_tf1_12/optimize_for_inference_quant/optimized.pb

tf 1.2
--model-path /hdd/hanoch/runmodels/img_quality/export_onnx_var_batch_tf1_12/optimize_for_inference_quant/optimized.pb --metadata-json-path /hdd/hanoch/runmodels/img_quality/export_onnx_var_batch_tf1_12/saved_state_mobilenet_v2_256_win_n_lyrs___1609764290.json --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/holdout --outline-pickle-path /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/holdout/quality_holdout_outlines_merged.pkl --result-dir /hdd/hanoch/runmodels/img_quality/export_onnx_var_batch_tf1_12/saved_outputs --gpu-id 2
tf2.4
--model-path /hdd/hanoch/runmodels/img_quality/export_onnx_TF2p4/saved_state_mobilenet_v2_256_win_n_lyrs___1609764290_pb --metadata-json-path /hdd/hanoch/runmodels/img_quality/export_onnx_var_batch_tf1_12/saved_state_mobilenet_v2_256_win_n_lyrs___1609764290.json --database-root /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/holdout --outline-pickle-path /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/holdout/quality_holdout_outlines_merged.pkl --result-dir /hdd/hanoch/runmodels/img_quality/saved_outputs --gpu-id 2
new model
--model-path /hdd/hanoch/runmodels/img_quality/export_onnx_TF2p5_1625844658/saved_state_mobilenet_v2_256_win_n_lyrs___1625844658_pb --metadata-json-path /hdd/hanoch/runmodels/img_quality/export_onnx_TF2p5_1625844658/saved_state_mobilenet_v2_256_win_n_lyrs___1625844658_pb/saved_state_mobilenet_v2_256_win_n_lyrs___1625844658.json --database-root /hdd/hanoch/data/cages_holdout/annotated_images_eileen_batch2 --outline-pickle-path /hdd/hanoch/data/lice-data-bbox-20191106-simple-sharded-part/cutout_data/merged_outlines.pkl --ref-vectors /hdd/hanoch/runmodels/img_quality/export_onnx_TF2p5_1625844658/1625844658_stat_collect__tta_0_inference_best_qual.pkl --result-dir /hdd/hanoch/runmodels/img_quality/saved_outputs --gpu-id 2  --outline-file-format fname_n_cutout_id --process-folder-non-recursively --debug 

KF site
--database-root /hdd/annotator_uploads/70e91f29c8f3f1147475c252c4e369c6/blindImages --outline-pickle-path /hdd/annotator_uploads/70e91f29c8f3f1147475c252c4e369c6/blindImages/merged_outlines_json.pkl --result-dir /hdd/hanoch/runmodels/img_quality/saved_outputs --gpu-id 2  --outline-file-format fname_n_cutout_id --process-folder-non-recursively --debug
KF site old model

--model-path /hdd/hanoch/runmodels/img_quality/export_onnx_TF2p4/saved_state_mobilenet_v2_256_win_n_lyrs___1609764290_pb --metadata-json-path /hdd/hanoch/runmodels/img_quality/export_onnx_TF2p4/saved_state_mobilenet_v2_256_win_n_lyrs___1609764290_pb/model_meta_data.json --database-root /hdd/annotator_uploads/70e91f29c8f3f1147475c252c4e369c6/blindImages --outline-pickle-path /hdd/annotator_uploads/70e91f29c8f3f1147475c252c4e369c6/blindImages/merged_outlines_json.pkl --result-dir /hdd/hanoch/runmodels/img_quality/saved_outputs/1609764290 --gpu-id 2  --outline-file-format fname_n_cutout_id --process-folder-non-recursively

using defualts 
--database-root /hdd/annotator_uploads/70e91f29c8f3f1147475c252c4e369c6/blindImages --outline-pickle-path /hdd/annotator_uploads/70e91f29c8f3f1147475c252c4e369c6/blindImages/merged_outlines_json.pkl --result-dir /hdd/hanoch/runmodels/img_quality/saved_outputs/1609764290 --gpu-id 0  --outline-file-format fname_n_cutout_id --process-folder-non-recursively

206 tiles
--model-path /hdd/hanoch/runmodels/img_quality/export_onnx_TF2p5_1625844658/saved_state_mobilenet_v2_256_win_n_lyrs___1625844658_pb  --database-root /home/user/blindVision/louse-amb-simple-QF-best-2021_07_08 --outline-pickle-path /home/user/blindVision/louse-amb-simple-QF-best-2021_07_08/to_hanoch.pkl --ref-vectors /hdd/hanoch/runmodels/img_quality/export_onnx_TF2p5_1625844658/1625844658_stat_collect__tta_0_inference_best_qual.pkl --result-dir /hdd/hanoch/runmodels/img_quality/saved_outputs --gpu-id 0  --outline-file-format misc --process-folder-non-recursively --debug
"""
