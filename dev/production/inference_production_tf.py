import numpy as np

import os
import time
import torch
import tensorflow as tf
import json
from argparse import ArgumentParser
from torchvision import transforms
from PIL import Image
import tqdm
import glob
import pickle
from collections import namedtuple
from skimage import measure
from tensorflow.python.client import timeline
from dev.production.export_to_onnx_tf import get_gpu_total_memory

from modules.blind_quality.svm_utils import blindSvmPredictors as blindSvmPredictors
# from src.svm.svm import blindSvm
#from src.svm.svm import blindSvmPredictors
from modules import load_pb
from dev.production.export_to_onnx_tf import create_dummy_input

Point = namedtuple('Point', ['x', 'y'])

# 4/2/21 since it is based on TF1.12 model then it is run under venv under DL1 of blind_svm_quality
path_to_ref_model_out_vector = '/hdd/hanoch/runmodels/img_quality/export_onnx_var_batch_tf1_12/saved_state_mobilenet_v2_256_win_n_lyrs___1609764290ref_model_out_vectors.npz'


class blindQualityNode(object):
    def __init__(self,  **kwargs):
        self.debug = kwargs['debug_is_active']
        self.model_metadata_dictionary = kwargs['model_metadata_dictionary']
        for k, v in self.model_metadata_dictionary.items():
            print(k, v)
        # TODO add a missing : with tf.device(config['device']):  like in finder.py config['device'] is the GPU no.
        if tf.__version__ == '1.12.0' or tf.__version__ == '1.15.0':  # tf.__version__ == '1.12.0' was added to resolve remove!!!
            self.tf_graph, y = load_pb(kwargs['model_path'])
        elif tf.__version__ >= '2.0':
            memory_fraction = 0.07 # GPU mem fraction

            total_memory = get_gpu_total_memory()
            physical_devices = tf.config.list_physical_devices('GPU')

            tf.config.experimental.set_virtual_device_configuration(
                physical_devices[kwargs['gpu_id']],
                [tf.config.experimental.VirtualDeviceConfiguration(
                    memory_limit=int(memory_fraction * total_memory[kwargs['gpu_id']]))])

            # TODO table of amount of GPU mem to batch size
            self.max_batch_size = 16
            if int(memory_fraction * total_memory[kwargs['gpu_id']]) < 600:
                self.max_batch_size = 32
            elif int(memory_fraction * total_memory[kwargs['gpu_id']]) > 1200:
                self.max_batch_size = 48
                print('GPU mem to B was not optimized at {} !!!!'.format(int(memory_fraction * total_memory[kwargs['gpu_id']])))

            tf.keras.backend.clear_session()
            tf.config.experimental.set_visible_devices(physical_devices[kwargs['gpu_id']], 'GPU')
            self.tf_graph = tf.saved_model.load(kwargs['model_path'])
            if 0: # cause exception
                tf.config.experimental.enable_mlir_graph_optimization()
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            gpu_devices = tf.config.list_physical_devices('GPU')
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
        self.transformations = transforms.Compose(
                            [transforms.ToPILImage(),
                             transforms.ToTensor(),
                             transforms.Normalize(self.model_metadata_dictionary['normalize_rgb_mean'],
                                                  self.model_metadata_dictionary['normalize_rgb_std'])])
        self.tile_conf_th = self.model_metadata_dictionary['confidence_threshold']
        self.images_hard_fusion_th = self.model_metadata_dictionary['voting_confidence_threshold']
        self.images_soft_fusion_th = self.model_metadata_dictionary['average_pooling_threshold']
        self.fusion_method = 'hard_fusion'
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
        _ = self._run_cnn_cls(torch.zeros(self.max_batch_size, 3, self.tile_size, self.tile_size))

    def handle_image_unitest(self, dummy_input):
        # for op in self.tf_graph.get_operations():
        #     print(op.values())
        dummy_input = dummy_input[np.newaxis]
        if 1:
            # 16 times the batch[16]
            batch_input = torch.cat(36 * [dummy_input.cpu()])
        else:
            batch_input = dummy_input.cpu()

        # with tf.Session(graph=self.tf_graph, config=config) as sess:  #Device memory is reserved
        for i in range(10):
            tock0 = time.perf_counter()
            if tf.__version__ == '1.12.0' or tf.__version__ == '1.15.0':
                output = self.sess.run(self.output_tensor, feed_dict={self.input_tensor: batch_input})
            elif tf.__version__ >= '2.0':
                predictions = self._run_cnn_cls(batch_input)
    # bit exactness test

        with open(path_to_ref_model_out_vector, 'rb') as f:
            dummy_output = np.load(f)
        diff_tf_pytorch = dummy_output[0, :] - output
        diff_tf_pytorch = np.sum(np.abs(diff_tf_pytorch))
        print("Bit exactness test pyTorch TF model output error {} Batch:{}".format(diff_tf_pytorch,
                                                                                    batch_input.shape[0]))

        return

    def _voting_over_tiles(self, good_cls_ratio, image_fusion_voting_th, confidence_threshold,
                          image_fusion_voting_method='vote_4_good_class'):

        if image_fusion_voting_method == 'vote_4_good_class':
            if good_cls_ratio >= image_fusion_voting_th:  # (good_cls_ratio>0.5) n_good_tiles should be > bad_tiles 'concensus_and_vote'
                th_of_class = confidence_threshold
                acc = good_cls_ratio
                label = 3
            else:
                th_of_class = 1 - confidence_threshold  # Threshold for class good => for class bad it is 1-th
                label = 1
                acc = 1 - good_cls_ratio

        elif image_fusion_voting_method == 'concensus_and_vote':
            if good_cls_ratio >= image_fusion_voting_th and good_cls_ratio > 0.5:  # (good_cls_ratio>0.5) n_good_tiles should be > bad_tiles 'concensus_and_vote'
                th_of_class = confidence_threshold
                acc = good_cls_ratio
                label = 3
            else:  # bad quality
                th_of_class = 1 - confidence_threshold  # Threshold for class good => for class bad it is 1-th
                acc = 1 - good_cls_ratio
                label = 1
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
        # print("Created batch of %d tiles" % (tiles.shape[0]))

        # over single CPU is it efficient to convert to tensor and qpply transformations() over all ?
        batch_images_tens = (tiles/255.0 - np.array(self.model_metadata_dictionary['normalize_rgb_mean'])[None, :]) / np.array(
                                                    self.model_metadata_dictionary['normalize_rgb_std'])[None, :]
        batch_images_tens = torch.from_numpy(batch_images_tens).permute(0, 3, 1, 2)
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
                predictions = torch.nn.functional.softmax(torch.from_numpy(output), dim=1).detach().cpu().numpy()
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


        predict_all_img_label, acc_all_img, _ = self._voting_over_tiles(good_cls_ratio=good_cls_ratio,
                                                                    image_fusion_voting_th=self.images_hard_fusion_th,
                                                                    confidence_threshold=self.tile_conf_th,
                                                                    image_fusion_voting_method=self.hard_fusion_voting_method)

        if self.debug:
            self.nested_record_stat = {'tile_good_class_pred': predictions[:, self.class_labels['good']],
                                  'pred_label': predict_all_img_label,
                                  'image_fusion_voting_th': self.images_hard_fusion_th,
                                  'confidence_threshold': self.tile_conf_th}

            # self.resulat_stat_acc = {self.img_dummy_cnt: nested_record_stat}

            self.img_dummy_cnt += 1

        return predict_all_img_label  #TODO convert to old classes 1, 3

    def _run_cnn_cls(self, batch_images_tens):
        if self.debug:
            model_run_time = list()

        tock0 = time.perf_counter()
        actual_minibatch_size = 0
        actual_batch_size = batch_images_tens.shape[0]
        if actual_batch_size == 0:
            return None

        if int(actual_batch_size / self.max_batch_size) == (actual_batch_size / self.max_batch_size):
            minibatch_of_batches_size = int(actual_batch_size / self.max_batch_size)
        else:
            minibatch_of_batches_size = int(actual_batch_size / self.max_batch_size + 1)

        print(actual_batch_size, minibatch_of_batches_size)
        padding_to_batch = self.max_batch_size - actual_batch_size % self.max_batch_size
        batch_images_tens = torch.cat((batch_images_tens, torch.zeros(
            [padding_to_batch, batch_images_tens.shape[1], batch_images_tens.shape[2],
             batch_images_tens.shape[3]])), axis=0)
        all_predictions = list()
        infer = self.tf_graph.signatures['serving_default']
        for batch in range(minibatch_of_batches_size):
            actual_minibatch_size += (batch + 1) * self.max_batch_size
            mini_batch_images_tens = batch_images_tens[
                                     batch * self.max_batch_size: min((batch + 1) * self.max_batch_size,
                                                                      batch_images_tens.shape[0]), :, :, :]
            all_predictions_batch = infer(tf.convert_to_tensor(mini_batch_images_tens.numpy(), dtype=tf.float32))['output_0']
            all_predictions.append(all_predictions_batch)
        # slice back to the original batch size
        all_predictions = np.concatenate(all_predictions)
        all_predictions = all_predictions[:actual_batch_size, :]

        predictions = torch.nn.functional.softmax(torch.from_numpy(np.array(all_predictions)), dim=1)
        predictions = predictions.numpy()

        if self.debug:
            tock1 = time.perf_counter()
            model_run_time += [tock1 - tock0]
            print("*********** Finished batch of %d tiles process %.4e" % (actual_batch_size, model_run_time[-1]))

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
                self._tile_mapping(dim_all_tiles, cutout_shape=image.shape[:2], outline=contour, ref_edges=ref_edges))
            ref_edges = 'topright'
            tile_map.append(
                self._tile_mapping(dim_all_tiles, cutout_shape=image.shape[:2], outline=contour, ref_edges=ref_edges))
            ref_edges = 'bottomleft'
            tile_map.append(
                self._tile_mapping(dim_all_tiles, cutout_shape=image.shape[:2], outline=contour, ref_edges=ref_edges))
            ref_edges = 'bottomright'
            tile_map.append(
                self._tile_mapping(dim_all_tiles, cutout_shape=image.shape[:2], outline=contour, ref_edges=ref_edges))

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
        if tiles.size == 0:
            print("Warning contour gave no Tiles!!!!")
            return -1
        # Tiles are ordered row-wise by pred.tile_map telling which is relevent in outline by TRue/False
        # saving all the tiles
        assert (np.where(self.pred.tile_map.reshape((-1)))[0].shape[0] == tiles.shape[0])

        return tiles

    def _tile_mapping(self, tile_shape, cutout_shape, outline, ref_edges='center'):
        R, C = tile_shape
        H, W = cutout_shape
        # find centroid coordinates of tiles

        if ref_edges == 'center':
            offset_c = 0.5
            offset_r = 0.5
        elif ref_edges == 'topleft':
            offset_c = 0
            offset_r = 0
        elif ref_edges == 'topright':
            offset_c = 1
            offset_r = 0
        elif ref_edges == 'bottomleft':
            offset_c = 0
            offset_r = 1
        elif ref_edges == 'bottomright':
            offset_c = 1
            offset_r = 1

        tile_C, tile_R = np.meshgrid(
            (np.arange(C) + offset_c) * W / C,
            (np.arange(R) + offset_r) * H / R)
        tile_centroids = np.concatenate(
            (tile_C[..., None], tile_R[..., None]), axis=-1)

        # select tiles whose centroid is inside the provided outline
        return measure.points_in_poly(
            tile_centroids.reshape((-1, 2)),
            [(pt.x, pt.y) for pt in outline]).reshape((R, C))

def comapre_test_vectors(result_dir, pkl_name):

    with open(os.path.join(result_dir, '1609764290_stat_collect_inference_best_qual.pkl'), 'rb') as f:
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
        else:
            print('skip cu id : {}'.format(cutout))

def main(args: list = None):
    parser = ArgumentParser()

    parser.add_argument("--model-path", type=str, required=True, metavar='PATH',
                                        help="full path to the neural network model to use")

    parser.add_argument('--gpu-id', type=int, default=0, metavar='INT',
                        help="cuda device id ")

    parser.add_argument("--metadata-json-path", type=str, required=True, default=None, metavar='PATH',
                                        help="training metadata necessary[currently not mandatory] for inference mainly with HCF,")

    parser.add_argument("--process-folder-non-recursively", action='store_true', help="If true.")

    parser.add_argument('--database-root', type=str, required=False, metavar='PATH',
                             help="path to the database")

    parser.add_argument("--outline-file-format", type=str, required=False, default='media_id_cutout_id',
                        choices=['media_id_cutout_id', 'misc', 'other'], metavar='STRING',
                                        help="")

    parser.add_argument("--outline-pickle-path", type=str, required=True, default=None, metavar='PATH',
                                        help="outline-pickle-path")

    parser.add_argument('--result-dir', type=str, default=None, metavar='PATH',
                            help="log files and debug ")


    parser.add_argument("--run-unittest", action='store_true', help="If true.")
    parser.add_argument("--debug", action='store_true', help="If true.")

    args = parser.parse_args(args)

    unitest = args.run_unittest
    debug_is_active = args.debug
    if debug_is_active:
        resulat_stat_acc = dict()

    if debug_is_active and args.result_dir is None:
        raise

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)


    with open(args.metadata_json_path, 'r') as f:
        model_metadata_dictionary = json.load(f)
    print("model meta paramaters")
    for k, v in model_metadata_dictionary.items():
        print(k, v)

    blind_quality = blindQualityNode(model_metadata_dictionary=model_metadata_dictionary,
                                   gpu_id=args.gpu_id, model_path=args.model_path,
                                   debug_is_active=debug_is_active)

    if unitest:
        if 1:
            image_file = '/hdd/hanoch/runmodels/img_quality/export_onnx_models/data/fdff7fa9-000a-5aaf-9d9f-ba4dbee17dea_b6722802-5155-5aa6-80c0-67b248e45ef0_cls_good_tile_82.png'
            dummy_input = create_dummy_input(image_file=image_file, norm_mean=model_metadata_dictionary['normalize_rgb_mean'],
                                             norm_std=model_metadata_dictionary['normalize_rgb_std'])

        blind_quality.handle_image_unitest(dummy_input)
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

        for idx, file in enumerate(tqdm.tqdm(filenames)):
            if file.split('.')[-1] == 'png':
                img = Image.open(file).convert("RGB")
                image = np.asarray(img)
                if args.outline_file_format is 'media_id_cutout_id':
                    cutout_id = file.split('.')[-2].split('_')[-1]
                    outline = cutout_2_outline_n_meta.get(cutout_id, None)
                    if outline == None:
                        print("Outline for cutout_id {} was not found {}!!!".format(cutout_id, file))
                        continue

                    if isinstance(outline, dict):
                        outline = outline['outline']

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

#TODO add monitoring for counting Acceptance rate of images as class=3/good out of all as a EMA
                blind_quality_category = blind_quality.handle_image(image=image, contour=outline)
                if debug_is_active:
                    print(blind_quality_category)
                    if blind_quality.chrome_trace:
                        with open(os.path.join(args.result_dir, 'timeline.json'), 'w') as f:
                            f.write(blind_quality.chrome_trace)
                    pkl_file_name = 'tf_model_output2.pkl'
                    resulat_stat_acc.update({cutout_id: blind_quality.nested_record_stat})
                    with open(os.path.join(args.result_dir, pkl_file_name), 'wb') as f:
                        pickle.dump(resulat_stat_acc, f)

                    comapre_test_vectors(result_dir=args.result_dir,
                                         pkl_name=resulat_stat_acc)


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



"""
# TODO For integration give the new blindSvmPredictors.blindSvmPredictors() Class with rescale flag