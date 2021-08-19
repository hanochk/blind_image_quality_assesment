import numpy as np

import os
import time
import torch
import torch.nn as nn
import onnx
from onnx_tf.backend import prepare  #https://github.com/onnx/onnx-tensorflow/blob/master/doc/API.md
from modules import get_gpu_total_memory
# from onnx_tf.backend_rep import TensorflowRep
# pip install tensorflow-gpu==1.9,  pip install tensorboard==1.14, pip install --upgrade tensorboard
#[pip install tensorflow==2.3.0 , pip install --upgrade pip , pip3 install tensorflow-gpu==2.4.0 , pip3 install tensorflow-estimator==2.4]

# finallly : pip uninstall tensorflow -y  + pip uninstall tensorflow-gpu + uninstall pip uninstall tensorboard ; pip install --upgrade tensorflow-gpu==1.12.0
#HACK: onnx-tf 1.3.0 fix manually : #HK: fix to support onnx-tf 1.3 fixed bug of BN v9.0 with TF 1.12 :https://github.com/onnx/onnx-tensorflow/issues/499 #onnx-tensorflow\onnx_tf\handlers\backend\is_inf.py":  @tf_func(tf.math.is_inf) @tf_func(tf.is_inf)  comment everything in scatter_nd.py
#Best option :  tf-gpu 1.14 with onnx-tf 1.5.0 but need to upgrade CUDA

# import uff
# import graphsurgeon as gs
# import tensorrt as trt
# Uncomment next line only on tensor-rt creation model since inside the converter.convert() it isn't obey the GPU allocation rather than that env setting
#  --convert-tf-to-tf-models
# os.environ["CUDA_VISIBLE_DEVICES"] = '"' + str(0) + '"'
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
if tf.__version__ > '2.0':
    from tensorflow.python.compiler.tensorrt import trt_convert as trt

import json
from argparse import ArgumentParser
from torchvision import transforms
from PIL import Image

from dev.models import initialize_model

# no outline based model check bit extactness only
def create_dummy_input(image_file, model_metadata):
    norm_mean = model_metadata['normalize_rgb_mean']
    norm_std = model_metadata['normalize_rgb_std']

    for k, v in model_metadata.items():
        print(k, v)

    transformations = transforms.Compose(
        [transforms.ToPILImage(),
         transforms.ToTensor(),
         transforms.Normalize(norm_mean, norm_std)])

    img = Image.open(image_file).convert("RGB")
    image = np.asarray(img)
    image = transformations(image)

    return image

def run_mps(savedmodel_path, batch_input):
    #according to NVIDIA MPS doc
    """

PROHIBITED – the GPU is not available for compute applications.
EXCLUSIVE_PROCESS – the GPU is assigned to only one process at a time, and
individual process threads may submit work to the GPU concurrently.
DEFAULT – multiple processes can use the GPU simultaneously. Individual threads
of each process may submit work to the GPU simultaneously.

export CUDA_VISIBLE_DEVICES=0 # Select GPU 0.
nvidia-smi -i 0 -c EXCLUSIVE_PROCESS # Set GPU 0 to exclusive mode.
nvidia-cuda-mps-control -d # Start the daemon.
or
echo quit | sudo nvidia-cuda-mps-control

Note that CUDA_VISIBLE_DEVICES should not be set in the client process’s
environment.

To shut down the daemon, as root, run
echo quit | nvidia-cuda-mps-control
    """

    tf_graph = tf.saved_model.load(savedmodel_path)
    tf_graph2 = tf.saved_model.load(savedmodel_path)
    tf_graph3 = tf.saved_model.load(savedmodel_path)
    tf_graph4 = tf.saved_model.load(savedmodel_path)
    tf_graph5 = tf.saved_model.load(savedmodel_path)
    tf_graph6 = tf.saved_model.load(savedmodel_path)
    tf_graph7 = tf.saved_model.load(savedmodel_path)
    tf_graph8 = tf.saved_model.load(savedmodel_path)
    tf_graph9 = tf.saved_model.load(savedmodel_path)
    tf_graph10 = tf.saved_model.load(savedmodel_path)

    infer = tf_graph.signatures['serving_default']
    infer2 = tf_graph2.signatures['serving_default']
    infer3 = tf_graph3.signatures['serving_default']
    infer4 = tf_graph4.signatures['serving_default']
    infer5 = tf_graph5.signatures['serving_default']
    infer6 = tf_graph6.signatures['serving_default']
    infer7 = tf_graph7.signatures['serving_default']
    infer8 = tf_graph8.signatures['serving_default']
    infer9 = tf_graph9.signatures['serving_default']
    infer10 = tf_graph10.signatures['serving_default']

    model_run_time = list()
    print('Starting Multi processing jobs!!!')
    for i in range(12):
        tock0 = time.perf_counter()

        output1 = infer(tf.convert_to_tensor(batch_input.numpy()))['output_0']
        output2 = infer2(tf.convert_to_tensor(batch_input.numpy()))['output_0']
        output3 = infer3(tf.convert_to_tensor(batch_input.numpy()))['output_0']
        output4 = infer4(tf.convert_to_tensor(batch_input.numpy()))['output_0']
        # output5 = infer5(tf.convert_to_tensor(batch_input.numpy()))['output_0']
        # output6 = infer6(tf.convert_to_tensor(batch_input.numpy()))['output_0']
        # output7 = infer7(tf.convert_to_tensor(batch_input.numpy()))['output_0']
        # output8 = infer8(tf.convert_to_tensor(batch_input.numpy()))['output_0']
        # output9 = infer9(tf.convert_to_tensor(batch_input.numpy()))['output_0']
        # output10 = infer10(tf.convert_to_tensor(batch_input.numpy()))['output_0']

        tock1 = time.perf_counter()
        model_run_time += [tock1 - tock0]
        print("Finished tensor process %.4e" % (model_run_time[-1]))


    # with open(path_to_ref_model_out_vector, 'rb') as f:
    #     dummy_output = np.load(f)
    # diff_tf_pytorch = dummy_output[0, :] - output
    # diff_tf_pytorch = np.sum(np.abs(diff_tf_pytorch))
    for i in range(10):
        print("Bit exactness test pyTorch TF model output {} ".format(globals()[f'output{i+1}']))


def load_pb_tf2_try(path_to_pb):
    from tensorflow.python.platform import gfile
    from tensorflow.core.protobuf import saved_model_pb2
    from tensorflow.python.util import compat
    with tf.compat.v1.Session() as sess:
        with gfile.FastGFile(path_to_pb, 'rb') as f:
            data = compat.as_bytes(f.read())
            sm = saved_model_pb2.SavedModel()
            sm.ParseFromString(data)
            g_in = tf.import_graph_def(sm.meta_graphs[0].graph_def)
    return g_in

#TF 1.14+onnx-tf=1.5  :
def load_pb_tfgpu_1_14_onnxtf_1_5(path_to_pb):
    with tf.io.gfile.GFile(path_to_pb, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph
# ver_tf_1_12_onnx_tf_1_3 later moved to onnx-tf 1.7

def save_onnx_model(model, dummy_input, model_path,
                                  model_name='onnx_model.onnx'):
    # Very important to stick to that name dropping of in/out to be addressed by TF model
    # input_names=['test_input'], output_names=['test_output']
    # torch.onnx.export(model, dummy_input, os.path.join(model_path, model_name),
    #                   input_names=['test_input'], output_names=['test_output'],
    #                   dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
    #                                 'output': {0: 'batch_size'}}, verbose=False)

    torch.onnx.export(model, dummy_input, os.path.join(model_path, model_name), export_params=True,
                      input_names=['test_input'], output_names=['test_output'],
                      verbose=True)
    # model_metadata_path = '/hdd/hanoch/data'
    # confidence_threshold = 0.1
    # voting_confidence_threshold = 0.25
    # n_model_outputs = 2
    # tta = False

    for child_name, child in model.classifier.named_children():
        if isinstance(child, nn.Linear):
            n_model_outputs = getattr(child, 'out_features')
            print("n_model_outputs : {}".format(n_model_outputs))
#     model_metadata_dictionary = {
#         'confidence_threshold': round(confidence_threshold, 3),
#         'voting_confidence_threshold': round(voting_confidence_threshold, 3),
#         'n_model_outputs': n_model_outputs,
#         'tta': tta
#     #    Normalize([0.13804892, 0.21836744, 0.20237076], [0.08498618, 0.07658653, 0.07137364])])
#         #  HCF dict {'f1': {mean, std}}
#     }
#     with open(os.path.join(model_path, meta_data_file_name), 'w') as f:
#         json.dump(model_metadata_dictionary, f)


def main(args: list = None):
    parser = ArgumentParser()

    parser.add_argument('run_config_name', type=str, metavar='RUN_CONFIGURATION',
                             help="name of an existing run configuration")

    parser.add_argument("--model-path", type=str, required=True, metavar='PATH',
                                        help="full path to the neural network model to use")

    parser.add_argument("--onnx-out-model-path", type=str, required=True, metavar='PATH',
                                        help="onnx path to the neural network model to use")

    parser.add_argument('--gpu-id', type=int, default=0, metavar='INT',
                        help="cuda device id ")

    parser.add_argument("--metadata-json-path", type=str, required=True, default=None, metavar='PATH',
                                        help="training metadata necessary[currently not mandatory] for inference mainly with HCF,")
    
    parser.add_argument('--convert-pytorch-to-tf-models', action='store_true',
                        help='')

    parser.add_argument('--convert-tf-to-tf-models', action='store_true',
                        help='')

    parser.add_argument('--tf-model-type', type=str, default='tf', metavar='STRING',
                        help='tflite, tensor-rt')

    parser.add_argument('--quantization-8bit', action='store_true',
                        help='')
    # tflite/tensor_rt over an existing TF model
    args = parser.parse_args(args)

    if 'CUDA_VISIBLE_DEVICES' not in os.environ and args.convert_tf_to_tf_models and (args.tf_model_type == 'tensor-rt'):
        raise EnvironmentError("Need to set manually the CUDA_VISIBLE_DEVICES before import TF in the above piece of code !!!!")
    #data
    quantization = args.quantization_8bit

    memory_format_torch_channels_last = False

    load_pytorch_ref_model = False
    if args.convert_pytorch_to_tf_models:
        load_pytorch_ref_model = True

    batching = True

    with open(args.metadata_json_path, 'r') as f:
        model_metadata = json.load(f)

    if model_metadata['model_architecture'] != args.run_config_name:
        raise ValueError('model config name does not equal to the JSON content')

    device = torch.device("cuda:" + str(args.gpu_id) + "" if torch.cuda.is_available() else "cpu")
    # dummy_input = torch.from_numpy(np.random.rand(3, 256, 256)[np.newaxis]).float().to(device)
    image_file = '/hdd/hanoch/runmodels/img_quality/export_onnx_models/data/fdff7fa9-000a-5aaf-9d9f-ba4dbee17dea_b6722802-5155-5aa6-80c0-67b248e45ef0_cls_good_tile_82.png'
    dummy_input = create_dummy_input(image_file=image_file, model_metadata=model_metadata)
    if batching:
        batch_size = 16
        dummy_input = torch.cat(batch_size * [dummy_input[np.newaxis]])
    else:
        dummy_input = dummy_input[np.newaxis]


    onnx_model_name = args.model_path.split('/')[-1] + '.onnx'
    ref_model_out_vector = args.model_path.split('/')[-1] + 'ref_model_out_vectors.npz'
    pb_model_name = args.model_path.split('/')[-1] + '.pb'
    pb_model_path = args.model_path.split('/')[-1] + '_pb'
    meta_data_file_name = args.model_path.split('/')[-1] + '.json'
    output_uff_filename = args.model_path.split('/')[-1] + '.uff'


    mode_q_str = ''
    if quantization:
        mode_q_str = '_qaunt_8bit'
    path_to_tflie = os.path.join(args.onnx_out_model_path, mode_q_str + 'model.tflite')

    path_to_onnx = os.path.join(args.onnx_out_model_path, onnx_model_name)
    path_to_pb = os.path.join(args.onnx_out_model_path, pb_model_path, pb_model_name)
    path_to_ref_model_out_vector = os.path.join(args.onnx_out_model_path, ref_model_out_vector)
    output_tensorrt_dir = os.path.join(args.onnx_out_model_path, 'tf-trt' + mode_q_str)

    if args.convert_pytorch_to_tf_models:
        if not os.path.exists(args.onnx_out_model_path):
            os.makedirs(args.onnx_out_model_path)
        if not os.path.exists(os.path.join(args.onnx_out_model_path, pb_model_path)):
            os.makedirs(os.path.join(args.onnx_out_model_path, pb_model_path))

    if load_pytorch_ref_model:
        # load+running pytorch model
        model_name = args.run_config_name  # "mobilenet_v2_256_win" #"mobilenet_v2" #"resnet" #"squeezenet"
        model, input_size = initialize_model(model_name, num_classes=2, feature_extract=True,
                                             pretrained_type={'source': 'imagenet', 'path': None})

        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint)
        if torch.cuda.is_available():
            if memory_format_torch_channels_last:
                model = model.to(memory_format=torch.channels_last)
            # print(model)
            model = model.to(device)

        model.eval()
        with torch.no_grad():
            if memory_format_torch_channels_last:
                dummy_input = dummy_input.to(memory_format=torch.channels_last)  #print(dummy_input.is_contiguous(memory_format=torch.channels_last)) print(dummy_input.stride())
            dummy_input = dummy_input.to(device)
            dummy_output = model(dummy_input)
            print("Output tensor PyTorch model".format(dummy_output))
            # remove model from GPU
            dummy_output = dummy_output.detach().cpu()
            with open(path_to_ref_model_out_vector, 'wb') as f:
                np.save(f, np.array(dummy_output))

    #pytorch to ONNX

    if args.convert_pytorch_to_tf_models or args.convert_tf_to_tf_models:
# for conversion enough to set the env variable but for inferenc eit contradict the "tf.config.set_visible_devices(physical_devices[args.gpu_id], 'GPU')
#         os.environ["CUDA_VISIBLE_DEVICES"] = '"' + str(args.gpu_id) + '"'
#         import tensorflow as tf
#         from tensorflow.python.compiler.tensorrt import trt_convert as trt

        if args.tf_model_type == 'tf':

            print("Load pytorch model and convert to ONNX and ONNX to TF")
            # copy paste to dest production folder
            with open(os.path.join(args.onnx_out_model_path, meta_data_file_name), 'w') as f:
                json.dump(model_metadata, f)

            # model_path = args.model_path
            # model_to_export = load_model(model_path)

            save_onnx_model(model=model, dummy_input=dummy_input,
                                         model_path=args.onnx_out_model_path,
                                         model_name=onnx_model_name)

            del model
            with torch.cuda.device(device):
                torch.cuda.empty_cache()

            # Load ONNX model and convert to TensorFlow format
            model_onnx = onnx.load(path_to_onnx)
            onnx.checker.check_model(model_onnx)
            # print(model_onnx.graph.input)
            # print(model_onnx.graph.output)
    #
    # supports multi batch in ONNX
            model_onnx.graph.input[0].type.tensor_type.shape.dim[0].dim_param = '?'
            #onnx-tf 1.5.0 Nov.12 2019
            if tf.__version__ > '2.0':
                physical_devices = tf.config.list_physical_devices('GPU')
                tf.config.set_visible_devices(physical_devices[args.gpu_id], 'GPU')
            # tf_rep = prepare(model=model_onnx, device='GPU', logging_level='DEBUG')   # device='CUDA' only in ver 1.7 we have 1.5  https://github.com/onnx/onnx-tensorflow/releases
            tf_rep = prepare(model=model_onnx, device='CUDA', logging_level='DEBUG')   # got warning  : WARNING:tensorflow:From /home/hanoch/.local/share/virtualenvs/blind_quality_36_venv/lib/python3.6/site-packages/onnx_tf/handlers/backend/reshape.py:31: sparse_to_dense (from tensorflow.python.ops.sparse_ops) is deprecated and will be removed in a future version. Instructions for updating:Create a `tf.sparse.SparseTensor` and use `tf.sparse.to_dense` instead. WARNING:tensorflow:From /home/hanoch/.local/share/virtualenvs/blind_quality_36_venv/lib/python3.6/site-packages/onnx_tf/handlers/backend/global_average_pool.py:15: calling reduce_mean (from tensorflow.python.ops.math_ops) with keep_dims is deprecated and will be removed in a future version. Instructions for updating:
    #TODO try ti run the model here : output = prepare(onnx_model).run(input)
            # Export model as .pb file
            if tf.__version__ == '1.12.0':
                tf_rep.export_graph(path_to_pb)  #  change this line to shape=[None, None, None, 3], it worked for me. sig = [tf.TensorSpec(shape=[1, None, None, 3],
            if tf.__version__ == '1.15.0':
                tf_rep.export_graph(path_to_pb.split('.pb')[0])
            elif tf.__version__ > '2.0':
                print("saving graph to {}".format(os.path.join(args.onnx_out_model_path, pb_model_path)))
                tf_rep.export_graph(os.path.join(args.onnx_out_model_path, pb_model_path))
            else:
                raise
            # Input nodes to the model
            print('inputs:', tf_rep.inputs)

            # Output nodes from the model
            print('outputs:', tf_rep.outputs)

            # All nodes in the model
            print('tensor_dict:')
            print(tf_rep.tensor_dict)    # imported_model = tf.saved_model.load(os.path.join(args.onnx_out_model_path, pb_model_path))
# tflite over an existing TF model
        elif args.tf_model_type == 'tflite':
        # Tflite
            if tf.__version__ > '2.0':
                tf.keras.backend.clear_session()
                physical_devices = tf.config.list_physical_devices('GPU')
                tf.config.set_visible_devices(physical_devices[args.gpu_id], 'GPU')
            elif tf.__version__ == '1.12.0':
                tf.device('/gpu:' + str(args.gpu_id))
            else:
                raise
            if tf.__version__ == '1.12.0':
                converter = tf.contrib.lite.TFLiteConverter.from_frozen_graph(path_to_pb,# TensorFlow freezegraph .pb model file
                                                                        input_arrays=['test_input'],# name of input arrays as defined in torch.onnx.export function before.
                                                                        output_arrays=['test_output']# name of output arrays defined in torch.onnx.export function before.
                                                                        )
                # tell converter which type of optimization techniques to use
                # to view the best option for optimization read documentation of tflite about optimization
                # go to this link https://www.tensorflow.org/lite/guide/get_started#4_optimize_your_model_optional
                # To let toco_from_protos be avaialble define in Pycharm for this session under Environment variables the PATH which is concatenation : PATH=/home/hanoch/.local/share/virtualenvs/blind_quality_36_venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/usr/local/cuda/bin:/home/hanoch/.local/share/virtualenvs/blind_quality_36_venv


                converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
                converter.allow_custom_ops = True
                if quantization:
                    converter.quantized_input_stats = {'input': (0., 1.)}  # mean, std_dev (input range is [-1, 1])  : https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/r1/convert/python_api.md#checkpoints
                    converter.inference_type = tf.int8  # this is the recommended type.
                    # converter.inference_input_type=tf.uint8 # optional
                    # converter.inference_output_type=tf.uint8 # optional
                tflite_quantized_model = converter.convert()
                # Save the model.
                with open(path_to_tflie, 'wb') as f:
                    f.write(tflite_quantized_model)

                tf_lite_model = converter.convert()  # toco_from_protos  : it doesn't hep :converter.experimental_new_converter = True + downgraded numpy 1.19.4 to 1.16.4 + but still this version of TFlite doeant support NCHW
            elif tf.__version__ > '2.0':
                # only from SAvedModel type
                # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
                converter = tf.lite.TFLiteConverter.from_saved_model(os.path.join(args.onnx_out_model_path, pb_model_path))
                converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS] # what is tf.lite.OpsSet.SELECT_TF_OPS?

                # following the guide :https://towardsdatascience.com/my-journey-in-converting-pytorch-to-tensorflow-lite-d244376beed
                if 1:
                    converter.experimental_new_converter = True
                # converter.inference_input_type = tf.float32
                # converter.inference_output_type = tf.float32
                if quantization:
                    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                    converter.inference_input_type = tf.int8  # or tf.uint8
                    converter.inference_output_type = tf.int8  # or tf.uint8

                tflite_quantized_model = converter.convert()
                print("Save model tflite to path {}".format(tflite_quantized_model))
                with open(path_to_tflie, 'wb') as f:
                    f.write(tflite_quantized_model)

                return
        elif args.tf_model_type == 'tensor-rt':
            # tf.keras.backend.clear_session()
            # physical_devices = tf.config.list_physical_devices('GPU')
            from tensorflow.python.client import device_lib
            print(device_lib.list_local_devices())
            #TODO: can you try setting this env variable to see if that help? TF_DEBUG_TRT_ALLOW_INEFFICIENT_TRANSPOSE=1  :https://github.com/tensorflow/tensorrt/issues/195
            create_tftrt_engine_offline = False

            if quantization:
                conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
                conversion_params = conversion_params._replace(use_calibration=True)
                conversion_params = conversion_params._replace(precision_mode="INT8") #max_workspace_size_bytes=1<<30
                conversion_params = conversion_params._replace(
                    max_workspace_size_bytes=(1 << 30))
                conversion_params = conversion_params._replace(maximum_cached_engines=1)
            else:
                if 1:
                # 2.2.4. TF-TRT 2.0 Workflow With A SavedModel : https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html
                    conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
                    conversion_params = conversion_params._replace(
                        max_workspace_size_bytes=(1 << 24))
                    conversion_params = conversion_params._replace(precision_mode="FP16")
                    conversion_params = conversion_params._replace(maximum_cached_engines=100)
                    # conversion_params = conversion_params._replace(max_batch_size=8) # added 15/2 no change
                else:
                    conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
                    conversion_params = conversion_params._replace(precision_mode="FP32") #max_workspace_size_bytes=1<<30
                    conversion_params = conversion_params._replace(max_workspace_size_bytes=(1 << 26))
                    # conversion_params = conversion_params._replace(minimum_segment_size=5) # max_batch_size
                    conversion_params = conversion_params._replace(maximum_cached_engines=16)

            converter = trt.TrtGraphConverterV2(
                input_saved_model_dir=os.path.join(args.onnx_out_model_path, pb_model_path),
                conversion_params=conversion_params)

            def my_input_fn():
                # Input for a single inference call, for a network that has two input tensors:
                batch_input = tf.convert_to_tensor(dummy_input[0:1, :, :, :].numpy()) #dummy_input[0:1, :, :, :].cpu()
                yield (batch_input)
            def my_calibration_input_fn():
                # Input for a single inference call, for a network that has two input tensors:
                np_tens = dummy_input[0:1, :, :, :].numpy() * 255.0
                batch_input = tf.convert_to_tensor(np_tens.astype('int')) #dummy_input[0:1, :, :, :].cpu()
                yield (batch_input)

            if quantization:
                converter.convert(calibration_input_fn=my_calibration_input_fn)
            else:
                converter.convert()

            # converter.convert(calibration_input_fn=my_input_fn) # for INT8 only
            if create_tftrt_engine_offline:
                converter.build(input_fn=my_input_fn) # for optimizing TensorRT engines during graph optimization
            print("Save Tf-TRT model {}".format(output_tensorrt_dir))
            converter.save(output_tensorrt_dir)
            return

        else:
            raise NameError("Conversion option isn;t supported {}".format(args.tf_model_type))
############   END of converting models to other frameworks
    else:
        # import tensorflow as tf # from now on GPU handling is embedded in code and not by env var
        if 1:
            # 16 times the batch[16]
            batch_input = torch.cat(2 * [dummy_input.cpu()])
            # batch_input = torch.cat(1 * [dummy_input.cpu()]) # was 2*
            # batch_input = torch.cat(1 * [dummy_input[:8, :, :, :].cpu()]) # was 2*
            # batch_input = torch.cat(1 * [dummy_input[:4, :, :, :].cpu()]) # was 2*
            print("*********  Batch size : {}".format(batch_input.shape[0]))

        else:
            batch_input = dummy_input.cpu()
#load tflite/TF/tensor-rt model
        if args.tf_model_type == 'tflite':
            if tf.__version__ >= '2.0':
                print('')
                # os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # according to post
                tf.keras.backend.clear_session()
                physical_devices = tf.config.list_physical_devices('GPU')
                tf.config.set_visible_devices(physical_devices[args.gpu_id], 'GPU')

            elif tf.__version__ == '1.12.0':
                tf.device('/gpu:' + str(args.gpu_id))
            else:
                raise
            interpreter = tf.compat.v1.lite.Interpreter(model_path=path_to_tflie)
            # interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            # interpreter.set_tensor(input_details[0]['index'], batch_input)

            interpreter.resize_tensor_input(input_details[0]['index'], batch_input.shape)
            interpreter.allocate_tensors()
            interpreter.set_tensor(input_details[0]['index'], tf.convert_to_tensor(batch_input.numpy()))

            model_run_time = list()
            for i in range(10):
                tock0 = time.perf_counter()
                interpreter.invoke()
                print(interpreter.get_tensor(output_details[0]['index']))  # printing the result
                tock1 = time.perf_counter()

                model_run_time += [tock1 - tock0]
                print("Finished tensor process %.4e" % (model_run_time[-1]))

            with open(path_to_ref_model_out_vector, 'rb') as f:
                dummy_output = np.load(f)
            diff_tf_pytorch = dummy_output[0, :] - output
            diff_tf_pytorch = np.sum(np.abs(diff_tf_pytorch))
            print("Bit exactness test pyTorch TF model output error {} Batch:{}".format(diff_tf_pytorch, batch_input.shape[0]))


        elif args.tf_model_type == 'tensor-rt':
            # os.environ["TF_DEBUG_TRT_ALLOW_INEFFICIENT_TRANSPOSE"] = "1"
            # TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            #TODO TF_DEBUG_TRT_ALLOW_INEFFICIENT_TRANSPOSE=1  https://github.com/tensorflow/tensorrt/issues/200
            memory_fraction = 0.05 # 0.05 goes with batch of 8
            total_memory = get_gpu_total_memory()
            # tf.keras.backend.clear_session()
            physical_devices = tf.config.list_physical_devices('GPU')
            tf.config.set_visible_devices(physical_devices[args.gpu_id], 'GPU')

            tf.config.experimental.set_virtual_device_configuration(
                physical_devices[args.gpu_id],
                [tf.config.experimental.VirtualDeviceConfiguration(
                    memory_limit=int(memory_fraction * total_memory[args.gpu_id]))])

            # Load converted model and infer
            saved_model_loaded = tf.saved_model.load(output_tensorrt_dir, tags=[tag_constants.SERVING])
            infer = saved_model_loaded.signatures['serving_default']
            print(infer)
            print(type(infer))
            model_run_time = list()
            for i in range(10):
                tock0 = time.perf_counter()
                output = infer(tf.convert_to_tensor(batch_input.numpy()))['output_0']
                # https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/guide/checkpoint.ipynb
                tock1 = time.perf_counter()
                model_run_time += [tock1 - tock0]
                print("Finished tensor process %.4e" % (model_run_time[-1]))

            with open(path_to_ref_model_out_vector, 'rb') as f:
                dummy_output = np.load(f)
            diff_tf_pytorch = dummy_output[0, :] - output
            diff_tf_pytorch = np.sum(np.abs(diff_tf_pytorch))
            print("Bit exactness test pyTorch TF model output error {} Batch:{}".format(diff_tf_pytorch, batch_input.shape[0]))

            # graph_func = saved_model_loaded.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
            # frozen_func = convert_to_constants.convert_variables_to_constants_v2(graph_func)
            # output = frozen_func(input_data)[0].numpy()

            if 0: #https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html
                model = tf.saved_model.load(output_tensorrt_dir)
                func = root.signatures['serving_default']
                output = func(batch_input)

        elif args.tf_model_type == 'tf': # TF
            # with tf.device('/gpu:2'):
            if tf.__version__ == '1.12.0' or tf.__version__ == '1.15.0': #tf.__version__ == '1.12.0' was added to resolve remove!!!
                tf_graph, y = load_pb(path_to_pb)
            elif tf.__version__ >= '2.0':
                cuda_mps_service = False
                if cuda_mps_service:
                    memory_fraction = 1
                else:
                    memory_fraction = 0.05  # 0.05

                total_memory = get_gpu_total_memory()
                tf.keras.backend.clear_session()
                physical_devices = tf.config.list_physical_devices('GPU')
                tf.config.set_visible_devices(physical_devices[args.gpu_id], 'GPU')

                tf.config.experimental.set_virtual_device_configuration(
                    physical_devices[args.gpu_id],
                    [tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=int(memory_fraction * total_memory[args.gpu_id]))])

                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(physical_devices), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
                if cuda_mps_service:
                    run_mps(savedmodel_path=os.path.join(args.onnx_out_model_path, pb_model_path),
                            batch_input=batch_input)
                else:
                    tf_graph = tf.saved_model.load(os.path.join(args.onnx_out_model_path, pb_model_path))
                # tf_graph = load_pb_tfgpu_1_14_onnxtf_1_5(
                #     os.path.join(args.onnx_out_model_path, pb_model_path, 'saved_model.pb')) # default saves model as saved_model.pb under the given path in export
                # tf_graph = load_pb_tfgpu_1_14_onnxtf_1_5(path_to_pb)
            else:
                raise

            #if optimization:
                # Frozen graph optimization : https://medium.com/@sebastingarcaacosta/how-to-export-a-tensorflow-2-x-keras-model-to-a-frozen-and-optimized-graph-39740846d9eb
                #python -m tensorflow.python.tools.optimize_for_inference --input /hdd/hanoch/runmodels/img_quality/export_onnx_var_batch_tf1_12/saved_state_mobilenet_v2_256_win_n_lyrs___1609764290_pb/saved_state_mobilenet_v2_256_win_n_lyrs___1609764290.pb --output /hdd/hanoch/runmodels/img_quality/export_onnx_var_batch_tf1_12/optimize_for_inference/optimized.pb --frozen_graph=True --input_names='test_input' --output_names='test_output'
                # TODO : Quantize graph : --mode=weights_rounded
            if tf.__version__ == '1.12.0' or tf.__version__ == '1.15.0':
                if 1:
                    config = tf.ConfigProto(log_device_placement=False,
                                            gpu_options=tf.GPUOptions(visible_device_list=repr(args.gpu_id).encode("utf-8"),
                                            per_process_gpu_memory_fraction=0.05))
                else:
                    config = tf.ConfigProto(log_device_placement=False,
                                            gpu_options=tf.GPUOptions(
                                                visible_device_list=repr(args.gpu_id).encode("utf-8")))
            elif tf.__version__ >= '2.0':
                print(tf.__version__)
            else:
                raise

            model_run_time = list()
            if tf.__version__ == '1.12.0' or tf.__version__ == '1.15.0':
                sess = tf.Session(graph=tf_graph, config=config)#Device memory is reserved
                # sess = tf.compat.v1.Session(graph=tf_graph, config=config)

                # for op in tf_graph.get_operations():
                #     print(op.values())

                output_tensor = tf_graph.get_tensor_by_name('test_output:0')  # from some reason it doesn't find the node name!!
                input_tensor = tf_graph.get_tensor_by_name('test_input:0')
                for i in range(10):
                    tock0 = time.perf_counter()
                    output = sess.run(output_tensor, feed_dict={input_tensor: batch_input})
                    tock1 = time.perf_counter()
                    model_run_time += [tock1 - tock0]
                    print("Finished tensor process %.4e" % (model_run_time[-1]))

            elif tf.__version__ >= '2.0':
                infer = tf_graph.signatures['serving_default']
                if 0:
                    print(list(tf_graph.signatures.keys()))
                    print(tf_graph.signatures['serving_default'].structured_input_signature)
                    print(infer.structured_outputs)
                # from some reason on TF2.4 it doesn't get PyTorch tensor

                for i in range(10):
                    tock0 = time.perf_counter()
                    output = infer(tf.convert_to_tensor(batch_input.numpy()))['output_0']
#https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/guide/checkpoint.ipynb
                    tock1 = time.perf_counter()
                    model_run_time += [tock1 - tock0]
                    print("Finished tensor process %.4e" % (model_run_time[-1]))
        # bit exactness test
            with open(path_to_ref_model_out_vector, 'rb') as f:
                dummy_output = np.load(f)
            diff_tf_pytorch = dummy_output[0, :] - output
            diff_tf_pytorch = np.sum(np.abs(diff_tf_pytorch))
            print("Bit exactness test pyTorch TF model output error {} Batch:{}".format(diff_tf_pytorch, batch_input.shape[0]))

if 0:
    with tf.Session(graph=tf_graph) as sess:
        output = sess.run(output_tensor, feed_dict={input_tensor: dummy_input.cpu()})
    print(output)
    diff_tf_pytorch = dummy_output.cpu() - output
    diff_tf_pytorch = diff_tf_pytorch.abs()
    diff_tf_pytorch = diff_tf_pytorch.abs().sum()

    print("Bit exactness test pyTorch TF model output error {}".format(diff_tf_pytorch))

    """
# replace the input tensor with fixed size as in the ONNX model to placeholder supporting variable batch size
        input_shape = (None, 3, None, None)

        input_tensor = tf.placeholder(
            dtype=tf.float32, shape=input_shape, name='test_input')

        [g2] = tf.import_graph_def(tf_graph.as_graph_def(), input_map={'test_input:0': input_tensor},
                                   return_elements=['test_output:0'])

        # tf.reset_default_graph()
        with tf.Session() as sess:
            print('with Dataset:')
            try:
                # batch of 64
                batch_input = torch.cat(16 * [dummy_input.cpu()])
                # batch_tensor = tf.concat([input_tensor] * 64, axis=0)
                output_batch = sess.run(g2, feed_dict={input_tensor: batch_input})

                print(output_batch.shape)
                diff_tf_pytorch = dummy_output.cpu() - output_batch
                diff_tf_pytorch = diff_tf_pytorch.abs()
                diff_tf_pytorch = diff_tf_pytorch.abs().sum()
                print("Bit exactness test pyTorch TF model output error {}".format(diff_tf_pytorch))

            except tf.errors.OutOfRangeError:
                pass

        # Show tensor names in graph

        [g2] = tf.import_graph_def(tf_graph.as_graph_def(), input_map={'test_input:0': input_tensor},
                                   return_elements=['test_output:0'])

        tf.reset_default_graph()
        with tf.Session() as sess:
            print('with Dataset:')
            try:
                output = sess.run(g2, feed_dict={input_tensor: dummy_input.cpu()})
                print(output)
                diff_tf_pytorch = dummy_output.cpu() - output
                diff_tf_pytorch = diff_tf_pytorch.abs()
                diff_tf_pytorch = diff_tf_pytorch.abs().sum()
                print("Bit exactness test pyTorch TF model output error {}".format(diff_tf_pytorch))
    # # batch of 64
    #             batch_input = torch.cat(64 * [dummy_input.cpu()])
    #             batch_tensor = tf.concat([input_tensor] * 64, axis=0)
    #             output_batch = sess.run(g2, feed_dict={batch_tensor: batch_input})

            except tf.errors.OutOfRangeError:
                pass

        if 0:
            output_tensor = tf_graph.get_tensor_by_name('test_output:0')
            input_tensor = tf_graph.get_tensor_by_name('test_input:0')
        # pay attention that the Tf model include the dat normalization built in !!!
            with tf.Session(graph=tf_graph) as sess:
                output = sess.run(output_tensor, feed_dict={input_tensor: dummy_input.cpu()})
            print(output)
            diff_tf_pytorch = dummy_output.cpu() - output
            diff_tf_pytorch = diff_tf_pytorch.abs()
            diff_tf_pytorch = diff_tf_pytorch.abs().sum()

            print("Bit exactness test pyTorch TF model output error {}".format(diff_tf_pytorch))
            batch_input = torch.cat(64 * [dummy_input.cpu()])
            batch_tensor = tf.concat([input_tensor] * 64, axis=0)
            with tf.Session(graph=tf_graph) as sess:
                output = sess.run(output_tensor, feed_dict={batch_tensor: batch_input})
    
    """

if __name__ == '__main__':
    main()


"""
mobilenet_v2_256_win_n_lyrs --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/clip_grad_new_norm/saved_state_mobilenet_v2_256_win_n_lyrs___1606063570 --onnx-out-model-path /hdd/hanoch/runmodels/img_quality/export_onnx_models2 --gpu-id 2

python -u ./src/CNN/production/export_to_onnx_tf.py mobilenet_v2_256_win_n_lyrs --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/clip_grad_new_norm/fitjar/saved_state_mobilenet_v2_256_win_n_lyrs___1609764290 --onnx-out-model-path /hdd/hanoch/runmodels/img_quality/export_onnx_var_batch --metadata-json-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/clip_grad_new_norm/fitjar/mobilenet_v2_256_win_n_lyrs_1609764290.json --gpu-id 2

mobilenet_v2_256_win_n_lyrs --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/clip_grad_new_norm/fitjar/saved_state_mobilenet_v2_256_win_n_lyrs___1609764290 --onnx-out-model-path /hdd/hanoch/runmodels/img_quality/export_onnx_var_batch --metadata-json-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/clip_grad_new_norm/fitjar/mobilenet_v2_256_win_n_lyrs_1609764290.json --gpu-id 2

https://www.xspdf.com/resolution/50346271.html

Get tensor name in the graph 
tensors_per_node = [node.values() for node in graph.get_operations()]
tensor_names = [tensor.name for tensors in tensors_per_node for tensor in tensors]

print(tf.contrib.graph_editor.get_tensors(tf_graph))

#usage:  
# --convert-pytorch-to-tf-models : convert pytorch or run inference
# -- convert-tf-to-tf_models : convert tf model previously created by PyTorch, into tflite/Tensor-rt
# --convert-pytorch-to-tf-models and -- convert-tf-to-tf_models should work simultanuously
# -tf-model-type options  tflite/tensor_rt over an existing TF model
# use venv:
TF 2.x DL2: source /home/hanoch/GIT/Finders/Finder2/bin/activate
TF 1.12 : DL1: source /home/hanoch/.local/share/virtualenvs/blind_quality_36_venv/bin/activate
MPS CUDA:
tf_graph = tf.saved_model.load(os.path.join(args.onnx_out_model_path, pb_model_path))
infer = tf_graph.signatures['serving_default']
infer2 = tf_graph2.signatures['serving_default']
output = infer(tf.convert_to_tensor(batch_input.numpy()))['output_0']
output2 = infer2(tf.convert_to_tensor(batch_input.numpy()))['output_0']


tf_graph = tf.saved_model.load(os.path.join(args.onnx_out_model_path, pb_model_path))
tf_graph2 = tf.saved_model.load(os.path.join(args.onnx_out_model_path, pb_model_path))
tf_graph3 = tf.saved_model.load(os.path.join(args.onnx_out_model_path, pb_model_path))
tf_graph4 = tf.saved_model.load(os.path.join(args.onnx_out_model_path, pb_model_path))
tf_graph5 = tf.saved_model.load(os.path.join(args.onnx_out_model_path, pb_model_path))
tf_graph6 = tf.saved_model.load(os.path.join(args.onnx_out_model_path, pb_model_path))
tf_graph7 = tf.saved_model.load(os.path.join(args.onnx_out_model_path, pb_model_path))
tf_graph8 = tf.saved_model.load(os.path.join(args.onnx_out_model_path, pb_model_path))
tf_graph9 = tf.saved_model.load(os.path.join(args.onnx_out_model_path, pb_model_path))
tf_graph10 = tf.saved_model.load(os.path.join(args.onnx_out_model_path, pb_model_path))

infer = tf_graph.signatures['serving_default']
infer2 = tf_graph2.signatures['serving_default']
infer3 = tf_graph3.signatures['serving_default']
infer4 = tf_graph4.signatures['serving_default']
infer5 = tf_graph5.signatures['serving_default']
infer6 = tf_graph6.signatures['serving_default']
infer7 = tf_graph7.signatures['serving_default']
infer8 = tf_graph8.signatures['serving_default']
infer9 = tf_graph9.signatures['serving_default']
infer10 = tf_graph10.signatures['serving_default']
output = infer(tf.convert_to_tensor(batch_input.numpy()))['output_0']
output2 = infer2(tf.convert_to_tensor(batch_input.numpy()))['output_0']
output3 = infer3(tf.convert_to_tensor(batch_input.numpy()))['output_0']
output4 = infer4(tf.convert_to_tensor(batch_input.numpy()))['output_0']
output5 = infer5(tf.convert_to_tensor(batch_input.numpy()))['output_0']
output6 = infer6(tf.convert_to_tensor(batch_input.numpy()))['output_0']
output7 = infer7(tf.convert_to_tensor(batch_input.numpy()))['output_0']
output8 = infer8(tf.convert_to_tensor(batch_input.numpy()))['output_0']
output9 = infer9(tf.convert_to_tensor(batch_input.numpy()))['output_0']
output10 = infer10(tf.convert_to_tensor(batch_input.numpy()))['output_0']

inference tf1.2
mobilenet_v2_256_win_n_lyrs --model-path /hdd/hanoch/runmodels/img_quality/export_onnx_var_batch_tf1_12/saved_state_mobilenet_v2_256_win_n_lyrs___1609764290_pb/saved_state_mobilenet_v2_256_win_n_lyrs___1609764290.pb --metadata-json-path /hdd/hanoch/runmodels/img_quality/export_onnx_var_batch_tf1_12/saved_state_mobilenet_v2_256_win_n_lyrs___1609764290.json --onnx-out-model-path /hdd/hanoch/runmodels/img_quality/export_onnx_var_batch_tf1_12 --tf-model-type tf

Create model tf 1.15
mobilenet_v2_256_win_n_lyrs --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/clip_grad_new_norm/fitjar/saved_state_mobilenet_v2_256_win_n_lyrs___1609764290 --metadata-json-path /hdd/hanoch/runmodels/img_quality/export_onnx_var_batch_tf1_15/saved_state_mobilenet_v2_256_win_n_lyrs___1609764290.json --onnx-out-model-path /hdd/hanoch/runmodels/img_quality/export_onnx_var_batch_tf1_15 --tf-model-type tf --convert-pytorch-to-tf-models
Run inference tf 1.15
mobilenet_v2_256_win_n_lyrs --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/clip_grad_new_norm/fitjar/saved_state_mobilenet_v2_256_win_n_lyrs___1609764290 --metadata-json-path /hdd/hanoch/runmodels/img_quality/export_onnx_var_batch_tf1_15/saved_state_mobilenet_v2_256_win_n_lyrs___1609764290.json --onnx-out-model-path /hdd/hanoch/runmodels/img_quality/export_onnx_var_batch_tf1_15 --tf-model-type tf

Run inference tf 2.4
python -u ./src/CNN/production/export_to_onnx_tf.py mobilenet_v2_256_win_n_lyrs --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/clip_grad_new_norm/fitjar/saved_state_mobilenet_v2_256_win_n_lyrs___1609764290 --metadata-json-path /hdd/hanoch/runmodels/img_quality/export_onnx_var_batch_tf1_15/saved_state_mobilenet_v2_256_win_n_lyrs___1609764290.json --onnx-out-model-path /hdd/hanoch/runmodels/img_quality/export_onnx_TF2p4 --tf-model-type tf
Improved model over concencus data 
python -u ./dev/production/export_to_onnx_tf.py mobilenet_v2_256_win_n_lyrs --model-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/new_db_7818_bin_class_penn_regularization/saved_state_mobilenet_v2_256_win_n_lyrs___1625844658 --metadata-json-path /hdd/hanoch/runmodels/img_quality/results/mobilenet_v2_256_win_n_lyrs/new_db_7818_bin_class_penn_regularization/mobilenet_v2_256_win_n_lyrs_1625844658.json --onnx-out-model-path /hdd/hanoch/runmodels/img_quality/export_onnx_TF2p51625844658 --tf-model-type tf
"""
#TODO: try pytorch2keras https://learnopencv.com/pytorch-to-tensorflow-model-conversion/
#TODO how does pytorch NCHW to TF NHWC since tf_rep.tensor_dict L tf.Tensor 'BatchNormalization_1/add_1:0' shape=(?, 32, 128, 128) dtype=float32>, same NCHW
