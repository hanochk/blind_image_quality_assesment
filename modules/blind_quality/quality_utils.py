import tensorflow as tf
import numpy as np
from skimage import measure
import subprocess as sp
from collections import namedtuple
Point = namedtuple('Point', ['x', 'y'])

def tile_mapping(tile_shape, cutout_shape, outline, ref_edges='center'):
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



def load_pb(path_to_pb):
    with tf.gfile.GFile(path_to_pb, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        y = tf.import_graph_def(graph_def, name='')
        return graph, y

    # input_shape = (None, 3, None, None)
    # input_tensor = tf.placeholder(
    #     dtype=tf.float32, shape=input_shape, name='test_input')
    # with tf.Graph().as_default() as graph:
    #     tf.import_graph_def(graph_def, name='', input_map={"test_input": input_tensor})
    #     return graph

def get_gpu_total_memory():
    sp_output = sp.check_output(["nvidia-smi", "--query-gpu=memory.total", "--format=csv"])
    total_memory = [int(x.split(' ')[0]) for x in sp_output.decode().split('\n') if
                    'MiB' in x and x.split(' ')[0].isnumeric()]
    return total_memory


def get_gpu_free_memory():
    sp_output = sp.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv"])
    free_memory = [int(x.split(' ')[0]) for x in sp_output.decode().split('\n') if
                   'MiB' in x and x.split(' ')[0].isnumeric()]

    return free_memory

def check_if_gpu():
    is_cuda_gpu_available = tf.test.is_gpu_available(cuda_only=True)
    return is_cuda_gpu_available