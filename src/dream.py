import numpy as np
import PIL.Image
import tensorflow as tf
from download import download_model_if_not_exists
from helpers import plot_image_from_array
import argparse, os, sys


imagenet_mean = 117.0

model_file = download_model_if_not_exists('data')

graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)

with tf.gfile.FastGFile(model_file, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

t_input = tf.placeholder(np.float32, name='input')
t_preprocessed = tf.expand_dims(t_input - imagenet_mean, 0)
tf.import_graph_def(graph_def, {'input': t_preprocessed})

layers = [op.name for op in graph.get_operations() if op.type == 'Conv2D' and 'import/' in op.name]
feature_nums = [int(graph.get_tensor_by_name(name + ':0').get_shape()[-1]) for name in layers]


def print_features():
    print('Available layers:')
    for layer in layers:
        print(layer)
    print('Number of layers', len(layers))
    print('Total number of feature channels:', sum(feature_nums))


def tffunc(*argtypes):
    placeholders = list(map(tf.placeholder, argtypes))
    def wrap(f):
        out = f(*placeholders)
        def wrapper(*args, **kw):
            session = kw.get('session')
            return out.eval(dict(zip(placeholders, args)), session=session)
        return wrapper
    return wrap


def resize(img, size):
    img = tf.expand_dims(img, 0)
    return tf.image.resize_bilinear(img, size)[0, :, :, :]


def calc_grad_tiled(img, t_grad, tile_size=512):
    h, w = img.shape[:2]
    sx, sy = np.random.randint(tile_size, size=2)
    img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
    grad = np.zeros_like(img)
    for y in range(0, max(h - tile_size // 2, tile_size), tile_size):
        for x in range(0, max(w - tile_size // 2, tile_size), tile_size):
            sub = img_shift[y:y + tile_size, x:x + tile_size]
            g = sess.run(t_grad, { t_input: sub })
            grad[y:y + tile_size, x:x + tile_size] = g
    return np.roll(np.roll(grad, -sx, 1), -sy, 0)


resize = tffunc(np.float32, np.int32)(resize)


def render_deepdream(t_obj, image_noise, iter_n, step, octave_n, octave_scale):
    if image_noise is None:
        image_noise = np.random.uniform(size=(224, 224, 3)) + 100.0
    t_score = tf.reduce_mean(t_obj)
    t_grad = tf.gradients(t_score, t_input)[0]
    img = image_noise
    octaves = []
    for _ in range(octave_n - 1):
        hw = img.shape[:2]
        lo = resize(img, np.int32(np.float32(hw) / octave_scale))
        hi = img - resize(lo, hw)
        img = lo
        octaves.append(hi)
    for octave in range(octave_n):
        if octave > 0:
            hi = octaves[-octave]
            img = resize(img, hi.shape[:2]) + hi
        for _ in range(iter_n):
            g = calc_grad_tiled(img, t_grad)
            img += g * (step / (np.abs(g).mean() + 1e-7))
    plot_image_from_array(img / 255.0)

parser = argparse.ArgumentParser(description='DeepDream')
parser.add_argument('--input_image', metavar='i', type=str, help='Path of input image')
parser.add_argument('--layer', metavar='l', type=str, help='Name of the layer')
parser.add_argument('--channel', metavar='c', type=int, help='Channel')
parser.add_argument('--iterations', metavar='t', type=int, help='Iterations')
parser.add_argument('--octaves', metavar='o', type=int, help='Octaves')
parser.add_argument('--octaves_scale', metavar='h', type=int, help='Octaves scale')
parser.add_argument('--step', metavar='s', type=float, help='Step size')

args = parser.parse_args()

layer = args.layer if args.layer is not None else 'mixed4d_3x3_bottleneck_pre_relu'
channel = args.channel if args.channel is not None else 200

if not os.path.isfile(args.input_image):
    print('Input image does not exist')
    sys.exit(-1)

local_image = PIL.Image.open(args.input_image)
local_image = np.float32(local_image)
layer_tensor = graph.get_tensor_by_name(f'import/{layer}:0')
render_deepdream(
    tf.square(layer_tensor),
    local_image,
    args.iterations if args.iterations is not None else 2,
    args.step,
    args.octaves,
    args.octaves_scale)