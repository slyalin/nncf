"""
 Copyright (c) 2020 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.python.keras import backend
from tensorflow.python.keras import models
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.layers import VersionAwareLayers
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.lib.io import file_io

from examples.tensorflow.common.vgg import VGG16
from examples.tensorflow.common.mobilenet import MobileNet

def mobilenet_v2_100_224(input_shape=None,
                         trainable=True,
                         batch_norm_momentum=0.997,
                         **_):
    handle = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/4"
    model = tf.keras.Sequential(
        [
            hub.KerasLayer(handle=handle, trainable=trainable,
                           arguments=dict(batch_norm_momentum=batch_norm_momentum)),
            tf.keras.layers.Activation('softmax')
        ]
    )

    input_shape = [None] + list(input_shape)
    model.build(input_shape)

    return model


def MobileNetV3(stack_fn, last_point_ch, input_shape=None, model_type='large', **_):
    if input_shape is None:
        input_shape = (None, None, 3)

    if backend.image_data_format() == 'channels_last':
        row_axis, col_axis = (0, 1)
    else:
        row_axis, col_axis = (1, 2)
    rows = input_shape[row_axis]
    cols = input_shape[col_axis]
    if rows and cols and (rows < 32 or cols < 32):
        raise ValueError('Input size must be at least 32x32; got `input_shape=' +
                         str(input_shape) + '`')

    img_input = layers.Input(shape=input_shape)
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

    kernel = 5
    activation = hard_swish
    se_ratio = 0.25

    x = img_input
    x = layers.Rescaling(scale=1. / 127.5, offset=-1.)(x)
    x = layers.Conv2D(
        16,
        kernel_size=3,
        strides=(2, 2),
        padding='same',
        use_bias=False,
        name='Conv')(x)
    x = layers.BatchNormalization(
        axis=channel_axis, epsilon=1e-3,
        momentum=0.999, name='Conv/BatchNorm')(x)
    x = activation(x)

    x = stack_fn(x, kernel, activation, se_ratio)

    last_conv_ch = _depth(backend.int_shape(x)[channel_axis] * 6)

    x = layers.Conv2D(
        last_conv_ch,
        kernel_size=1,
        padding='same',
        use_bias=False,
        name='Conv_1')(x)
    x = layers.BatchNormalization(
        axis=channel_axis, epsilon=1e-3,
        momentum=0.999, name='Conv_1/BatchNorm')(x)
    x = activation(x)

    x = layers.GlobalAveragePooling2D()(x)
    if channel_axis == 1:
        x = layers.Reshape((last_conv_ch, 1, 1))(x)
    else:
        x = layers.Reshape((1, 1, last_conv_ch))(x)

    x = layers.Conv2D(
        last_point_ch,
        kernel_size=1,
        padding='same',
        use_bias=True,
        name='Conv_2')(x)
    x = activation(x)

    x = layers.Dropout(0.2)(x)
    x = layers.Conv2D(1000, kernel_size=1, padding='same', name='Logits')(x)
    x = layers.Flatten()(x)
    x = layers.Activation(activation='softmax',
                          name='Predictions')(x)

    # Create model.
    model = models.Model(img_input, x, name='MobilenetV3{}'.format(model_type))

    BASE_WEIGHT_PATH = ('https://storage.googleapis.com/tensorflow/'
                        'keras-applications/mobilenet_v3/')
    WEIGHTS_HASHES = {
        'large': '59e551e166be033d707958cf9e29a6a7',
        'small': '8768d4c2e7dee89b9d02b2d03d65d862',
    }

    file_name = 'weights_mobilenet_v3_{}_224_1.0_float.h5'.format(model_type)
    file_hash = WEIGHTS_HASHES[model_type]

    weights_path = data_utils.get_file(
        file_name,
        BASE_WEIGHT_PATH + file_name,
        cache_subdir='models',
        file_hash=file_hash)
    model.load_weights(weights_path)

    return model


def MobileNetV3Small(input_shape=None, **kwargs):
    def stack_fn(x, kernel, activation, se_ratio):
        x = _inverted_res_block(x, 1, _depth(16), 3, 2, se_ratio, relu, 0)
        x = _inverted_res_block(x, 72. / 16, _depth(24), 3, 2, None, relu, 1)
        x = _inverted_res_block(x, 88. / 24, _depth(24), 3, 1, None, relu, 2)
        x = _inverted_res_block(x, 4, _depth(40), kernel, 2, se_ratio, activation, 3)
        x = _inverted_res_block(x, 6, _depth(40), kernel, 1, se_ratio, activation, 4)
        x = _inverted_res_block(x, 6, _depth(40), kernel, 1, se_ratio, activation, 5)
        x = _inverted_res_block(x, 3, _depth(48), kernel, 1, se_ratio, activation, 6)
        x = _inverted_res_block(x, 3, _depth(48), kernel, 1, se_ratio, activation, 7)
        x = _inverted_res_block(x, 6, _depth(96), kernel, 2, se_ratio, activation, 8)
        x = _inverted_res_block(x, 6, _depth(96), kernel, 1, se_ratio, activation, 9)
        x = _inverted_res_block(x, 6, _depth(96), kernel, 1, se_ratio, activation, 10)
        return x

    return MobileNetV3(stack_fn, 1024, input_shape, model_type='small', **kwargs)


def MobileNetV3Large(input_shape=None, **kwargs):
    def stack_fn(x, kernel, activation, se_ratio):
        x = _inverted_res_block(x, 1, _depth(16), 3, 1, None, relu, 0)
        x = _inverted_res_block(x, 4, _depth(24), 3, 2, None, relu, 1)
        x = _inverted_res_block(x, 3, _depth(24), 3, 1, None, relu, 2)
        x = _inverted_res_block(x, 3, _depth(40), kernel, 2, se_ratio, relu, 3)
        x = _inverted_res_block(x, 3, _depth(40), kernel, 1, se_ratio, relu, 4)
        x = _inverted_res_block(x, 3, _depth(40), kernel, 1, se_ratio, relu, 5)
        x = _inverted_res_block(x, 6, _depth(80), 3, 2, None, activation, 6)
        x = _inverted_res_block(x, 2.5, _depth(80), 3, 1, None, activation, 7)
        x = _inverted_res_block(x, 2.3, _depth(80), 3, 1, None, activation, 8)
        x = _inverted_res_block(x, 2.3, _depth(80), 3, 1, None, activation, 9)
        x = _inverted_res_block(x, 6, _depth(112), 3, 1, se_ratio, activation, 10)
        x = _inverted_res_block(x, 6, _depth(112), 3, 1, se_ratio, activation, 11)
        x = _inverted_res_block(x, 6, _depth(160), kernel, 2, se_ratio, activation, 12)
        x = _inverted_res_block(x, 6, _depth(160), kernel, 1, se_ratio, activation, 13)
        x = _inverted_res_block(x, 6, _depth(160), kernel, 1, se_ratio, activation, 14)
        return x

    return MobileNetV3(stack_fn, 1280, input_shape, model_type='large', **kwargs)


def relu(x):
    return layers.ReLU()(x)


def hard_sigmoid(x):
    return layers.ReLU(6.)(x + 3.) * (1. / 6.)


def hard_swish(x):
    return layers.Multiply()([hard_sigmoid(x), x])


def _depth(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _se_block(inputs, filters, se_ratio, prefix):
    x = layers.GlobalAveragePooling2D(name=prefix + 'squeeze_excite/AvgPool')(
        inputs)
    if backend.image_data_format() == 'channels_first':
        x = layers.Reshape((filters, 1, 1))(x)
    else:
        x = layers.Reshape((1, 1, filters))(x)
    x = layers.Conv2D(
        _depth(filters * se_ratio),
        kernel_size=1,
        padding='same',
        name=prefix + 'squeeze_excite/Conv')(
        x)
    x = layers.ReLU(name=prefix + 'squeeze_excite/Relu')(x)
    x = layers.Conv2D(
        filters,
        kernel_size=1,
        padding='same',
        name=prefix + 'squeeze_excite/Conv_1')(
        x)
    x = hard_sigmoid(x)
    x = layers.Multiply(name=prefix + 'squeeze_excite/Mul')([inputs, x])
    return x


def _inverted_res_block(x, expansion, filters, kernel_size, stride, se_ratio,
                        activation, block_id):
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    shortcut = x
    prefix = 'expanded_conv/'
    infilters = backend.int_shape(x)[channel_axis]
    if block_id:
        # Expand
        prefix = 'expanded_conv_{}/'.format(block_id)
        x = layers.Conv2D(
            _depth(infilters * expansion),
            kernel_size=1,
            padding='same',
            use_bias=False,
            name=prefix + 'expand')(
            x)
        x = layers.BatchNormalization(
            axis=channel_axis,
            epsilon=1e-3,
            momentum=0.999,
            name=prefix + 'expand/BatchNorm')(
            x)
        x = activation(x)

    if stride == 2:
        x = layers.ZeroPadding2D(
            padding=imagenet_utils.correct_pad(x, kernel_size),
            name=prefix + 'depthwise/pad')(
            x)
    x = layers.DepthwiseConv2D(
        kernel_size,
        strides=stride,
        padding='same' if stride == 1 else 'valid',
        use_bias=False,
        name=prefix + 'depthwise')(
        x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + 'depthwise/BatchNorm')(
        x)
    x = activation(x)

    if se_ratio:
        x = _se_block(x, _depth(infilters * expansion), se_ratio, prefix)

    x = layers.Conv2D(
        filters,
        kernel_size=1,
        padding='same',
        use_bias=False,
        name=prefix + 'project')(
        x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + 'project/BatchNorm')(
        x)

    if stride == 1 and infilters == filters:
        x = layers.Add(name=prefix + 'Add')([shortcut, x])
    return x


def ResNet(stack_fn,
           preact,
           use_bias,
           model_name='resnet',
           include_top=True,
           weights='imagenet',
           input_tensor=None,
           input_shape=None,
           pooling=None,
           classes=1000,
           classifier_activation='softmax',
           **kwargs):
    """Instantiates the ResNet, ResNetV2, and ResNeXt architecture.

    Reference:
    - [Deep Residual Learning for Image Recognition](
        https://arxiv.org/abs/1512.03385) (CVPR 2015)

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    Arguments:
      stack_fn: a function that returns output tensor for the
        stacked residual blocks.
      preact: whether to use pre-activation or not
        (True for ResNetV2, False for ResNet and ResNeXt).
      use_bias: whether to use biases for convolutional layers or not
        (True for ResNet and ResNetV2, False for ResNeXt).
      model_name: string, model name.
      include_top: whether to include the fully-connected
        layer at the top of the network.
      weights: one of `None` (random initialization),
        'imagenet' (pre-training on ImageNet),
        or the path to the weights file to be loaded.
      input_tensor: optional Keras tensor
        (i.e. output of `layers.Input()`)
        to use as image input for the model.
      input_shape: optional shape tuple, only to be specified
        if `include_top` is False (otherwise the input shape
        has to be `(224, 224, 3)` (with `channels_last` data format)
        or `(3, 224, 224)` (with `channels_first` data format).
        It should have exactly 3 inputs channels.
      pooling: optional pooling mode for feature extraction
        when `include_top` is `False`.
        - `None` means that the output of the model will be
            the 4D tensor output of the
            last convolutional layer.
        - `avg` means that global average pooling
            will be applied to the output of the
            last convolutional layer, and thus
            the output of the model will be a 2D tensor.
        - `max` means that global max pooling will
            be applied.
      classes: optional number of classes to classify images
        into, only to be specified if `include_top` is True, and
        if no `weights` argument is specified.
      classifier_activation: A `str` or callable. The activation function to use
        on the "top" layer. Ignored unless `include_top=True`. Set
        `classifier_activation=None` to return the logits of the "top" layer.
      **kwargs: For backwards compatibility only.
    Returns:
      A `keras.Model` instance.

    Raises:
      ValueError: in case of invalid argument for `weights`,
        or invalid input shape.
      ValueError: if `classifier_activation` is not `softmax` or `None` when
        using a pretrained top layer.
    """
    global layers
    if 'layers' in kwargs:
        layers = kwargs.pop('layers')
    else:
        layers = VersionAwareLayers()
    if kwargs:
        raise ValueError('Unknown argument(s): %s' % (kwargs,))
    if not (weights in {'imagenet', None} or file_io.file_exists_v2(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = imagenet_utils.obtain_input_shape(
        input_shape,
        default_size=224,
        min_size=32,
        data_format=backend.image_data_format(),
        require_flatten=include_top,
        weights=weights)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    x = layers.ZeroPadding2D(
        padding=((3, 3), (3, 3)), name='conv1_pad')(img_input)
    x = layers.Conv2D(64, 7, strides=2, use_bias=use_bias, name='conv1_conv')(x)

    if not preact:
        x = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name='conv1_bn')(x)
        x = layers.Activation('relu', name='conv1_relu')(x)

    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
    x = layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)

    x = stack_fn(x)

    if preact:
        x = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name='post_bn')(x)
        x = layers.Activation('relu', name='post_relu')(x)

    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        imagenet_utils.validate_activation(classifier_activation, weights)
        x = layers.Dense(classes, name='dense')(x)
        x = layers.Softmax(name='predictions')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D(name='max_pool')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = layer_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = training.Model(inputs, x, name=model_name)

    # Load weights.
    BASE_WEIGHTS_PATH = (
        'https://storage.googleapis.com/tensorflow/keras-applications/resnet/')
    WEIGHTS_HASHES = {
        'resnet50': ('2cb95161c43110f7111970584f804107',
                     '4d473c1dd8becc155b73f8504c6f6626'),
        'resnet101': ('f1aeb4b969a6efcfb50fad2f0c20cfc5',
                      '88cf7a10940856eca736dc7b7e228a21'),
        'resnet152': ('100835be76be38e30d865e96f2aaae62',
                      'ee4c566cf9a93f14d82f913c2dc6dd0c'),
        'resnet50v2': ('3ef43a0b657b3be2300d5770ece849e0',
                       'fac2f116257151a9d068a22e544a4917'),
        'resnet101v2': ('6343647c601c52e1368623803854d971',
                        'c0ed64b8031c3730f411d2eb4eea35b5'),
        'resnet152v2': ('a49b44d1979771252814e80f8ec446f9',
                        'ed17cf2e0169df9d443503ef94b23b33'),
        'resnext50': ('67a5b30d522ed92f75a1f16eef299d1a',
                      '62527c363bdd9ec598bed41947b379fc'),
        'resnext101':
            ('34fb605428fcc7aa4d62f44404c11509', '0f678c91647380debd923963594981b3')
    }

    if (weights == 'imagenet') and (model_name in WEIGHTS_HASHES):
        if include_top:
            file_name = model_name + '_weights_tf_dim_ordering_tf_kernels.h5'
            file_hash = WEIGHTS_HASHES[model_name][0]
        else:
            file_name = model_name + '_weights_tf_dim_ordering_tf_kernels_notop.h5'
            file_hash = WEIGHTS_HASHES[model_name][1]
        weights_path = data_utils.get_file(
            file_name,
            BASE_WEIGHTS_PATH + file_name,
            cache_subdir='models',
            file_hash=file_hash)
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model


def block1(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
    """A residual block.

    Arguments:
      x: input tensor.
      filters: integer, filters of the bottleneck layer.
      kernel_size: default 3, kernel size of the bottleneck layer.
      stride: default 1, stride of the first layer.
      conv_shortcut: default True, use convolution shortcut if True,
          otherwise identity shortcut.
      name: string, block label.

    Returns:
      Output tensor for the residual block.
    """
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    if conv_shortcut:
        shortcut = layers.Conv2D(
            4 * filters, 1, strides=stride, name=name + '_0_conv')(x)
        shortcut = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = layers.Conv2D(filters, 1, strides=stride, name=name + '_1_conv')(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    x = layers.Conv2D(
        filters, kernel_size, padding='SAME', name=name + '_2_conv')(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)

    x = layers.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_3_bn')(x)

    x = layers.Add(name=name + '_add')([shortcut, x])
    x = layers.Activation('relu', name=name + '_out')(x)
    return x


def stack1(x, filters, blocks, stride1=2, name=None):
    """A set of stacked residual blocks.

    Arguments:
      x: input tensor.
      filters: integer, filters of the bottleneck layer in a block.
      blocks: integer, blocks in the stacked blocks.
      stride1: default 2, stride of the first layer in the first block.
      name: string, stack label.

    Returns:
      Output tensor for the stacked blocks.
    """
    x = block1(x, filters, stride=stride1, name=name + '_block1')
    for i in range(2, blocks + 1):
        x = block1(x, filters, conv_shortcut=False, name=name + '_block' + str(i))
    return x


def block2(x, filters, kernel_size=3, stride=1, conv_shortcut=False, name=None):
    """A residual block.

    Arguments:
        x: input tensor.
        filters: integer, filters of the bottleneck layer.
        kernel_size: default 3, kernel size of the bottleneck layer.
        stride: default 1, stride of the first layer.
        conv_shortcut: default False, use convolution shortcut if True,
          otherwise identity shortcut.
        name: string, block label.

    Returns:
      Output tensor for the residual block.
    """
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    preact = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_preact_bn')(x)
    preact = layers.Activation('relu', name=name + '_preact_relu')(preact)

    if conv_shortcut:
        shortcut = layers.Conv2D(
            4 * filters, 1, strides=stride, name=name + '_0_conv')(preact)
    else:
        shortcut = layers.MaxPooling2D(1, strides=stride)(x) if stride > 1 else x

    x = layers.Conv2D(
        filters, 1, strides=1, use_bias=False, name=name + '_1_conv')(preact)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '_2_pad')(x)
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=stride,
        use_bias=False,
        name=name + '_2_conv')(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)

    x = layers.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
    x = layers.Add(name=name + '_out')([shortcut, x])
    return x


def stack2(x, filters, blocks, stride1=2, name=None):
    """A set of stacked residual blocks.

    Arguments:
        x: input tensor.
        filters: integer, filters of the bottleneck layer in a block.
        blocks: integer, blocks in the stacked blocks.
        stride1: default 2, stride of the first layer in the first block.
        name: string, stack label.

    Returns:
        Output tensor for the stacked blocks.
    """
    x = block2(x, filters, conv_shortcut=True, name=name + '_block1')
    for i in range(2, blocks):
        x = block2(x, filters, name=name + '_block' + str(i))
    x = block2(x, filters, stride=stride1, name=name + '_block' + str(blocks))
    return x


def block3(x,
           filters,
           kernel_size=3,
           stride=1,
           groups=32,
           conv_shortcut=True,
           name=None):
    """A residual block.

    Arguments:
      x: input tensor.
      filters: integer, filters of the bottleneck layer.
      kernel_size: default 3, kernel size of the bottleneck layer.
      stride: default 1, stride of the first layer.
      groups: default 32, group size for grouped convolution.
      conv_shortcut: default True, use convolution shortcut if True,
          otherwise identity shortcut.
      name: string, block label.

    Returns:
      Output tensor for the residual block.
    """
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    if conv_shortcut:
        shortcut = layers.Conv2D(
            (64 // groups) * filters,
            1,
            strides=stride,
            use_bias=False,
            name=name + '_0_conv')(x)
        shortcut = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = layers.Conv2D(filters, 1, use_bias=False, name=name + '_1_conv')(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    c = filters // groups
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '_2_pad')(x)
    x = layers.DepthwiseConv2D(
        kernel_size,
        strides=stride,
        depth_multiplier=c,
        use_bias=False,
        name=name + '_2_conv')(x)
    x_shape = backend.int_shape(x)[1:-1]
    x = layers.Reshape(x_shape + (groups, c, c))(x)
    x = layers.Lambda(
        lambda x: sum(x[:, :, :, :, i] for i in range(c)),
        name=name + '_2_reduce')(x)
    x = layers.Reshape(x_shape + (filters,))(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)

    x = layers.Conv2D(
        (64 // groups) * filters, 1, use_bias=False, name=name + '_3_conv')(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_3_bn')(x)

    x = layers.Add(name=name + '_add')([shortcut, x])
    x = layers.Activation('relu', name=name + '_out')(x)
    return x


def stack3(x, filters, blocks, stride1=2, groups=32, name=None):
    """A set of stacked residual blocks.

    Arguments:
      x: input tensor.
      filters: integer, filters of the bottleneck layer in a block.
      blocks: integer, blocks in the stacked blocks.
      stride1: default 2, stride of the first layer in the first block.
      groups: default 32, group size for grouped convolution.
      name: string, stack label.

    Returns:
      Output tensor for the stacked blocks.
    """
    x = block3(x, filters, stride=stride1, groups=groups, name=name + '_block1')
    for i in range(2, blocks + 1):
        x = block3(
            x,
            filters,
            groups=groups,
            conv_shortcut=False,
            name=name + '_block' + str(i))
    return x


def ResNet50(include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000,
             **kwargs):
    """Instantiates the ResNet50 architecture."""

    def stack_fn(x):
        x = stack1(x, 64, 3, stride1=1, name='conv2')
        x = stack1(x, 128, 4, name='conv3')
        x = stack1(x, 256, 6, name='conv4')
        return stack1(x, 512, 3, name='conv5')

    return ResNet(stack_fn, False, True, 'resnet50', include_top, weights,
                  input_tensor, input_shape, pooling, classes, **kwargs)


def ResNet101(include_top=True,
              weights='imagenet',
              input_tensor=None,
              input_shape=None,
              pooling=None,
              classes=1000,
              **kwargs):
    """Instantiates the ResNet101 architecture."""

    def stack_fn(x):
        x = stack1(x, 64, 3, stride1=1, name='conv2')
        x = stack1(x, 128, 4, name='conv3')
        x = stack1(x, 256, 23, name='conv4')
        return stack1(x, 512, 3, name='conv5')

    return ResNet(stack_fn, False, True, 'resnet101', include_top, weights,
                  input_tensor, input_shape, pooling, classes, **kwargs)


def ResNet152(include_top=True,
              weights='imagenet',
              input_tensor=None,
              input_shape=None,
              pooling=None,
              classes=1000,
              **kwargs):
    """Instantiates the ResNet152 architecture."""

    def stack_fn(x):
        x = stack1(x, 64, 3, stride1=1, name='conv2')
        x = stack1(x, 128, 8, name='conv3')
        x = stack1(x, 256, 36, name='conv4')
        return stack1(x, 512, 3, name='conv5')

    return ResNet(stack_fn, False, True, 'resnet152', include_top, weights,
                  input_tensor, input_shape, pooling, classes, **kwargs)
