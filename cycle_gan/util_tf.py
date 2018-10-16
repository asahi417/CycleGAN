import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.python.ops import array_ops


def discriminator_patch(input_image,
                        leaky_relu_alpha: float = 0.2,
                        reuse: bool = None,
                        scope: str = None,
                        dim: int = 64,
                        ks: int = 4):

    """ Discriminator (Patch GAN)
    Input 256 x 256 x 3 images and get 32 x 32 x 1 output, which supposed to be logit to be classified by
    discriminator. This can be seen as 32 x 32 patch GAN since the discriminator would try to classify
    piece of image (32 x 32 x 1) instead of entire image.

     Architecture
    --------------
    1. conv (stride: 2, filter: dim, kernel: ks)
    2. conv (stride: 2, filter: dim * 2, kernel: ks)
    3. conv (stride: 2, filter: dim * 4, kernel: ks)
    4. conv (stride: 1, filter: dim * 8, kernel: ks)
    5. conv (stride: 1, filter: 1, kernel: ks)


     Parameter
    ------------
    :param input_image: tensor (batch, 256, 256, 3)
    :param leaky_relu_alpha:
    :param reuse:
    :param scope:
    :param dim: base filter size
    :param ks: kernel size

     Return
    --------------
    :return: logit (batch, 32, 32, 1)
    """

    def leaky_relu(x):
        return tf.maximum(tf.minimum(0.0, leaky_relu_alpha * x), x)

    with tf.variable_scope(scope or "patch_gan_discriminator", reuse=reuse):

        with tf.variable_scope("conv_1"):
            layer = convolution(input_image, weight_shape=[ks, ks, 3, dim], stride=[2, 2])
            layer = leaky_relu(layer)

        with tf.variable_scope("conv_2"):
            layer = convolution(layer, weight_shape=[ks, ks, dim, dim * 2], stride=[2, 2])
            layer = instance_norm(layer)
            layer = leaky_relu(layer)

        with tf.variable_scope("conv_3"):
            layer = convolution(layer, weight_shape=[ks, ks, dim * 2, dim * 4], stride=[2, 2])
            layer = instance_norm(layer)
            layer = leaky_relu(layer)

        with tf.variable_scope("conv_4"):
            layer = convolution(layer, weight_shape=[ks, ks, dim * 4, dim * 8], stride=[1, 1])
            layer = instance_norm(layer)
            layer = leaky_relu(layer)

        with tf.variable_scope("conv_5"):
            layer = convolution(layer, weight_shape=[ks, ks, dim * 8, 1], stride=[1, 1])

    return layer


def generator_resnet(input_image,
                     reuse: bool = None,
                     scope: str = None):
    """ ResNet Generator for `Image to Image`

    :param input_image: tensor, image from source domain (batch, 256, 256, 3)
    :param reuse:
    :param scope:
    :return: tensor as same shape as input_image, converted to target domain
    """

    batch_size = dynamic_batch_size(input_image)
    f_ks = 7  # first conv's kernel size
    ks = 3  # kernel size
    dim = 32  # output ch

    with tf.variable_scope(scope or "resnet_generator", reuse=reuse):

        pad_input = tf.pad(input_image, [[0, 0], [ks, ks], [ks, ks], [0, 0]], mode="REFLECT")

        with tf.variable_scope("conv_1"):
            layer = convolution(pad_input, weight_shape=[f_ks, f_ks, 3, dim], stride=[1, 1], padding='VALID')
            layer = instance_norm(layer)
            layer = tf.nn.relu(layer)

        with tf.variable_scope("conv_2"):
            layer = convolution(layer, weight_shape=[ks, ks, dim, dim*2], stride=[2, 2])
            layer = instance_norm(layer)
            layer = tf.nn.relu(layer)

        with tf.variable_scope("conv_3"):
            layer = convolution(layer, weight_shape=[ks, ks, dim*2, dim * 4], stride=[2, 2])
            layer = instance_norm(layer)
            layer = tf.nn.relu(layer)

        layer = resnet_block(layer, scope='res_block_1')
        layer = resnet_block(layer, scope='res_block_2')
        layer = resnet_block(layer, scope='res_block_3')
        layer = resnet_block(layer, scope='res_block_4')
        layer = resnet_block(layer, scope='res_block_5')
        layer = resnet_block(layer, scope='res_block_6')
        layer = resnet_block(layer, scope='res_block_7')
        layer = resnet_block(layer, scope='res_block_8')
        layer = resnet_block(layer, scope='res_block_9')

        with tf.variable_scope("trans_conv_1"):
            layer = convolution_trans(layer,
                                      weight_shape=[ks, ks, dim*2, dim*4],
                                      output_shape=[batch_size, 128, 128, dim*2],
                                      stride=[2, 2])
            layer = instance_norm(layer)
            layer = tf.nn.relu(layer)

        with tf.variable_scope("trans_conv_2"):
            layer = convolution_trans(layer,
                                      weight_shape=[ks, ks, dim, dim * 2],
                                      output_shape=[batch_size, 256, 256, dim],
                                      stride=[2, 2])
            layer = instance_norm(layer)
            layer = tf.nn.relu(layer)

        with tf.variable_scope("conv_output"):
            layer = convolution(layer, weight_shape=[f_ks, f_ks, dim, 3], stride=[1, 1])
            layer = instance_norm(layer)
            layer = tf.nn.tanh(layer)

        return layer


def resnet_block(x,
                 scope=None,
                 reuse=None):
    """ single resnet block

    :param x: 3-d tensor
    :param scope:
    :param reuse:
    :return: tensor with same shape of x
    """

    dim = x.get_shape()[-1]
    conv_weight = [3, 3, dim, dim]
    with tf.variable_scope(scope or "residual_block", reuse=reuse):

        with tf.variable_scope('conv_1'):
            x_res = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
            # conv
            x_res = convolution(x_res, weight_shape=conv_weight, stride=[1, 1], padding='VALID')
            # instance norm
            x_res = instance_norm(x_res)
            # relu activate
            x_res = tf.nn.relu(x_res)

        with tf.variable_scope('conv_2'):
            x_res = tf.pad(x_res, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
            x_res = convolution(x_res, weight_shape=conv_weight, stride=[1, 1], padding='VALID')
            # instance norm
            x_res = instance_norm(x_res)
            # add residual connection
            x_res = x + x_res
            # relu activate
            x_res = tf.nn.relu(x_res)
        return x_res


def instance_norm(inputs,
                  epsilon=1e-5,
                  scope=None,
                  reuse=None):

    """Instance Normarization"""

    with tf.variable_scope(scope or "instance_norm", reuse=reuse):
        mean, var = tf.nn.moments(inputs, [1, 2], keep_dims=True)
        scale = tf.get_variable('scale',
                                shape=[inputs.get_shape()[-1]],
                                initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02))
        offset = tf.get_variable('offset',
                                 shape=[inputs.get_shape()[-1]],
                                 initializer=tf.constant_initializer(0.0))
        return scale * tf.div(inputs - mean, tf.sqrt(var + epsilon)) + offset


def full_connected(x,
                   weight_shape,
                   scope=None,
                   bias=True,
                   reuse=None):
    """ fully connected layer
    - weight_shape: input size, output size
    - priority: batch norm (remove bias) > dropout and bias term
    """
    with tf.variable_scope(scope or "fully_connected", reuse=reuse):
        w = tf.get_variable("weight", shape=weight_shape, dtype=tf.float32)
        x = tf.matmul(x, w)
        if bias:
            b = tf.get_variable("bias", initializer=[0.0] * weight_shape[-1])
            return tf.add(x, b)
        else:
            return x


def convolution(x,
                weight_shape,
                stride,
                padding="SAME",
                scope=None,
                bias=True,
                reuse=None,
                stddev=0.02):
    """2d convolution
     Parameter
    -------------------
    weight_shape: width, height, input channel, output channel
    stride (list): [stride for axis 1, stride for axis 2]
    """
    with tf.variable_scope(scope or "2d_convolution", reuse=reuse):
        w = tf.get_variable('weight',
                            shape=weight_shape,
                            dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        x = tf.nn.conv2d(x, w, strides=[1, stride[0], stride[1], 1], padding=padding)
        if bias:
            b = tf.get_variable("bias", initializer=[0.0] * weight_shape[-1])
            return tf.add(x, b)
        else:
            return x


def convolution_trans(x,
                      weight_shape,
                      output_shape,
                      stride,
                      padding="SAME",
                      scope=None,
                      bias=True,
                      reuse=None):
    """2d fractinally-strided convolution (transposed-convolution)
     Parameter
    --------------------
    weight_shape: width, height, output channel, input channel
    stride (list): [stride for axis 1, stride for axis 2]
    output_shape (list): [batch, width, height, output channel]
    """
    with tf.variable_scope(scope or "convolution_trans", reuse=reuse):
        w = tf.get_variable('weight', shape=weight_shape, dtype=tf.float32)
        x = tf.nn.conv2d_transpose(x,
                                   w,
                                   output_shape=output_shape,
                                   strides=[1, stride[0], stride[1], 1],
                                   padding=padding,
                                   data_format="NHWC")
        if bias:
            b = tf.get_variable("bias", initializer=[0.0] * weight_shape[2])
            return tf.add(x, b)
        else:
            return x


def dynamic_batch_size(inputs):
    """ Dynamic batch size, which is able to use in a model without deterministic batch size.
    See https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/python/ops/rnn.py
    """
    while nest.is_sequence(inputs):
        inputs = inputs[0]
    return array_ops.shape(inputs)[0]
