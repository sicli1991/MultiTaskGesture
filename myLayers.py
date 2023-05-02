from abc import ABC

import tensorflow as tf
from tensorflow import keras
from dataHandle import config_read_in
import numpy as np
# import tensorflow.python.platform.build_info as build
config = config_read_in()
BatchNormalization = tf.keras.layers.BatchNormalization
Conv2D = tf.keras.layers.Conv2D
Input = tf.keras.layers.Input
MP2D = tf.keras.layers.MaxPooling2D
Activation = tf.keras.layers.Activation
Flatten = tf.keras.layers.Flatten
Dropout = tf.keras.layers.Dropout
Dense = tf.keras.layers.Dense
Model = tf.keras.models.Model
Conv2DTrans = tf.keras.layers.Conv2DTranspose
K_Multiply = tf.keras.layers.Multiply
Concatenate = tf.keras.layers.Concatenate
GlobalAP = tf.keras.layers.GlobalAveragePooling2D
GlobalMP = tf.keras.layers.GlobalMaxPool2D
sigmoid = tf.keras.activations.sigmoid
he_normal = tf.keras.initializers.he_normal
l2 = tf.keras.regularizers.l2
Add = tf.keras.layers.Add

G1_FL = [0, 0, 0, 0, 0]
G2_FL = [0, 1, 0, 0, 0]
G3_FL = [1, 0, 0, 0, 0]
G4_FL = [0, 1, 1, 0, 0]
G5_FL = [1, 1, 1, 1, 1]
G6_FL = [1, 1, 0, 0, 0]
G7_FL = [0, 1, 0, 0, 1]
G8_FL = [1, 0, 0, 0, 1]
G9_FL = [0, 1, 1, 1, 1]
G10_FL = [1, 1, 0, 0, 1]
gesture_finger_list = [G1_FL, G2_FL, G3_FL, G4_FL, G5_FL, G6_FL, G7_FL, G8_FL, G9_FL, G10_FL]
gesture_finger_list = np.array(gesture_finger_list, dtype=np.float32)


def optimizer_generation(opt):
    # print(pm)
    if opt == "SGD":
        pm = config["train_parameters"]["optimizer_SGD"]
        txt = "keras.optimizers.SGD(learning_rate={lr}, momentum={mm}, nesterov={nt})"
        fo = txt.format(lr=pm["learning_rate"], mm=pm["momentum"], nt=pm["nesterov"])
        return eval(fo)
    elif opt == "Adam":
        pm = config["train_parameters"]["optimizer_Adam"]
        txt = "keras.optimizers.Adam(learning_rate={lr}, beta_1={b1}, beta_2={b2}, epsilon={eps}, amsgrad=(ams))"
        fo = txt.format(lr=pm["learning_rate"], b1=pm["beta_1"], b2=pm["beta_2"], eps=pm["epsilon"], ams=["amsgrad"])
        return eval(fo)
    else:
        return None


def conv_batch_act(**pm):
    channel = pm['channels']
    kernel = pm.setdefault('kernel_size', (3, 3))
    stride = pm.setdefault('stride', (1, 1))
    padding = pm.setdefault('padding', 'same')
    initializer = pm.setdefault('initializer', he_normal())
    regularizer = pm.setdefault('regularizer', l2(1e-3))
    activation = pm.setdefault('activation', 'relu')
    # name = pm.setdefault('name', None)
    # dilation_rate = pm.setdefault('dilation_rate', 1)
    # maxpool = pm.setdefault('maxpooling', True)
    # maxpool_size = pm.setdefault('pooling_size', (2, 2))

    def struc(input_layer):
        conv = Conv2D(channel, kernel_size=kernel, strides=stride, padding=padding,
                      kernel_initializer=initializer, kernel_regularizer=regularizer)(input_layer)
        batch = BatchNormalization()(conv)
        act = Activation(activation)(batch)

        return act

    return struc


def de_conv_comb(**pm):
    channel = pm['channels']
    kernel = pm.setdefault('kernel_size', (3, 3))
    stride = pm.setdefault('stride', (1, 1))
    de_stride = pm.setdefault('de_stride', (2, 2))
    padding = pm.setdefault('padding', 'same')
    de_padding = pm.setdefault('de_padding', 'same')
    initializer = pm.setdefault('initializer', he_normal())
    regularizer = pm.setdefault('regularizer', l2(1e-3))
    # activation = pm.setdefault('activation', 'relu')
    # name = pm.setdefault('name', None)
    # dilation_rate = pm.setdefault('dilation_rate', 1)
    combine = pm.setdefault('combine', 'multiply')

    def struc_extra_conv(small_layer, larger_layer):
        conv = Conv2D(channel, kernel_size=kernel, strides=stride, padding=padding,
                      kernel_initializer=initializer, kernel_regularizer=regularizer)(larger_layer)
        de_conv = Conv2DTrans(channel, kernel_size=kernel, strides=de_stride, padding=de_padding)(small_layer)

        if combine == 'multiply':
            output = K_Multiply()([conv, de_conv])  # 12*12*256
            return output
        elif combine == 'concatenate':
            output = Concatenate()([conv, de_conv])
            return output
        elif combine is None:
            return de_conv
        else:
            print("Error in de_conv_up")
            return None

    return struc_extra_conv


def downsampling_conv(**pm):
    channel = pm['channels']
    kernel = pm.setdefault('kernel_size', (3, 3))
    stride = pm.setdefault('stride', (1, 1))
    padding = pm.setdefault('padding', 'same')
    # de_padding = pm.setdefault('de_padding', 'same')
    initializer = pm.setdefault('initializer', he_normal())
    regularizer = pm.setdefault('regularizer', l2(1e-3))
    # activation = pm.setdefault('activation', 'relu')
    name = pm.setdefault('name', None)
    dilation_rate = pm.setdefault('dilation_rate', 1)
    # extra_conv = pm.setdefault('extra_conv', True)
    # combine = pm.setdefault('combine', 'multiply')

    def struc(larger_layer):
        conv = Conv2D(channel, kernel_size=kernel, strides=stride, padding=padding, name=name,
                      kernel_initializer=initializer, kernel_regularizer=regularizer,
                      dilation_rate=dilation_rate)(larger_layer)
        return conv

    return struc


class WeightAdd(keras.layers.Layer):
    def __init__(self, row=1, col=1, dim=1, units="float32", **kwargs):
        self.w = None
        self.row = row
        self.col = col
        self.dim = dim
        self.units = units
        super(WeightAdd, self).__init__()
        self.__class__.__name__ = "myMultiply"

    def build(self, input_shape):
        self.w = self.add_weight(name='kernel',
                                 shape=(self.row, self.col, self.dim),
                                 dtype=self.units,
                                 initializer="random_normal",
                                 trainable=True).numpy()

        super(WeightAdd, self).build(input_shape)

    def get_config(self):
        g_config = super().get_config()
        g_config.update({
            "w": self.w,
            'row': self.row,
            'col': self.col,
            'dim': self.dim,
            'units': self.units
        })
        return g_config

    def call(self, inputs):
        input_base, input_amplifier = inputs
        weighted_input = tf.multiply(input_amplifier, self.w)
        act = Activation('sigmoid')(weighted_input)
        return tf.add(input_base, act)


class gestureTransform(keras.layers.Layer):
    def __init__(self, op_shape, units="float32", **kwargs):
        self.out = None
        self.batch_size = None
        self.op_shape = op_shape
        self.units = units
        self.gsl = gesture_finger_list
        super(gestureTransform, self).__init__()
        self.__class__.__name__ = "GestureTrans"

    def get_config(self):
        g_config = super().get_config()
        g_config.update({
            "gsl": self.gsl,
            'units': self.units,
            'op_shape': self.op_shape
        })
        return g_config

    def build(self, input_shape):
        self.batch_size = input_shape[0]
        final = []
        """
        for b in range(self.batch_size):
            index = tf.keras.backend.argmax(
                b,
                axis=-1
            )
            print(index)
            fm_list = self.gsl[index]
            finger_map = []
            for i in fm_list:
                if i == 1:
                    finger_map.append(tf.ones(self.op_shape, dtype=self.units))
                else:
                    finger_map.append(tf.zeros(self.op_shape, dtype=self.units))
            final.append(finger_map)
        """
        self.final = tf.zeros(self.op_shape, dtype=self.units)
        self.out = tf.reshape(final, [-1, self.op_shape[0], self.op_shape[1], self.op_shape[2]])

        super(gestureTransform, self).build(input_shape)  # Be sure to call this at the end
        
    def call(self, inputs):
        with tf.compat.v1.Session() as session:
            result = session.run(self.out)  # feed_dict={self.final: inputs}
        return self.out


class testly(tf.keras.layers.Layer):

    def __init__(self, row, col, position=5):
        super(testly, self).__init__()
        self.cut_position = position
        self.row = row
        self.col = col
        # self.total = tf.Variable(initial_value=tf.zeros(input_dim), trainable=False)

        self.__class__.__name__ = "testly"

    def call(self, inputs):
        # m = tf.keras.backend.argmax(x, axis=-1)
        # print(m.shape)
        # tf.compat.v1.disable_eager_execution()
        input1, input2 = inputs
        ut = tf.unstack(input1, axis=-1)
        fm = tf.stack(ut[0:self.cut_position], axis=-1)
        wm = ut[-1][:, :, :, np.newaxis]
        print(fm)
        print(input2)
        return tf.multiply(fm, input2)

        # return fm, wm[:, :, :, np.newaxis]


class fingermap_filter(keras.layers.Layer):
    def __init__(self, units="float32", **kwargs):
        self.w = None
        self.units = units
        super(fingermap_filter, self).__init__()
        self.__class__.__name__ = "FM_subtract"

    def get_config(self):
        g_config = super().get_config()
        g_config.update({
            'units': self.units
        })
        return g_config

    def call(self, inputs):
        fm, ges = inputs
        fm_finger, fm_wrist = fm[:, :, :, :5], fm[:, :, :, 5:6]
        ges = ges[:, tf.newaxis, tf.newaxis, :]
        squares = fm_finger * ges
        # squares = tf.multiply(fm_finger, ges)
        act = Activation('sigmoid')(squares)
        res = tf.concat([act, fm_wrist], axis=-1)

        return res


class fingermap_filter_v2(keras.layers.Layer): # noqa
    def __init__(self, units="float32", **kwargs):
        self.w = None
        self.units = units
        super(fingermap_filter_v2, self).__init__()
        self.__class__.__name__ = "FM_filter"

    def get_config(self):
        g_config = super().get_config()
        g_config.update({
            'units': self.units
        })
        return g_config

    def call(self, inputs, *args):
        fm, ges = inputs
        fm_finger, fm_wrist = fm[:, :, :, :5], fm[:, :, :, 5:6]
        ges = ges[:, tf.newaxis, tf.newaxis, :]

        squares = fm_finger * ges
        # squares = tf.multiply(fm_finger, ges)
        act = Activation('sigmoid')(squares)
        fm_res = tf.concat([act, fm_wrist], axis=-1)

        finger_prob = tf.reduce_max(fm_finger, axis=-1)
        return fm_res, finger_prob
