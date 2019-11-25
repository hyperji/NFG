# -*- coding: utf-8 -*-
# @Time    : 19-9-14 下午11:16
# @Author  : HeJi
# @FileName: NFG.py
# @E-mail: hj@jimhe.cn

print("NFG reforged")

import tensorflow as tf
from tensorflow.python.keras.utils import conv_utils

def conv_output_length_valid(L, ksize, stride):
    return (L-ksize)//stride + 1


class NFG4img(tf.keras.layers.Layer):
    def __init__(self, ksize, strides, d_neuron, dk, dv, Nh, dact, final_dim, padding='SAME'):
        """
        :param ksize: 窗口大小
        :param strides: 步长大小
        :param d_neuron: 神经元编码长度
        :param dk: key depth
        :param dv: value depth
        :param Nh: num of head
        :param dact: 激活量函数深度
        :param final_dim: final output dims
        :param padding: 是否padding 类型(和ＣＮＮ一样)
        """
        super(NFG4img, self).__init__()

        #define the neuron encodings
        self.neuron_embeddings = self.add_weight(name="neuron_embeddings", shape=(ksize, ksize, d_neuron),
                                                 dtype=tf.float32,
                                                 initializer=tf.keras.initializers.glorot_uniform(),
                                                 trainable=True)

        self.dkh = dk // Nh
        self.dvh = dv // Nh
        self.dacth = dact // Nh
        self.Nh = Nh

        self.wkq = tf.keras.layers.Dense(dk + dk, activation=None)


        self.wv = tf.keras.layers.Dense(dv)

        self.wgactq = tf.keras.layers.Dense(dact, activation=None)
        self.wgactk = tf.keras.layers.Dense(dact, activation=None)

        self.n_neurons = ksize * ksize
        self.d_neuron = d_neuron
        self.dv = dv
        self.dk = dk
        self.dact = dact

        self.wf = tf.keras.layers.Dense(final_dim)

        self.ksize = ksize
        self.strides = strides
        self.final_dim = final_dim
        self._ksize = conv_utils.normalize_tuple(self.ksize, 2, name="kernel_size")
        self._strides = conv_utils.normalize_tuple(self.strides, 2, name="strides")
        self.padding = padding
        self._padding = conv_utils.normalize_padding(self.padding)
        self.max_pool = tf.keras.layers.MaxPool1D(ksize**2)

    def split_heads_2d(self, inputs, Nh):
        """Split channels into multiple heads."""
        s = inputs.shape[:-1].as_list()
        n_channels = inputs.shape[-1]
        ret_shape = s + [Nh, n_channels // Nh]
        ret_shape[0] = tf.shape(inputs)[0]
        split = tf.reshape(inputs, ret_shape)
        return tf.transpose(split, [0, 3, 1, 2, 4])

    def combine_heads_2d(self, inputs):
        """Combine heads (inverse of split heads 2d)"""
        transposed = tf.transpose(inputs, [0, 2, 3, 1, 4])
        a, b = transposed.shape[-2:]
        ret_shape = transposed.shape[:-2].as_list() + [a * b]
        ret_shape[0] = tf.shape(transposed)[0]
        return tf.reshape(transposed, ret_shape)

    def compute_neural_structures(self):
        H, W, _ = self.neuron_embeddings.shape
        kq = self.wkq(self.neuron_embeddings)
        kq = tf.expand_dims(kq, axis=0)
        k, q = tf.split(kq, [self.dk, self.dk], axis=3)

        # print("k", k)
        k = self.split_heads_2d(k, Nh=self.Nh)
        q = self.split_heads_2d(q, Nh=self.Nh)

        flat_k = tf.reshape(k, [self.Nh, H * W, self.dkh])
        flat_q = tf.reshape(q, [self.Nh, H * W, self.dkh])

        #logits = tf.matmul(flat_q, flat_k, transpose_b=True)  # [128, 10, 1, 196]
        # print("logit", logits.shape)
        logits = flat_q + tf.transpose(flat_k, [0, 2, 1])
        #logits *= self.dkh ** -0.5
        structures = tf.nn.softmax(logits)
        #structures = tf.expand_dims(structures, axis=0)
        return structures

    def compute_activation_amounts(self, inps):
        H, W, _ = self.neuron_embeddings.shape
        _, vH, vW, _ = inps.shape

        actk = self.wgactk(inps)
        actq = self.wgactq(self.neuron_embeddings)
        actq = tf.expand_dims(actq, axis=0)

        actk = self.split_heads_2d(actk, Nh=self.Nh)
        #print("actq1", actq.shape)
        actq = self.split_heads_2d(actq, Nh=self.Nh)
        #print("actq2", actq.shape)

        flat_actk = tf.reshape(actk, [tf.shape(inps)[0], self.Nh, vH*vW, self.dacth])
        flat_actq = tf.reshape(actq, [1, self.Nh, H*W, self.dacth])

        #print("flat_actk", flat_actk.shape)
        #print("flat_actq", flat_actq.shape)

        #glogits = tf.matmul(flat_actq, flat_actk, transpose_b=True) # [128, 10, 9, 196]

        glogits = flat_actq[:, :, -2:-1, :] + tf.transpose(flat_actk, [0, 1, 3, 2])
        #print("glogits after matmul", glogits.shape)

        #glogits = glogits[:, :, -2:-1, :] #选取最后一个神经元的激活量作为整个神经功能团的激活量
        #print("glogits after slining", glogits.shape)

        #glogits *= self.dacth ** -0.5

        glogits = tf.nn.relu(glogits)

        sum_glogits = tf.reduce_sum(glogits, axis=-1, keepdims=True)
        #print("sum_glogits", sum_glogits.shape)
        activations = (tf.cast(vH*vW, tf.float32) / (sum_glogits + 1e-6)) * glogits
        #print("activations with norm", activations.shape)

        activations = tf.transpose(activations, [0, 1, 3, 2])
        #gweights = tf.reshape(gweights, [tf.shape(inps)[0], self.Nh, vH, vW, H*W])
        #print("activations after transpose", activations.shape)
        #print("inps_shape", [tf.shape(inps)[0], self.Nh, vH, vW, 1])
        activations = tf.reshape(activations, [tf.shape(inps)[0], self.Nh, vH, vW, 1])
        #gweights = self.combine_heads_2d(gweights)
        #print("#"*100)

        return activations

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            if self._padding == 'same':
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self._ksize[i],
                    padding=self._padding,
                    stride=self._strides[i],
                    dilation=0)
            elif self._padding == 'valid':
                new_dim = conv_output_length_valid(space[i], self._ksize[i], self._strides[i])
            new_space.append(new_dim)
        return tf.TensorShape([input_shape[0]] + new_space +
                              [self.final_dim])

    def get_config(self):
        base_config = super(NFG4img, self).get_config()
        #base_config['output_dim'] = self.output_dim
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inps):
        '''
        inps has shape [B, H, W, C]

        '''

        inps_shape = tf.shape(inps)

        batch_size = inps_shape[0]

        neural_structures = self.compute_neural_structures() #计算神经结构矩阵
        neural_structures = tf.expand_dims(neural_structures, axis=0)
        activation_amounts = self.compute_activation_amounts(inps) #计算激活量
        v = self.wv(inps)
        #print("v after wv", v.shape)

        v = self.split_heads_2d(v, Nh=self.Nh)
        v = tf.multiply(v, activation_amounts)
        v = self.combine_heads_2d(v)

        v = tf.image.extract_patches(
            images=v, sizes=[1, self.ksize, self.ksize, 1], padding=self.padding,
            strides=[1, self.strides, self.strides, 1], rates=[1, 1, 1, 1]) #把图像裁成小块（像CNN一样一块一块的算）
        #print("v after extract patches", v.shape)

        new_H = v.shape[1]
        new_W = v.shape[2]
        #v = tf.reshape(v, [batch_size, new_H, new_W, self.ksize ** 2, self.dv])
        v = tf.reshape(v, [batch_size * new_H * new_W, self.ksize ** 2, 1, self.dv])
        v = self.split_heads_2d(v, Nh=self.Nh)
        v = tf.reshape(v, [batch_size * new_H * new_W, self.Nh, self.ksize ** 2, self.dvh])

        neural_structures = neural_structures[:, :, 0:1, :]#以最后一个神经元的输出作为输出
        neural_structures = tf.transpose(neural_structures, [0, 1, 3, 2])

        v = tf.reduce_sum(tf.multiply(neural_structures, v), axis=2) #用神经结构左乘经过激活和变换后的输入

        v = tf.reshape(v, [batch_size, new_H, new_W, self.dv])

        v = self.wf(v) #最后的变换
        return v

class NFG4img_v2(tf.keras.layers.Layer):
    def __init__(self, ksize, strides, d_neuron, dk, dv, Nh, dact, final_dim, padding='SAME'):
        """
        :param ksize: 窗口大小
        :param strides: 步长大小
        :param d_neuron: 神经元编码长度
        :param dk: key depth
        :param dv: value depth
        :param Nh: num of head
        :param dact: 激活量函数深度
        :param final_dim: final output dims
        :param padding: 是否padding 类型(和ＣＮＮ一样)
        """
        super(NFG4img_v2, self).__init__()

        #define the neuron encodings
        self.neuron_embeddings = self.add_weight(name="neuron_embeddings", shape=(ksize, ksize, d_neuron),
                                                 dtype=tf.float32,
                                                 initializer=tf.keras.initializers.glorot_uniform(),
                                                 trainable=True)

        self.dkh = dk // Nh
        self.dvh = dv // Nh
        self.dacth = dact // Nh
        self.Nh = Nh

        self.wkq = tf.keras.layers.Dense(dk + dk, activation=None)


        self.wv = tf.keras.layers.Dense(dv)

        self.wgactq = tf.keras.layers.Dense(dact, activation=None)
        self.wgactk = tf.keras.layers.Dense(dact, activation=None)

        self.n_neurons = ksize * ksize
        self.d_neuron = d_neuron
        self.dv = dv
        self.dk = dk
        self.dact = dact

        self.wf = tf.keras.layers.Dense(final_dim)

        self.ksize = ksize
        self.strides = strides
        self.final_dim = final_dim
        self._ksize = conv_utils.normalize_tuple(self.ksize, 2, name="kernel_size")
        self._strides = conv_utils.normalize_tuple(self.strides, 2, name="strides")
        self.padding = padding
        self._padding = conv_utils.normalize_padding(self.padding)
        self.max_pool = tf.keras.layers.MaxPool1D(ksize**2)

    def split_heads_2d(self, inputs, Nh):
        """Split channels into multiple heads."""
        s = inputs.shape[:-1].as_list()
        n_channels = inputs.shape[-1]
        ret_shape = s + [Nh, n_channels // Nh]
        ret_shape[0] = tf.shape(inputs)[0]
        split = tf.reshape(inputs, ret_shape)
        return tf.transpose(split, [0, 3, 1, 2, 4])

    def combine_heads_2d(self, inputs):
        """Combine heads (inverse of split heads 2d)"""
        transposed = tf.transpose(inputs, [0, 2, 3, 1, 4])
        a, b = transposed.shape[-2:]
        ret_shape = transposed.shape[:-2].as_list() + [a * b]
        ret_shape[0] = tf.shape(transposed)[0]
        return tf.reshape(transposed, ret_shape)

    def compute_neural_structures(self):
        H, W, _ = self.neuron_embeddings.shape
        kq = self.wkq(self.neuron_embeddings)
        kq = tf.expand_dims(kq, axis=0)
        k, q = tf.split(kq, [self.dk, self.dk], axis=3)

        # print("k", k)
        k = self.split_heads_2d(k, Nh=self.Nh)
        q = self.split_heads_2d(q, Nh=self.Nh)

        flat_k = tf.reshape(k, [self.Nh, H * W, self.dkh])
        flat_q = tf.reshape(q, [self.Nh, H * W, self.dkh])

        #logits = tf.matmul(flat_q, flat_k, transpose_b=True)  # [128, 10, 1, 196]
        # print("logit", logits.shape)
        logits = flat_q + tf.transpose(flat_k, [0, 2, 1])
        #logits *= self.dkh ** -0.5
        structures = tf.nn.softmax(logits)
        #structures = tf.expand_dims(structures, axis=0)
        return structures

    def compute_activation_amounts(self, inps):
        H, W, _ = self.neuron_embeddings.shape
        _, vH, vW, _ = inps.shape

        actk = self.wgactk(inps)
        actq = self.wgactq(self.neuron_embeddings)
        actq = tf.expand_dims(actq, axis=0)

        actk = self.split_heads_2d(actk, Nh=self.Nh)
        #print("actq1", actq.shape)
        actq = self.split_heads_2d(actq, Nh=self.Nh)
        #print("actq2", actq.shape)

        flat_actk = tf.reshape(actk, [tf.shape(inps)[0], self.Nh, vH*vW, self.dacth])
        flat_actq = tf.reshape(actq, [1, self.Nh, H*W, self.dacth])

        #print("flat_actk", flat_actk.shape)
        #print("flat_actq", flat_actq.shape)

        #glogits = tf.matmul(flat_actq, flat_actk, transpose_b=True) # [128, 10, 9, 196]

        glogits = flat_actq[:, :, -2:-1, :] + tf.transpose(flat_actk, [0, 1, 3, 2])
        #print("glogits after matmul", glogits.shape)

        #glogits = glogits[:, :, -2:-1, :] #选取最后一个神经元的激活量作为整个神经功能团的激活量
        #print("glogits after slining", glogits.shape)

        #glogits *= self.dacth ** -0.5

        glogits = tf.nn.relu(glogits)

        sum_glogits = tf.reduce_sum(glogits, axis=-1, keepdims=True)
        #print("sum_glogits", sum_glogits.shape)
        activations = (tf.cast(vH*vW, tf.float32) / (sum_glogits + 1e-6)) * glogits
        #print("activations with norm", activations.shape)

        activations = tf.transpose(activations, [0, 1, 3, 2])
        #gweights = tf.reshape(gweights, [tf.shape(inps)[0], self.Nh, vH, vW, H*W])
        #print("activations after transpose", activations.shape)
        #print("inps_shape", [tf.shape(inps)[0], self.Nh, vH, vW, 1])
        activations = tf.reshape(activations, [tf.shape(inps)[0], self.Nh, vH, vW, 1])
        #gweights = self.combine_heads_2d(gweights)
        #print("#"*100)

        return activations

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            if self._padding == 'same':
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self._ksize[i],
                    padding=self._padding,
                    stride=self._strides[i],
                    dilation=0)
            elif self._padding == 'valid':
                new_dim = conv_output_length_valid(space[i], self._ksize[i], self._strides[i])
            new_space.append(new_dim)
        return tf.TensorShape([input_shape[0]] + new_space +
                              [self.final_dim])

    def get_config(self):
        base_config = super(NFG4img, self).get_config()
        #base_config['output_dim'] = self.output_dim
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inps):
        '''
        inps has shape [B, H, W, C]

        '''

        inps_shape = tf.shape(inps)

        batch_size = inps_shape[0]

        neural_structures = self.compute_neural_structures() #计算神经结构矩阵
        neural_structures = tf.expand_dims(neural_structures, axis=0)
        activation_amounts = self.compute_activation_amounts(inps) #计算激活量
        v = self.wv(inps)
        #print("v after wv", v.shape)

        v = self.split_heads_2d(v, Nh=self.Nh)
        v = tf.multiply(v, activation_amounts)
        v = self.combine_heads_2d(v)

        v = tf.image.extract_patches(
            images=v, sizes=[1, self.ksize, self.ksize, 1], padding=self.padding,
            strides=[1, self.strides, self.strides, 1], rates=[1, 1, 1, 1]) #把图像裁成小块（像CNN一样一块一块的算）
        #print("v after extract patches", v.shape)

        new_H = v.shape[1]
        new_W = v.shape[2]
        #v = tf.reshape(v, [batch_size, new_H, new_W, self.ksize ** 2, self.dv])
        v = tf.reshape(v, [batch_size * new_H * new_W, self.ksize ** 2, 1, self.dv])
        v = self.split_heads_2d(v, Nh=self.Nh)
        v = tf.reshape(v, [batch_size * new_H * new_W, self.Nh, self.ksize ** 2, self.dvh])

        neural_structures = neural_structures[:, :, 0:1, :]#以最后一个神经元的输出作为输出
        neural_structures = tf.transpose(neural_structures, [0, 1, 3, 2])

        v = tf.reduce_sum(tf.multiply(neural_structures, v), axis=2) #用神经结构左乘经过激活和变换后的输入

        v = tf.reshape(v, [batch_size, new_H, new_W, self.dv])

        v = self.wf(v) #最后的变换
        return v



class Aggregator(tf.keras.layers.Layer):
    def __init__(self, ksize, strides, padding):
        super(Aggregator, self).__init__()
        self.ksize = ksize
        self.strides = strides
        self.padding = padding

        self._ksize = conv_utils.normalize_tuple(self.ksize, 2, name="kernel_size")
        self._strides = conv_utils.normalize_tuple(self.strides, 2, name="strides")
        self.padding = padding
        self._padding = conv_utils.normalize_padding(self.padding)


    def call(self, x):
        inps_pitches = tf.image.extract_patches(
            images=x, sizes=[1, self.ksize, self.ksize, 1], padding=self.padding,
            strides=[1, self.strides, self.strides, 1], rates=[1, 1, 1, 1])
        return inps_pitches

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self._ksize[i],
                padding=self._padding,
                stride=self._strides[i],
                dilation=0)
            new_space.append(new_dim)
        return tf.TensorShape([input_shape[0]] + new_space +
                                        [self.ksize**2])

    def get_config(self):
        base_config = super(Aggregator, self).get_config()
        #base_config['output_dim'] = self.output_dim
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)