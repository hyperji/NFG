# -*- coding: utf-8 -*-
# @Time    : 19-7-1 下午10:10
# @Author  : HeJi
# @FileName: meta_learning.py
# @E-mail: hj@jimhe.cn

import tensorflow as tf
from NFG_lite import Aggregator, NFG4img_v2, NFG4img
import numpy as np
from StandAlongSelfAtten import SASA
print("meta learning.py, 12.30, 22:25, encoder ksize=3 Nh = 8, relmod ksize=5, agg3 with nfgencoder_v2 protonet with nfgecoder_v3, Nh=8")

class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, out_channels, conv_padding = "SAME", pooling_padding = "VALID"):
        self.conv = tf.keras.layers.Conv2D(filters=out_channels,
                                           kernel_size=3, strides=1,
                                           padding=conv_padding,
                                           use_bias=True)
        self.bn = tf.keras.layers.BatchNormalization(center=True, scale=True)
        self.relu = tf.keras.layers.ReLU()
        self.max_pool = tf.keras.layers.MaxPool2D(2, padding=pooling_padding)

        super(ConvBlock, self).__init__()
    def call(self, x):
        x_ = self.conv(x)
        x_ = self.bn(x_)
        x_ = self.relu(x_)
        x_ = self.max_pool(x_)
        return x_


class CNNEncoder(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, final_dim):
        super(CNNEncoder, self).__init__()
        self.conv1 = ConvBlock(hidden_dim)
        self.conv2 = ConvBlock(hidden_dim)
        self.conv3 = ConvBlock(hidden_dim)
        self.conv4 = ConvBlock(final_dim)

    def call(self, x):
        x_ = self.conv1(x)
        x_ = self.conv2(x_)
        x_ = self.conv3(x_)
        x_ = self.conv4(x_)
        return x_

class CNNEncoder_v2(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, final_dim):
        super(CNNEncoder_v2, self).__init__()
        self.conv1 = ConvBlock(hidden_dim)
        self.conv2 = ConvBlock(hidden_dim)
        self.conv3 = ConvBlock(hidden_dim)
        self.conv4 = ConvBlock(final_dim)
        self.agg3 = Aggregator(ksize=3, strides=1, padding="SAME")
        self.agg2 = Aggregator(ksize=2, strides=1, padding="SAME")

    def call(self, x):
        x_ = self.agg3(x)
        x_ = self.conv1(x_)

        x_ = self.agg2(x_)
        x_ = self.conv2(x_)

        x_ = self.agg2(x_)
        x_ = self.conv3(x_)

        x_ = self.agg2(x_)
        x_ = self.conv4(x_)
        return x_


class NFGBlock(tf.keras.layers.Layer):
    def __init__(self, ksize, strides, d_neuron, dk, dv, Nh, dact, final_dim, padding='SAME'):
        super(NFGBlock, self).__init__()
        self.nfg = NFG4img(ksize, strides, d_neuron, dk, dv, Nh, dact, final_dim, padding)
        self.ln = tf.keras.layers.LayerNormalization()
    def call(self, x):
        x = self.nfg(x)
        x = self.ln(x)
        return x


class NFGBlock_v2(tf.keras.layers.Layer):
    def __init__(self, ksize, strides, d_neuron, dv, Nh, final_dim, padding='SAME'):
        super(NFGBlock_v2, self).__init__()
        self.nfg = NFG4img_v2(
            ksize=ksize, strides=strides, d_neuron=d_neuron, dv=dv, Nh=Nh, final_dim=final_dim, padding=padding)
        self.ln = tf.keras.layers.LayerNormalization()
    def call(self, x):
        x = self.nfg(x)
        x = self.ln(x)
        return x


class NFGEncoder(tf.keras.layers.Layer):
    def __init__(self, h_dim = 64, Nh = 8):
        super(NFGEncoder, self).__init__()
        self.nfg1 = NFGBlock(
            ksize=3, strides=2, d_neuron=h_dim, dk=Nh, dv=h_dim, Nh=Nh, dact=Nh, final_dim=h_dim, padding='SAME')
        self.nfg2 = NFGBlock(
            ksize=3, strides=2, d_neuron=h_dim, dk=Nh, dv=h_dim, Nh=Nh, dact=Nh, final_dim=h_dim, padding='SAME')
        self.nfg3 = NFGBlock(
            ksize=3, strides=2, d_neuron=h_dim, dk=Nh, dv=h_dim, Nh=Nh, dact=Nh, final_dim=h_dim, padding='SAME')
        self.nfg4 = NFGBlock(
            ksize=3, strides=2, d_neuron=h_dim, dk=Nh, dv=h_dim, Nh=Nh, dact=Nh, final_dim=h_dim, padding='VALID')
        self.agg3 = Aggregator(ksize=3, strides=1, padding="SAME")
        self.agg2 = Aggregator(ksize=2, strides=1, padding="SAME")

    def call(self, x):
        x_ = self.agg3(x)
        x_ = self.nfg1(x_)
        #print("X", x_.shape)

        #x_ = self.agg2(x_)
        x_ = self.nfg2(x_)
        #print("X", x_.shape)

        #x_ = self.agg2(x_)
        x_ = self.nfg3(x_)
        #print("X", x_.shape)

        #x_ = self.agg2(x_)
        #x_ = self.ln(x_)
        x_ = self.nfg4(x_)
        #print("X", x_.shape)

        return x_

class NFGEncoder_v2(tf.keras.layers.Layer):
    def __init__(self, h_dim=64, Nh = 8):
        super(NFGEncoder_v2, self).__init__()
        self.nfg1 = NFGBlock_v2(
            ksize=3, strides=2, d_neuron=h_dim, dv=h_dim, Nh=Nh, final_dim=h_dim, padding='SAME')
        self.nfg2 = NFGBlock_v2(
            ksize=3, strides=2, d_neuron=h_dim, dv=h_dim, Nh=Nh, final_dim=h_dim, padding='SAME')
        self.nfg3 = NFGBlock_v2(
            ksize=3, strides=2, d_neuron=h_dim, dv=h_dim, Nh=Nh, final_dim=h_dim, padding='SAME')
        self.nfg4 = NFGBlock_v2(
            ksize=3, strides=2, d_neuron=h_dim, dv=h_dim, Nh=Nh, final_dim=h_dim, padding='VALID')
        self.agg3 = Aggregator(ksize=3, strides=1, padding="SAME")
        self.agg2 = Aggregator(ksize=2, strides=1, padding="SAME")

    def call(self, x):
        x_ = self.agg3(x)
        x_ = self.nfg1(x_)
        #print("X", x_.shape)

        #x_ = self.agg2(x_)
        x_ = self.nfg2(x_)
        #print("X", x_.shape)

        #x_ = self.agg2(x_)
        x_ = self.nfg3(x_)
        #print("X", x_.shape)

        #x_ = self.agg2(x_)
        #x_ = self.ln(x_)
        x_ = self.nfg4(x_)
        #print("X", x_.shape)

        return x_


class NFGEncoder_v3(tf.keras.layers.Layer):
    def __init__(self, h_dim=64, Nh = 8):
        super(NFGEncoder_v3, self).__init__()
        self.nfg1 = NFGBlock_v2(
            ksize=3, strides=2, d_neuron=h_dim, dv=h_dim, Nh=Nh, final_dim=h_dim, padding='SAME')
        self.nfg2 = NFGBlock_v2(
            ksize=3, strides=2, d_neuron=h_dim, dv=h_dim, Nh=Nh, final_dim=h_dim, padding='SAME')
        self.nfg3 = NFGBlock_v2(
            ksize=3, strides=2, d_neuron=h_dim, dv=h_dim, Nh=Nh, final_dim=h_dim, padding='SAME')
        self.nfg4 = NFGBlock_v2(
            ksize=3, strides=2, d_neuron=h_dim, dv=h_dim, Nh=Nh, final_dim=h_dim, padding='VALID')
        self.agg3 = Aggregator(ksize=3, strides=1, padding="SAME")
        self.agg2 = Aggregator(ksize=2, strides=1, padding="SAME")

    def call(self, x):
        x_ = self.agg3(x)
        x_ = self.nfg1(x_)
        #print("X", x_.shape)

        x_ = self.agg2(x_)
        x_ = self.nfg2(x_)
        #print("X", x_.shape)

        x_ = self.agg2(x_)
        x_ = self.nfg3(x_)
        #print("X", x_.shape)

        x_ = self.agg2(x_)
        #x_ = self.ln(x_)
        x_ = self.nfg4(x_)
        #print("X", x_.shape)

        return x_

class SASAEncoder(tf.keras.layers.Layer):
    def __init__(self, ksize, strides, d_neuron, dk, dv, Nh, dact, final_dim, padding='SAME'):
        super(SASAEncoder, self).__init__()
        self.sasa = SASA(ksize, strides, dk, dv, Nh, final_dim, padding='SAME')
        self.ln = tf.keras.layers.LayerNormalization()
    def call(self, x):
        x = self.sasa(x)
        x = self.ln(x)
        return x


class CNNRelationModule(tf.keras.layers.Layer):
    def __init__(self, h_dims = (64, 16)):
        super(CNNRelationModule, self).__init__()
        self.conv1 = ConvBlock(h_dims[0], pooling_padding="SAME")
        self.conv2 = ConvBlock(h_dims[1], pooling_padding="VALID")
        self.fc1 = tf.keras.layers.Dense(8, activation=tf.nn.relu, use_bias=True)
        self.fc2 = tf.keras.layers.Dense(1, activation=tf.sigmoid, use_bias=True)
        self.flatten = tf.keras.layers.Flatten()

    def call(self, x):
        #net = tf.reshape(x, [-1, 5, 5, 64])
        net = self.conv1(x)
        net = self.conv2(net)

        #print("net after conv2", net.shape)

        net = self.flatten(net)

        net = self.fc1(net)
        net = self.fc2(net)

        net = self.flatten(net)
        return net


class NFGRelationModule(tf.keras.layers.Layer):
    def __init__(self, h_dims = (64, 16), Nhs = (8, 2)):
        super(NFGRelationModule, self).__init__()
        self.agg2 = Aggregator(ksize=2, strides=1, padding="SAME")
        self.nfg1 = NFGBlock(ksize=5, strides=2, d_neuron=h_dims[0], dk=Nhs[0], dv=h_dims[0],
                             Nh=Nhs[0], dact=Nhs[0], final_dim=h_dims[0], padding="VALID")
        '''
        self.nfg2 = NFGBlock(ksize=3, strides=2, d_neuron=h_dims[1], dk=Nhs[1], dv=h_dims[1],
                             Nh=Nhs[1], dact=Nhs[1], final_dim=h_dims[1], padding="VALID")
        '''
        self.fc1 = tf.keras.layers.Dense(8, activation=tf.nn.relu, use_bias=True)
        self.fc2 = tf.keras.layers.Dense(1, activation=tf.sigmoid, use_bias=False)
        self.flatten = tf.keras.layers.Flatten()

    def call(self, x):
        #net = tf.reshape(x, [-1, 5, 5, 64])
        #net = self.agg2(net)
        net = self.nfg1(x)
        #print("net after nfg1", net.shape)
        #net = self.nfg2(net)
        #print("net after nfg2", net.shape)

        net = self.flatten(net)

        net = self.fc1(net)
        net = self.fc2(net)

        net = self.flatten(net)
        return net


class NFGRelationModule_v2(tf.keras.layers.Layer):
    def __init__(self, h_dims = (64, 16), Nhs = (8, 2)):
        super(NFGRelationModule_v2, self).__init__()
        self.nfg1 = NFGBlock_v2(ksize=5, strides=2, d_neuron=h_dims[0], dv=h_dims[0],
                                Nh=Nhs[0], final_dim=h_dims[0], padding='VALID')
        '''
        self.nfg2 = NFGBlock_v2(ksize=3, strides=2, d_neuron=h_dims[1], dv=h_dims[1],
                                Nh=Nhs[1], final_dim=Nhs[1], padding="VALID")
        '''
        self.fc1 = tf.keras.layers.Dense(8, activation=tf.nn.relu, use_bias=True)
        self.fc2 = tf.keras.layers.Dense(1, activation=tf.sigmoid, use_bias=True)
        self.flatten = tf.keras.layers.Flatten()

    def call(self, x):
        #net = tf.reshape(x, [-1, 5, 5, 64])
        #net = self.agg2(net)
        net = self.nfg1(x)
        #print("net after nfg1", net.shape)
        #net = self.nfg2(net)
        #print("net after nfg2", net.shape)

        net = self.flatten(net)

        net = self.fc1(net)
        net = self.fc2(net)

        net = self.flatten(net)
        return net


class RelationNets(tf.keras.Model):
    def __init__(self, h_dim, z_dim, encoder_type = "CNN", relation_type = 'NONE'):
        super(RelationNets, self).__init__()
        self.h_dim = h_dim
        self.z_dim = z_dim

        if encoder_type == "CNN":
            self.encoder = CNNEncoder(h_dim, z_dim)
        elif encoder_type == "NFG":
            self.encoder = NFGEncoder_v2(h_dim, Nh=8)  # CNNEncoder4TPN(h_dim, z_dim)
        else:
            raise NotImplementedError
        if relation_type == 'CNN':
            self.relation = CNNRelationModule(h_dims=(64, 16))
        elif relation_type == "NFG":
            self.relation = NFGRelationModule_v2(h_dims = (64, 16), Nhs=(8, 2))
        elif relation_type == "NONE":
            self.relation = None
        else:
            raise NotImplementedError
        #self.flatten = tf.keras.layers.Flatten()

    def call(self, s, q, **kwargs):
        s_shape = s.shape
        q_shape = q.shape
        n_way = s_shape[0]
        n_shot = s_shape[1]
        n_query = q_shape[1]
        # s with shape [n_way, n_shot, W, H, C]
        # q with shape [n_way, n_query, W, H, C]
        y = tf.tile(tf.reshape(
            tf.range(0, n_way), [n_way, 1]), [1, n_query])

        y_one_hot = tf.one_hot(y, depth=n_way, dtype=tf.float32)

        s = tf.reshape(s, [-1, s_shape[2], s_shape[3], s_shape[4]])
        q = tf.reshape(q, [-1, q_shape[2], q_shape[3], q_shape[4]])
        x = tf.concat([s, q], axis=0)

        z = self.encoder(x)
        '''
        ###################################################
        z_ = self.flatten(z)

        z_dim = z_.shape[-1]
        z_proto_ = tf.reduce_mean(tf.reshape(
            z_[:n_way * n_shot], [n_way, n_shot, z_dim]), axis=1)
        zq_ = z_[n_way * n_shot:]

        dists = euclidean_distance(zq_, z_proto_)
        # print("dists.shape", dists.shape)

        _log_dists = tf.nn.log_softmax(-dists, axis=1)
        # print("_log_dists.shape", _log_dists.shape)
        _log_p_y = tf.reshape(_log_dists, [n_way, n_query, -1])

        # preds = tf.cast(tf.argmax(log_p_y, axis=-1), tf.int32)
        # print("preds", preds)

        # masks = tf.cast(tf.equal(preds, y), tf.float32)

        # print("masks", masks)

        proto_loss = -tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.multiply(y_one_hot, _log_p_y), axis=-1), [-1]))
        # acc = tf.reduce_mean(masks)

        ##################################################
        '''

        z_shape = z.shape

        z_proto = tf.reduce_mean(tf.reshape(
            z[:n_way * n_shot], [n_way, n_shot] + z_shape[1:]), axis=1)

        zq = tf.reshape(z[n_way * n_shot:], [n_way * n_query] + z_shape[1:])

        z_proto_ext = tf.tile(tf.expand_dims(z_proto, axis=0), [n_way * n_query, 1, 1, 1, 1])
        zq_ext = tf.tile(tf.expand_dims(zq, axis=0), [n_way, 1, 1, 1, 1])
        zq_ext = tf.transpose(zq_ext, [1,0, 2, 3, 4])

        relation_pairs = tf.concat([z_proto_ext, zq_ext], axis=-1)

        relation_pairs = tf.reshape(relation_pairs, [n_way * n_query * n_way] + z_shape[1:-1] + [z_shape[-1] * 2])

        relations = self.relation(relation_pairs)

        log_p_y = tf.reshape(relations, [n_way, n_query, n_way])
        preds = tf.cast(tf.argmax(log_p_y, axis=-1), tf.int32)
        masks = tf.cast(tf.equal(preds, y), tf.float32)
        acc = tf.reduce_mean(masks)

        ce_loss = tf.reduce_mean(tf.reshape(tf.reduce_sum((y_one_hot - log_p_y) ** 2, axis=-1), [-1]))

        return log_p_y, ce_loss, acc
        #return proto_loss, ce_loss, acc


class TPN_stop_grad(tf.keras.Model):
    def __init__(self, h_dim, z_dim, rn, k, alpha, encoder_type="CNN", relation_type='NONE'):
        super(TPN_stop_grad, self).__init__()
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.rn = rn
        self.k = k
        self.alpha = alpha
        if encoder_type == "CNN":
            self.encoder = CNNEncoder(h_dim, z_dim)
        elif encoder_type == "NFG":
            self.encoder = NFGEncoder_v3(h_dim, Nh=8)
        else:
            raise NotImplementedError
        if relation_type == 'CNN':
            self.relation = CNNRelationModule(h_dims=(64, 16))
        elif relation_type == "NFG":
            self.relation = NFGRelationModule_v2(h_dims=(64, 16), Nhs=(8, 2))
        elif relation_type == "NONE":
            self.relation = None
        else:
            raise NotImplementedError
        if self.rn == 300:  # learned sigma, fixed alpha
            self.alpha = tf.constant(self.alpha)
        else:  # learned sigma and alpha
            self.alpha = tf.Variable(self.alpha, name='alpha', trainable=True)

    def call(self, s, q, **kwargs):
        s_shape = s.shape
        q_shape = q.shape
        num_classes, num_support = s_shape[0], s_shape[1]
        num_queries = q_shape[1]
        ys = tf.tile(tf.reshape(
            tf.range(0, num_classes), [num_classes, 1]), [1, num_support])
        # ys = np.tile(np.arange(num_classes)[:, np.newaxis], (1, num_support)).astype(np.uint8)

        ys_one_hot = tf.one_hot(ys, depth=num_classes)

        y = tf.tile(tf.reshape(
            tf.range(0, num_classes), [num_classes, 1]), [1, num_queries])

        y_one_hot = tf.one_hot(y, depth=num_classes, dtype=tf.float32)

        s = tf.reshape(s, [-1, s_shape[2], s_shape[3], s_shape[4]])
        q = tf.reshape(q, [-1, q_shape[2], q_shape[3], q_shape[4]])

        emb_s = self.encoder(s)
        emb_q = self.encoder(q)

        # TODO this loss is not used in original paper, but in order to reach the acc the original paper, we had to use this
        '''
        z_dim = emb_s.shape[-1]
        z_proto = tf.reduce_mean(tf.reshape(
            emb_s, [num_classes, num_support, z_dim]), axis=1)

        # print("z_proto.shape", z_proto.shape)
        # print("zq.shape", zq.shape)

        dists = euclidean_distance(emb_q, z_proto)
        # print("dists.shape", dists.shape)

        _log_dists = tf.nn.log_softmax(-dists, axis=1)
        # print("_log_dists.shape", _log_dists.shape)
        log_p_y = tf.reshape(_log_dists, [num_classes, num_queries, -1])


        proto_loss = -tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.multiply(y_one_hot, log_p_y), axis=-1), [-1]))


        ###############################################################################################
        #TODO originally this is no stop gradient in original paper
        ce_loss, acc, sigma_value = self.label_prop(tf.stop_gradient(emb_s), tf.stop_gradient(emb_q), ys_one_hot)
        '''
        ce_loss, acc, sigma_value = self.label_prop(emb_s, emb_q, ys_one_hot)
        return sigma_value, ce_loss, acc
        # return proto_loss, ce_loss, acc

    def label_prop(self, x, u, ys):

        # x: NxD, u: UxD
        epsilon = np.finfo(float).eps
        s = tf.shape(ys)
        ys = tf.reshape(ys, (s[0] * s[1], -1))
        Ns, C = tf.shape(ys)[0], tf.shape(ys)[1]
        Nu = tf.shape(u)[0]

        yu = tf.zeros((Nu, C)) / tf.cast(C, tf.float32) + epsilon  # 0 initialization
        # yu = tf.ones((Nu,C))/tf.cast(C,tf.float32)            # 1/C initialization
        y = tf.concat([ys, yu], axis=0)
        gt = tf.reshape(tf.tile(tf.expand_dims(tf.range(C), 1), [1, tf.cast(Nu / C, tf.int32)]), [-1])

        all_un = tf.concat([x, u], axis=0)
        #all_un = tf.reshape(all_un, [-1, 1600])
        #N, d = tf.shape(all_un)[0], tf.shape(all_un)[1]

        # all_un_stop_grad = tf.stop_gradient(all_un)

        # compute graph weights
        if self.rn in [30, 300]:  # compute example-wise sigma
            #all_un = tf.reshape(all_un, [-1, 5, 5, 64])
            self.sigma = self.relation(all_un)

            all_un = tf.reshape(all_un, [-1, 1600])
            #all_un = tf.reshape(all_un, [-1, 2500])
            N, d = tf.shape(all_un)[0], tf.shape(all_un)[1]
            all_un = all_un / (self.sigma + epsilon)
            all1 = tf.expand_dims(all_un, axis=0)
            all2 = tf.expand_dims(all_un, axis=1)
            W = tf.reduce_mean(tf.square(all1 - all2), axis=2)
            W = tf.exp(-W / 2)

        # kNN Graph
        if self.k > 0:
            W = self.topk(W, self.k)

        # Laplacian norm
        D = tf.reduce_sum(W, axis=0)
        D_inv = 1.0 / (D + epsilon)
        D_sqrt_inv = tf.sqrt(D_inv)

        # compute propagated label
        D1 = tf.expand_dims(D_sqrt_inv, axis=1)
        D2 = tf.expand_dims(D_sqrt_inv, axis=0)
        S = D1 * W * D2

        F = tf.linalg.inv(tf.eye(N) - self.alpha * S + epsilon)
        F = tf.matmul(F, y)
        label = tf.argmax(F, axis=1)

        # loss computation
        F = tf.nn.softmax(F)

        y_one_hot = tf.reshape(tf.one_hot(gt, depth=C), [Nu, -1])
        y_one_hot = tf.concat([ys, y_one_hot], axis=0)

        ce_loss = y_one_hot * tf.math.log(F + epsilon)
        ce_loss = tf.negative(ce_loss)
        ce_loss = tf.reduce_mean(tf.reduce_sum(ce_loss, axis=1))

        # only consider query examples acc
        F_un = F[Ns:, :]
        acc = tf.reduce_mean(tf.cast(tf.equal(label[Ns:], tf.cast(gt, tf.int64)), tf.float32))

        return ce_loss, acc, self.sigma

    def topk(self, W, k):
        # construct k-NN and compute margin loss
        values, indices = tf.nn.top_k(W, k, sorted=False)
        my_range = tf.expand_dims(tf.range(0, tf.shape(indices)[0]), axis=1)
        my_range_repeated = tf.tile(my_range, [1, k])
        full_indices = tf.concat([tf.expand_dims(my_range_repeated, axis=2), tf.expand_dims(indices, 2)], axis=2)
        full_indices = tf.reshape(full_indices, [-1, 2])
        full_indices = tf.cast(full_indices, tf.int64)
        sparse_w = tf.sparse.SparseTensor(
            indices=full_indices, values=tf.reshape(values, [-1]), dense_shape=W.shape)
        topk_W = tf.sparse.to_dense(sparse_w, default_value=0, validate_indices=False)

        # topk_W = tf.compat.v1.sparse_to_dense(full_indices, tf.shape(W), tf.reshape(values, [-1]), default_value=0.,
        #                            validate_indices=False)
        ind1 = (topk_W > 0) | (tf.transpose(topk_W) > 0)  # union, k-nearest neighbor
        # ind2 = (topk_W > 0) & (tf.transpose(topk_W) > 0)  # intersection, mutal k-nearest neighbor
        ind1 = tf.cast(ind1, tf.float32)

        topk_W = ind1 * W

        return topk_W



class NFG_Prototypical_Nets(tf.keras.Model):
    def __init__(self, ksize, strides, d_neuron, dk, dv, Nh, dact, final_dim):
        super(NFG_Prototypical_Nets, self).__init__()
        self.agg3 = Aggregator(3, 1, padding="SAME")
        self.agg2 = Aggregator(2, 1, padding="SAME")
        self.nfg1 = SASAEncoder(ksize, strides, d_neuron, dk, dv, Nh, dact, final_dim, padding="SAME")
        self.nfg2 = SASAEncoder(ksize, strides, d_neuron, dk, dv, Nh, dact, final_dim, padding="SAME")
        self.nfg3 = SASAEncoder(ksize, strides, d_neuron, dk, dv, Nh, dact, final_dim, padding="SAME")
        self.nfg4 = SASAEncoder(ksize, strides, d_neuron, dk, dv, Nh, dact, final_dim, padding="VALID")
        self.flatten = tf.keras.layers.Flatten()
        self.ln = tf.keras.layers.LayerNormalization()
        self.loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    def call(self, s, q, **kwargs):
        s_shape = s.shape
        q_shape = q.shape
        n_way = s_shape[0]
        n_shot = s_shape[1]
        n_query = q_shape[1]
        # s with shape [n_way, n_shot, W, H, C]
        # q with shape [n_way, n_query, W, H, C]
        y = tf.tile(tf.reshape(
            tf.range(0, n_way), [n_way, 1]), [1, n_query])

        # print("y", y)
        # print("y.shape", y.shape)
        y_one_hot = tf.one_hot(y, depth=n_way, dtype=tf.float32)

        # print("s.shape", s.shape)

        s = tf.reshape(s, [-1, s_shape[2], s_shape[3], s_shape[4]])
        q = tf.reshape(q, [-1, q_shape[2], q_shape[3], q_shape[4]])
        x = tf.concat([s, q], axis=0)

        x = self.agg3(x)
        x = self.nfg1(x)
        # print("x", x.shape)

        x = self.agg2(x)
        x = self.nfg2(x)
        # print("x", x.shape)

        x = self.agg2(x)
        x = self.nfg3(x)
        # print("x", x.shape)

        x = self.agg2(x)
        x = self.ln(x)
        x = self.nfg4(x)
        # print("x", x.shape)

        z = self.flatten(x)
        # print()

        z_dim = z.shape[-1]
        z_proto = tf.reduce_mean(tf.reshape(
            z[:n_way * n_shot], [n_way, n_shot, z_dim]), axis=1)
        zq = z[n_way * n_shot:]

        # print("z_proto.shape", z_proto.shape)
        # print("zq.shape", zq.shape)

        dists = euclidean_distance(zq, z_proto)
        # print("dists.shape", dists.shape)

        _log_dists = tf.nn.log_softmax(-dists, axis=1)
        # print("_log_dists.shape", _log_dists.shape)
        log_p_y = tf.reshape(_log_dists, [n_way, n_query, -1])

        preds = tf.cast(tf.argmax(log_p_y, axis=-1), tf.int32)
        # print("preds", preds)

        masks = tf.cast(tf.equal(preds, y), tf.float32)

        # print("masks", masks)

        ce_loss = -tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.multiply(y_one_hot, log_p_y), axis=-1), [-1]))
        acc = tf.reduce_mean(masks)

        return log_p_y, ce_loss, acc


class Prototypical_Nets(tf.keras.Model):
    def __init__(self, hidden_dim, final_dim, encoder_type="CNN"):
        super(Prototypical_Nets, self).__init__()
        if encoder_type == "CNN":
            self.encoder = CNNEncoder(hidden_dim, final_dim)
        elif encoder_type == "NFG":
            self.encoder = NFGEncoder_v3(hidden_dim, Nh=8)
        else:
            raise NotImplementedError
        self.flatten = tf.keras.layers.Flatten()

    def call(self, s, q, **kwargs):
        s_shape = s.shape
        q_shape = q.shape
        n_way = s_shape[0]
        n_shot = s_shape[1]
        n_query = q_shape[1]
        # s with shape [n_way, n_shot, W, H, C]
        # q with shape [n_way, n_query, W, H, C]
        y = tf.tile(tf.reshape(
            tf.range(0, n_way), [n_way, 1]), [1, n_query])

        # print("y", y)
        # print("y.shape", y.shape)
        y_one_hot = tf.one_hot(y, depth=n_way, dtype=tf.float32)

        # print("s.shape", s.shape)

        s = tf.reshape(s, [-1, s_shape[2], s_shape[3], s_shape[4]])
        q = tf.reshape(q, [-1, q_shape[2], q_shape[3], q_shape[4]])
        x = tf.concat([s, q], axis=0)
        # x = self.fgsa(x)
        # x = self.ln(x)
        # print("x", x.shape)
        z = self.encoder(x)
        # print("z", z.shape)
        # print("z", z.shape)
        z = self.flatten(z)

        z_dim = z.shape[-1]
        z_proto = tf.reduce_mean(tf.reshape(
            z[:n_way * n_shot], [n_way, n_shot, z_dim]), axis=1)
        zq = z[n_way * n_shot:]

        # print("z_proto.shape", z_proto.shape)
        # print("zq.shape", zq.shape)

        dists = euclidean_distance(zq, z_proto)
        # print("dists.shape", dists.shape)

        _log_dists = tf.nn.log_softmax(-dists, axis=1)
        # print("_log_dists.shape", _log_dists.shape)
        log_p_y = tf.reshape(_log_dists, [n_way, n_query, -1])

        preds = tf.cast(tf.argmax(log_p_y, axis=-1), tf.int32)
        # print("preds", preds)

        masks = tf.cast(tf.equal(preds, y), tf.float32)

        # print("masks", masks)

        ce_loss = -tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.multiply(y_one_hot, log_p_y), axis=-1), [-1]))
        acc = tf.reduce_mean(masks)

        return log_p_y, ce_loss, acc

'''
class MatchingNetwork(tf.keras.Model):
    def __init__(self, hidden_dim, final_dim, encoder_type="CNN", lstm_size=32, batch_size=32):
        super(MatchingNetwork, self).__init__()

        self.batch_size = batch_size

        if encoder_type == "CNN":
            self.encoder = CNNEncoder(hidden_dim, final_dim)
        elif encoder_type == "NFG":
            self.encoder = NFGEncoder_v3(hidden_dim, Nh=8)
        else:
            raise NotImplementedError
        self.flatten = tf.keras.layers.Flatten()

        # Fully contextual embedding

        self.fce_dim = int(np.floor(84 / 16)) ** 2 * 64  # Input LSTM dimenstion
        self.fce = tf.keras.Sequential([
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_size, return_sequences=True))
        ])

    @tf.function
    def call(self, x_support, y_support, x_query, y_query):

        def _calc_cosine_distances(support, query_img):
            """
            Calculate cosine distances between support images and query one.
            Args:
                support (Tensor): Tensor of support images
                query_img (Tensor): query image
            Returns:
            """
            eps = 1e-10
            similarities = tf.zeros([self.support_samples, self.batch],
                                    tf.float32)
            i_sample = 0
            for support_image in support:
                sum_support = tf.reduce_sum(tf.square(support_image), axis=1)
                support_magnitude = tf.math.rsqrt(
                    tf.clip_by_value(sum_support, eps, float("inf")))
                dot_prod = batch_dot(
                    tf.expand_dims(query_img, 1),
                    tf.expand_dims(support_image, 2)
                )
                dot_prod = tf.squeeze(dot_prod)
                cos_sim = tf.multiply(dot_prod, support_magnitude)
                cos_sim = tf.reshape(cos_sim, [1, -1])
                similarities = tf.tensor_scatter_nd_update(similarities,
                                                           [[i_sample]],
                                                           cos_sim)
                i_sample += 1
            return tf.transpose(similarities)

        self.batch = x_support.shape[0]
        self.support_samples = x_support.shape[1]
        self.query_samples = x_query.shape[1]

        # Get one-hot representation
        y_support = tf.cast(y_support, tf.int32)
        y_support_one_hot = tf.one_hot(y_support, self.way, axis=-1)
        y_support_one_hot = tf.cast(y_support_one_hot, tf.float32)

        y_query = tf.cast(y_query, tf.int32)
        y_query_one_hot = tf.one_hot(y_query, self.way, axis=-1)
        y_query_one_hot = tf.cast(y_query_one_hot, tf.float32)

        # Embeddings for support images
        emb_imgs = []
        for i in range(self.support_samples):
            emb_imgs.append(self.g(x_support[:, i, :, :, :]))

        # Embeddings for query images
        for i_query in range(self.query_samples):
            query_emb = self.g(x_query[:, i_query, :, :, :])
            emb_imgs.append(query_emb)
            outputs = tf.stack(emb_imgs)

            # Fully contextual embedding
            outputs = self.fce(outputs)

            # Cosine similarity between support set and query
            similarities = _calc_cosine_distances(outputs[:-1], outputs[-1])

            # Produce predictions for target probabilities
            similarities = tf.nn.softmax(similarities)
            similarities = tf.expand_dims(similarities, 1)
            preds = tf.squeeze(batch_dot(similarities, y_support_one_hot))

            query_labels = y_query_one_hot[:, i_query, :]
            eq = tf.cast(tf.equal(
                tf.cast(tf.argmax(preds, axis=-1), tf.int32),
                tf.cast(y_query[:, i_query], tf.int32)), tf.float32)
            if i_query == 0:
                ce = categorical_crossentropy(query_labels, preds)
                acc = tf.reduce_mean(eq)
            else:
                ce += categorical_crossentropy(query_labels, preds)
                acc += tf.reduce_mean(eq)

            emb_imgs.pop()

        return ce / self.query_samples, acc / self.query_samples

    def save(self, dir_path):
        """
        Save model to the provided directory.
        Args:
            dir_path (str): path to the directory to save the model files.
        Returns: None
        """
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        # Save CNN encoder
        self.g.save(os.path.join(dir_path, 'cnn_encoder.h5'))
        # Save LSTM
        self.fce.save(os.path.join(dir_path, 'lstm.h5'))

    def load(self, dir_path):
        """
        Load model from provided directory.
        Args:
            dir_path (str): path to the directory from where restore model.
        Returns: None
        """
        # Encoder CNN
        encoder_path = os.path.join(dir_path, 'cnn_encoder.h5')
        self.g(tf.zeros([1, self.w, self.h, self.c]))
        self.g.load_weights(encoder_path)

        # LSTM
        lstm_path = os.path.join(dir_path, 'lstm.h5')
        self.fce(tf.zeros([1, self.batch_size, self.fce_dim]))
        self.fce.load_weights(lstm_path)
'''


def euclidean_distance(a, b):
    # a.shape = N x D
    # b.shape = M x D
    N, D = tf.shape(a)[0], tf.shape(a)[1]
    M = tf.shape(b)[0]
    a = tf.tile(tf.expand_dims(a, axis=1), (1, M, 1))
    b = tf.tile(tf.expand_dims(b, axis=0), (N, 1, 1))
    return tf.reduce_mean(tf.square(a - b), axis=2)






