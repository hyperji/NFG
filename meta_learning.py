# -*- coding: utf-8 -*-
# @Time    : 19-7-1 下午10:10
# @Author  : HeJi
# @FileName: meta_learning.py
# @E-mail: hj@jimhe.cn

import tensorflow as tf
from NFG_lite import NFG4img, Aggregator
import numpy as np
print("meta learning.py, 11:20 16:00")

class conv_block_v2(tf.keras.layers.Layer):
    def __init__(self, out_channels, conv_padding = "SAME", pooling_padding = "VALID"):

        self.conv = tf.keras.layers.Conv2D(filters=out_channels,
                                           kernel_size=3, strides=1,
                                           padding=conv_padding,
                                           use_bias=True)
        self.bn = tf.keras.layers.BatchNormalization()
        self.max_pool = tf.keras.layers.MaxPool2D(2, padding=pooling_padding)

        super(conv_block_v2, self).__init__()
    def call(self, x):
        x_ = self.conv(x)
        x_ = self.bn(x_)
        x_ = tf.nn.relu(x_)
        x_ = self.max_pool(x_)
        return x_

class ConvBlockNoMaxPool(tf.keras.layers.Layer):
    def __init__(self, out_channels, padding="SAME"):
        self.conv = tf.keras.layers.Conv2D(filters=out_channels,
                                           kernel_size=3, strides=1,
                                           padding=padding,
                                           use_bias=True)
        self.bn = tf.keras.layers.BatchNormalization()
        super(ConvBlockNoMaxPool, self).__init__()

    def call(self, x):
        x_ = self.conv(x)
        x_ = self.bn(x_)
        x_ = tf.nn.relu(x_)
        return x_

class CNNEncoder4RelNet(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, final_dim):
        super(CNNEncoder4RelNet, self).__init__()
        self.conv1 = conv_block_v2(hidden_dim, conv_padding="VALID")
        self.conv2 = conv_block_v2(hidden_dim, conv_padding="VALID")
        self.conv3 = ConvBlockNoMaxPool(hidden_dim, padding="SAME")
        self.conv4 = ConvBlockNoMaxPool(final_dim, padding="SAME")

    def call(self, x):
        x_ = self.conv1(x)
        x_ = self.conv2(x_)
        x_ = self.conv3(x_)
        x_ = self.conv4(x_)
        return x_

class CNNEncoder(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, final_dim):
        super(CNNEncoder, self).__init__()
        self.conv1 = conv_block_v2(hidden_dim)
        self.conv2 = conv_block_v2(hidden_dim)
        self.conv3 = conv_block_v2(hidden_dim)
        self.conv4 = conv_block_v2(final_dim)
        #self.flatten = tf.keras.layers.Flatten()

    def call(self, x):
        x_ = self.conv1(x)
        x_ = self.conv2(x_)
        x_ = self.conv3(x_)
        x_ = self.conv4(x_)
        #x_ = self.flatten(x_)
        return x_


class NFGEncoder(tf.keras.layers.Layer):
    def __init__(self, ksize, strides, d_neuron, dk, dv, Nh, dact, final_dim, padding='SAME'):
        super(NFGEncoder, self).__init__()
        self.nfg = NFG4img(ksize, strides, d_neuron, dk, dv, Nh, dact, final_dim, padding)
        self.ln = tf.keras.layers.LayerNormalization()
    def call(self, x):
        x = self.nfg(x)
        x = self.ln(x)
        return x


class NFGEncoder4RelNet(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, final_dim):
        super(NFGEncoder4RelNet, self).__init__()
        self.agg3 = Aggregator(3, 1, padding="SAME")
        self.agg2 = Aggregator(2, 1, padding="SAME")
        self.nfg1 = NFGEncoder(ksize=3, strides=2, d_neuron=hidden_dim, dk=8, dv=hidden_dim, Nh=8, dact=8, final_dim=final_dim, padding="VALID")
        self.nfg2 = NFGEncoder(ksize=3, strides=2, d_neuron=hidden_dim, dk=8, dv=hidden_dim, Nh=8, dact=8, final_dim=final_dim, padding="VALID")
        self.nfg3 = NFGEncoder(ksize=3, strides=1, d_neuron=hidden_dim, dk=8, dv=hidden_dim, Nh=8, dact=8, final_dim=final_dim, padding="SAME")
        self.nfg4 = NFGEncoder(ksize=3, strides=1, d_neuron=hidden_dim, dk=8, dv=hidden_dim, Nh=8, dact=8, final_dim=final_dim, padding="SAME")

    def call(self, x):
        x_ = self.agg3(x)
        x_ = self.nfg1(x_)

        x_ = self.agg2(x_)
        x_ = self.nfg2(x_)

        x_ = self.agg2(x_)
        x_ = self.nfg3(x_)

        x_ = self.agg2(x_)
        x_ = self.nfg4(x_)
        return x_


class NFGRelationModule(tf.keras.Model):
    """docstring for RelationNetwork"""
    def __init__(self):
        super(NFGRelationModule, self).__init__()
        self.agg2 = Aggregator(2, 1, padding="SAME")
        self.nfg1 = NFGEncoder(ksize=3, strides=2, d_neuron=64, dk=8, dv=64, Nh=8, dact=8, final_dim=64, padding="VALID")
        self.nfg2 = NFGEncoder(ksize=3, strides=2, d_neuron=64, dk=8, dv=64, Nh=8, dact=8, final_dim=64, padding="VALID")
        self.fc1 = tf.keras.layers.Dense(8, activation = 'relu', use_bias = True)
        self.fc2 = tf.keras.layers.Dense(1, activation = 'sigmoid', use_bias = True)

    def call(self, x):

        out = self.agg2(x)
        out = self.nfg1(out)

        #print("out after layer1", out.shape)
        out = self.agg2(out)
        out = self.nfg2(out)
        #print("out after layer2", out.shape)
        out = tf.reshape(out, [out.shape[0],-1])
        #print("out after reshape", out.shape)
        out = self.fc1(out)
        #print("out after fc1", out.shape)
        out = self.fc2(out)
        #print("out after fc2", out.shape)
        return out


class NFGFeatEnc4RelNet(tf.keras.Model):
    def __init__(self, hidden_dim, final_dim):
        super(NFGFeatEnc4RelNet, self).__init__()
        self.encoder = NFGEncoder4RelNet(hidden_dim, final_dim)
        self.flatten = tf.keras.layers.Flatten()

    def call(self, s, q):
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

        ###################################################
        z_ = self.flatten(z)

        z_dim = z_.shape[-1]
        z_proto_ = tf.reduce_mean(tf.reshape(
            z_[:n_way * n_shot], [n_way, n_shot, z_dim]), axis=1)
        zq_ = z_[n_way * n_shot:]

        # print("z_proto.shape", z_proto.shape)
        # print("zq.shape", zq.shape)

        dists = euclidean_distance(zq_, z_proto_)
        # print("dists.shape", dists.shape)

        _log_dists = tf.nn.log_softmax(-dists, axis=1)
        # print("_log_dists.shape", _log_dists.shape)
        log_p_y = tf.reshape(_log_dists, [n_way, n_query, -1])

        #preds = tf.cast(tf.argmax(log_p_y, axis=-1), tf.int32)
        # print("preds", preds)

        #masks = tf.cast(tf.equal(preds, y), tf.float32)

        # print("masks", masks)

        ce_loss = -tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.multiply(y_one_hot, log_p_y), axis=-1), [-1]))
        #acc = tf.reduce_mean(masks)

        ##################################################



        z_shape = z.shape

        z_proto = tf.reduce_mean(tf.reshape(
            z[:n_way * n_shot], [n_way, n_shot] + z_shape[1:]), axis=1)

        zq = tf.reshape(z[n_way * n_shot:], [n_way * n_query] + z_shape[1:])

        z_proto_ext = tf.tile(tf.expand_dims(z_proto, axis=0), [n_way * n_query, 1, 1, 1, 1])
        zq_ext = tf.tile(tf.expand_dims(zq, axis=0), [n_way, 1, 1, 1, 1])
        zq_ext = tf.transpose(zq_ext, [1, 0, 2, 3, 4])

        relation_pairs = tf.concat([z_proto_ext, zq_ext], axis=-1)

        relation_pairs = tf.reshape(relation_pairs, [n_way * n_query * n_way] + z_shape[1:-1] + [z_shape[-1] * 2])

        return ce_loss, relation_pairs


class RelationModule(tf.keras.Model):
    """docstring for RelationNetwork"""
    def __init__(self):
        super(RelationModule, self).__init__()
        self.layer1 = conv_block_v2(out_channels=64, conv_padding="VALID")
        self.layer2 = conv_block_v2(out_channels=64, conv_padding="VALID")
        self.fc1 = tf.keras.layers.Dense(8, activation = 'relu', use_bias = True)
        self.fc2 = tf.keras.layers.Dense(1, activation = 'sigmoid', use_bias = True)

    def call(self, x):
        out = self.layer1(x)
        #print("out after layer1", out.shape)
        out = self.layer2(out)
        #print("out after layer2", out.shape)
        out = tf.reshape(out, [out.shape[0],-1])
        #print("out after reshape", out.shape)
        out = self.fc1(out)
        #print("out after fc1", out.shape)
        out = self.fc2(out)
        #print("out after fc2", out.shape)
        return out




class FeatEnc4RelNet(tf.keras.Model):
    def __init__(self, hidden_dim, final_dim):
        super(FeatEnc4RelNet, self).__init__()
        self.encoder = CNNEncoder4RelNet(hidden_dim, final_dim)
        self.flatten = tf.keras.layers.Flatten()

    def call(self, s, q):
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

        ###################################################
        z_ = self.flatten(z)

        z_dim = z_.shape[-1]
        z_proto_ = tf.reduce_mean(tf.reshape(
            z_[:n_way * n_shot], [n_way, n_shot, z_dim]), axis=1)
        zq_ = z_[n_way * n_shot:]

        # print("z_proto.shape", z_proto.shape)
        # print("zq.shape", zq.shape)

        dists = euclidean_distance(zq_, z_proto_)
        # print("dists.shape", dists.shape)

        _log_dists = tf.nn.log_softmax(-dists, axis=1)
        # print("_log_dists.shape", _log_dists.shape)
        log_p_y = tf.reshape(_log_dists, [n_way, n_query, -1])

        #preds = tf.cast(tf.argmax(log_p_y, axis=-1), tf.int32)
        # print("preds", preds)

        #masks = tf.cast(tf.equal(preds, y), tf.float32)

        # print("masks", masks)

        ce_loss = -tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.multiply(y_one_hot, log_p_y), axis=-1), [-1]))
        #acc = tf.reduce_mean(masks)

        ##################################################



        z_shape = z.shape

        z_proto = tf.reduce_mean(tf.reshape(
            z[:n_way * n_shot], [n_way, n_shot] + z_shape[1:]), axis=1)

        zq = tf.reshape(z[n_way * n_shot:], [n_way * n_query] + z_shape[1:])

        z_proto_ext = tf.tile(tf.expand_dims(z_proto, axis=0), [n_way * n_query, 1, 1, 1, 1])
        zq_ext = tf.tile(tf.expand_dims(zq, axis=0), [n_way, 1, 1, 1, 1])
        zq_ext = tf.transpose(zq_ext, [1, 0, 2, 3, 4])

        relation_pairs = tf.concat([z_proto_ext, zq_ext], axis=-1)

        relation_pairs = tf.reshape(relation_pairs, [n_way * n_query * n_way] + z_shape[1:-1] + [z_shape[-1] * 2])

        return ce_loss, relation_pairs


class CNNEncoder4TPN(tf.keras.layers.Layer):
    def __init__(self, h_dim, z_dim):
        super(CNNEncoder4TPN, self).__init__()
        self.conv1 = conv_block_v2(h_dim)
        self.conv2 = conv_block_v2(h_dim)
        self.conv3 = conv_block_v2(h_dim)
        self.conv4 = conv_block_v2(z_dim)
        self.flatten = tf.keras.layers.Flatten()

    def call(self, x):
        x_ = self.conv1(x)
        x_ = self.conv2(x_)
        x_ = self.conv3(x_)
        x_ = self.conv4(x_)
        x_ = self.flatten(x_)
        return x_

class RelationModule4TPN(tf.keras.layers.Layer):
    def __init__(self, h_dim):
        super(RelationModule4TPN, self).__init__()
        self.conv1 = conv_block_v2(h_dim, pooling_padding="SAME")
        self.conv2 = conv_block_v2(1, pooling_padding="SAME")
        self.fc1 = tf.keras.layers.Dense(8, activation=tf.nn.relu, use_bias=True)
        self.fc2 = tf.keras.layers.Dense(1, activation=tf.sigmoid, use_bias=True)
        self.flatten = tf.keras.layers.Flatten()

    def call(self, x):
        net = tf.reshape(x, [-1, 5, 5, 64])
        net = self.conv1(net)
        net = self.conv2(net)

        net = self.flatten(net)

        net = self.fc1(net)
        net = self.fc2(net)

        net = self.flatten(net)
        return net


class CNN_TPN(tf.keras.Model):
    def __init__(self, h_dim, z_dim, rn, k, alpha):
        super(CNN_TPN, self).__init__()
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.rn = rn
        self.k = k
        self.alpha = alpha
        self.encoder = CNNEncoder4TPN(h_dim, z_dim)
        self.relation = RelationModule4TPN(h_dim)
        if self.rn==300:       # learned sigma, fixed alpha
            self.alpha = tf.constant(self.alpha)
        else:                          # learned sigma and alpha
            self.alpha = tf.Variable(self.alpha, name='alpha')


    def call(self, s, q):
        s_shape = s.shape
        q_shape = q.shape
        num_classes, num_support = s_shape[0], s_shape[1]
        num_queries = q_shape[1]
        #ys = tf.tile(tf.reshape(
        #    tf.range(0, num_classes), [num_classes, 1]), [1, num_support])
        ys = np.tile(np.arange(num_classes)[:, np.newaxis], (1, num_support)).astype(np.uint8)

        ys_one_hot = tf.one_hot(ys, depth=num_classes)

        y = tf.tile(tf.reshape(
            tf.range(0, num_classes), [num_classes, 1]), [1, num_queries])

        y_one_hot = tf.one_hot(y, depth=num_classes, dtype=tf.float32)

        s = tf.reshape(s, [-1, s_shape[2], s_shape[3], s_shape[4]])
        q = tf.reshape(q, [-1, q_shape[2], q_shape[3], q_shape[4]])

        emb_s = self.encoder(s)
        emb_q = self.encoder(q)
        #TODO this loss is not used in original paper, but in order to reach the acc the original paper, we had to use this

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

        return proto_loss, ce_loss, acc

    def label_prop(self, x, u, ys):

        epsilon = np.finfo(float).eps
        # x: NxD, u: UxD
        s = tf.shape(ys)
        ys = tf.reshape(ys, (s[0] * s[1], -1))
        Ns, C = tf.shape(ys)[0], tf.shape(ys)[1]
        Nu = tf.shape(u)[0]

        yu = tf.zeros((Nu, C)) / tf.cast(C, tf.float32) + epsilon  # 0 initialization
        # yu = tf.ones((Nu,C))/tf.cast(C,tf.float32)            # 1/C initialization
        y = tf.concat([ys, yu], axis=0)
        gt = tf.reshape(tf.tile(tf.expand_dims(tf.range(C), 1), [1, tf.cast(Nu / C, tf.int32)]), [-1])

        all_un = tf.concat([x, u], axis=0)
        all_un = tf.reshape(all_un, [-1, 1600])
        N, d = tf.shape(all_un)[0], tf.shape(all_un)[1]

        # compute graph weights
        if self.rn in [30, 300]:  # compute example-wise sigma
            self.sigma = self.relation(all_un)

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

        #topk_W = tf.compat.v1.sparse_to_dense(full_indices, tf.shape(W), tf.reshape(values, [-1]), default_value=0.,
        #                            validate_indices=False)
        ind1 = (topk_W > 0) | (tf.transpose(topk_W) > 0)  # union, k-nearest neighbor
        ind2 = (topk_W > 0) & (tf.transpose(topk_W) > 0)  # intersection, mutal k-nearest neighbor
        ind1 = tf.cast(ind1, tf.float32)

        topk_W = ind1 * W

        return topk_W



class NFG_Prototypical_Nets(tf.keras.Model):
    def __init__(self, ksize, strides, d_neuron, dk, dv, Nh, dact, final_dim):
        super(NFG_Prototypical_Nets, self).__init__()
        self.agg3 = Aggregator(3, 1, padding="SAME")
        self.agg2 = Aggregator(2, 1, padding="SAME")
        self.nfg1 = NFGEncoder(ksize, strides, d_neuron, dk, dv, Nh, dact, final_dim, padding="SAME")
        self.nfg2 = NFGEncoder(ksize, strides, d_neuron, dk, dv, Nh, dact, final_dim, padding="SAME")
        self.nfg3 = NFGEncoder(ksize, strides, d_neuron, dk, dv, Nh, dact, final_dim, padding="SAME")
        self.nfg4 = NFGEncoder(ksize, strides, d_neuron, dk, dv, Nh, dact, final_dim, padding="VALID")
        self.flatten = tf.keras.layers.Flatten()
        self.ln = tf.keras.layers.LayerNormalization()

    def call(self, s, q):
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
    def __init__(self, hidden_dim, final_dim):
        super(Prototypical_Nets, self).__init__()
        self.encoder = CNNEncoder(hidden_dim, final_dim)
        self.flatten = tf.keras.layers.Flatten()

    def call(self, s, q):
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



def euclidean_distance(a, b):
    # a.shape = N x D
    # b.shape = M x D
    N, D = tf.shape(a)[0], tf.shape(a)[1]
    M = tf.shape(b)[0]
    a = tf.tile(tf.expand_dims(a, axis=1), (1, M, 1))
    b = tf.tile(tf.expand_dims(b, axis=0), (N, 1, 1))
    return tf.reduce_mean(tf.square(a - b), axis=2)






