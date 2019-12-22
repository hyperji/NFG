import tensorflow as tf
from collections import OrderedDict

class EmbeddingImagenet_tf(tf.keras.layers.Layer):
    def __init__(self, emb_size):
        super(EmbeddingImagenet_tf, self).__init__()
        # set size
        self.hidden = 64
        self.last_hidden = self.hidden * 25
        self.emb_size = emb_size

        self.conv_1 = tf.keras.Sequential([tf.keras.layers.Conv2D(
            filters=self.hidden,
            kernel_size=3,
            padding="SAME",
            use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(2),
            tf.keras.layers.LeakyReLU()])


        self.conv_2 = tf.keras.Sequential([tf.keras.layers.Conv2D(
            filters=int(self.hidden*1.5),
            kernel_size=3,
            padding="SAME",
            use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(2),
            tf.keras.layers.LeakyReLU()])

        self.conv_3 = tf.keras.Sequential([tf.keras.layers.Conv2D(
            filters=int(self.hidden * 2),
            kernel_size=3,
            padding="SAME",
            use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(2),
            tf.keras.layers.LeakyReLU()])

        self.conv_4 = tf.keras.Sequential([tf.keras.layers.Conv2D(
            filters=int(self.hidden * 4),
            kernel_size=3,
            padding="SAME",
            use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(2),
            tf.keras.layers.LeakyReLU()])

        self.dropout1 = tf.keras.layers.Dropout(0.4)
        self.dropout2 = tf.keras.layers.Dropout(0.5)

        self.layer_last = tf.keras.Sequential([tf.keras.layers.Dense(self.emb_size, use_bias=True),
            tf.keras.layers.BatchNormalization()])

    def call(self, inputs, training = True):
        output_data = self.conv_1(inputs)

        output_data = self.conv_2(output_data)

        output_data = self.conv_3(output_data)
        if training:
            output_data = self.dropout1(output_data)

        output_data = self.conv_4(output_data)
        if training:
            output_data = self.dropout2(output_data)

        output_data = tf.reshape(output_data, [output_data.shape[0], -1])
        return self.layer_last(output_data)


class NodeUpdateNetwork_tf(tf.keras.layers.Layer):
    def __init__(self,
                 in_features,
                 num_features,
                 ratio=[2, 1],
                 dropout=0.0):
        super(NodeUpdateNetwork_tf, self).__init__()
        self.in_features = in_features
        self.num_features_list = [num_features * r for r in ratio]
        self.dropout = dropout

        # layers
        layer_list = OrderedDict()
        for l in range(len(self.num_features_list)):

            layer_list['conv{}'.format(l)] = tf.keras.layers.Conv2D(
                filters=self.num_features_list[l],
                kernel_size=1,
                use_bias=False)
            layer_list['norm{}'.format(l)] = tf.keras.layers.BatchNormalization()
            layer_list['relu{}'.format(l)] = tf.keras.layers.LeakyReLU()

            if self.dropout > 0 and l == (len(self.num_features_list) - 1):
                layer_list['drop{}'.format(l)] = tf.keras.layers.Dropout(self.dropout)

        self.network = tf.keras.Sequential(list(layer_list.values()))

    def call(self, node_feat, edge_feat, **kwargs):
        # get size
        num_tasks = node_feat.shape[0]
        num_data = node_feat.shape[1]

        # get eye matrix (batch_size x 2 x node_size x node_size)

        diag_mask = 1.0 - tf.tile(tf.expand_dims(tf.expand_dims(tf.eye(num_data), axis=0), axis=0), [num_tasks, 2, 1, 1])


        #print("diag_mask", diag_mask.shape)
        #print("edge_feat", edge_feat.shape)

        edge_feat = edge_feat * diag_mask

        # set diagonal as zero and normalize
        norms = tf.linalg.norm(edge_feat, ord=1, axis=-1, keepdims=True)
        edge_feat = tf.math.divide_no_nan(edge_feat,norms)
        #print(edge_feat)
        #return edge_feat
        #print(norms)
        #print("edge_feat after norm", edge_feat.shape)


        # compute attention and aggregate
        aggr_feat = tf.matmul(tf.squeeze(tf.concat(tf.split(edge_feat, [1, 1], axis=1), axis=2), axis=1), node_feat)

        #print("aggr_feat after matmul", aggr_feat.shape)

        aggr_feat = tf.concat(tf.split(aggr_feat, [num_data, num_data], axis=1), axis=-1)
        #print("aggr_feat", aggr_feat.shape)
        #print("num_data", num_data)

        #node_feat = torch.cat([node_feat, torch.cat(aggr_feat.split(num_data, 1), -1)], -1).transpose(1, 2)

        #print("tf.concat([node_feat, aggr_feat], axis=-1)", tf.concat([node_feat, aggr_feat], axis=-1).shape)

        node_feat = tf.concat([node_feat, aggr_feat], axis=-1)

        # non-linear transform
        node_feat = tf.expand_dims(node_feat, axis=2)
        #print("node_feat", node_feat.shape)

        node_feat = tf.squeeze(self.network(node_feat), axis=2)
        return node_feat



class EdgeUpdateNetwork_tf(tf.keras.layers.Layer):
    def __init__(self,
                 in_features,
                 num_features,
                 ratio=[2, 2, 1, 1],
                 separate_dissimilarity=False,
                 dropout=0.0):
        super(EdgeUpdateNetwork_tf, self).__init__()
        # set size
        self.in_features = in_features
        self.num_features_list = [num_features * r for r in ratio]
        self.separate_dissimilarity = separate_dissimilarity
        self.dropout = dropout

        # layers
        layer_list = OrderedDict()
        for l in range(len(self.num_features_list)):
            # set layer
            layer_list['conv{}'.format(l)] = tf.keras.layers.Conv2D(filters=self.num_features_list[l],
                                                                    kernel_size=1,
                                                                    use_bias=False)
            layer_list['norm{}'.format(l)] = tf.keras.layers.BatchNormalization()
            layer_list['relu{}'.format(l)] = tf.keras.layers.LeakyReLU()

            if self.dropout > 0:
                layer_list['drop{}'.format(l)] = tf.keras.layers.Dropout(self.dropout)

        layer_list['conv_out'] = tf.keras.layers.Conv2D(
            filters=1, kernel_size=1)
        self.sim_network = tf.keras.Sequential(list(layer_list.values()))

        if self.separate_dissimilarity:
            # layers
            layer_list = OrderedDict()
            for l in range(len(self.num_features_list)):
                # set layer
                layer_list['conv{}'.format(l)] = tf.keras.layers.Conv2D(
                    filters=self.num_features_list[l],
                    kernel_size=1,
                    use_bias=False)
                layer_list['norm{}'.format(l)] = tf.keras.layers.BatchNormalization()
                layer_list['relu{}'.format(l)] = tf.keras.layers.LeakyReLU()

                if self.dropout > 0:
                    layer_list['drop{}'.format(l)] = tf.keras.layers.Dropout(self.dropout)

            layer_list['conv_out'] = tf.keras.layers.Conv2D(filters=1, kernel_size=1)
            self.dsim_network = tf.keras.Sequential(list(layer_list.values()))

    def call(self, node_feat, edge_feat):
        # compute abs(x_i, x_j)
        x_i = tf.expand_dims(node_feat, axis=2)
        x_j = tf.transpose(x_i, [0, 2, 1, 3])
        x_ij = tf.math.abs(x_i - x_j)

        # compute similarity/dissimilarity (batch_size x feat_size x num_samples x num_samples)
        sim_val = tf.sigmoid(self.sim_network(x_ij))

        if self.separate_dissimilarity:
            dsim_val = tf.sigmoid(self.dsim_network(x_ij))
        else:
            dsim_val = 1.0 - sim_val

        sims = tf.concat([sim_val, dsim_val], axis=-1)
        sims = tf.transpose(sims, [0, 3, 1, 2])

        diag_mask = 1.0 - tf.tile(tf.expand_dims(tf.expand_dims(tf.eye(node_feat.shape[1]), axis=0), axis=0),
                                  [node_feat.shape[0], 2, 1, 1])

        edge_feat = edge_feat * diag_mask
        merge_sum = tf.reduce_sum(edge_feat, axis=-1, keepdims=True)

        edge_feat = edge_feat * sims
        # set diagonal as zero and normalize

        norms = tf.linalg.norm(edge_feat, ord=1, axis=-1, keepdims=True)
        edge_feat = tf.math.divide_no_nan(edge_feat, norms)

        edge_feat = edge_feat * merge_sum

        force_edge_feat = tf.tile(tf.expand_dims(tf.concat((tf.expand_dims(tf.eye(node_feat.shape[1]), axis=0),
                                                            tf.expand_dims(
                                                                tf.zeros([node_feat.shape[1], node_feat.shape[1]],
                                                                         dtype='float32'),
                                                                axis=0)), 0), 0), [node_feat.shape[0], 1, 1, 1])
        '''
        force_edge_feat = torch.eye(node_feat.size(1)).unsqueeze(0)

        dd = torch.zeros(node_feat.size(1), node_feat.size(1)).unsqueeze(0)

        force_edge_feat = torch.cat((force_edge_feat, dd), 0)

        force_edge_feat = force_edge_feat.unsqueeze(0)

        force_edge_feat = force_edge_feat.repeat(node_feat.size[0], 1, 1, 1)



        force_edge_feat = torch.cat(
            (torch.eye(node_feat.size(1)).unsqueeze(0), torch.zeros(node_feat.size(1), node_feat.size(1)).unsqueeze(0)),
            0).unsqueeze(0).repeat(node_feat.size(0), 1, 1, 1).to(tt.arg.device)
        '''

        edge_feat = edge_feat + force_edge_feat
        edge_feat = edge_feat + 1e-6
        edge_feat = edge_feat / tf.tile(tf.expand_dims(tf.reduce_sum(edge_feat, axis=1), axis=1), [1, 2, 1, 1])

        return edge_feat



class GraphNetwork_tf(tf.keras.layers.Layer):
    def __init__(self,
                 in_features,
                 node_features,
                 edge_features,
                 num_layers,
                 dropout=0.0):
        super(GraphNetwork_tf, self).__init__()
        # set size
        self.in_features = in_features
        self.node_features = node_features
        self.edge_features = edge_features
        self.num_layers = num_layers
        self.dropout = dropout
        self._modules = {}

        # for each layer
        for l in range(self.num_layers):
            # set edge to node
            edge2node_net = NodeUpdateNetwork_tf(in_features=self.in_features if l == 0 else self.node_features,
                                              num_features=self.node_features,
                                              dropout=self.dropout if l < self.num_layers-1 else 0.0)

            # set node to edge
            node2edge_net = EdgeUpdateNetwork_tf(in_features=self.node_features,
                                              num_features=self.edge_features,
                                              separate_dissimilarity=False,
                                              dropout=self.dropout if l < self.num_layers-1 else 0.0)

            self._modules['edge2node_net{}'.format(l)] = edge2node_net
            self._modules['node2edge_net{}'.format(l)] = node2edge_net


    # forward
    def call(self, node_feat, edge_feat):
        # for each layer
        edge_feat_list = []
        for l in range(self.num_layers):
            # (1) edge to node
            node_feat = self._modules['edge2node_net{}'.format(l)](node_feat, edge_feat)

            # (2) node to edge
            edge_feat = self._modules['node2edge_net{}'.format(l)](node_feat, edge_feat)

            # save edge feature
            edge_feat_list.append(edge_feat)

        # if tt.arg.visualization:
        #     for l in range(self.num_layers):
        #         ax = sns.heatmap(tt.nvar(edge_feat_list[l][0, 0, :, :]), xticklabels=False, yticklabels=False, linewidth=0.1,  cmap="coolwarm",  cbar=False, square=True)
        #         ax.get_figure().savefig('./visualization/edge_feat_layer{}.png'.format(l))


        return edge_feat_list



def label2edge_tf(labels):
    num_samples = labels.shape[1]

    label_i = tf.tile(tf.expand_dims(labels, axis=-1), [1, 1, num_samples])
    label_j = tf.transpose(label_i, [0, 2, 1])

    edge = tf.cast(tf.math.equal(label_i, label_j), tf.float32)

    edge = tf.expand_dims(edge, axis=1)
    edge = tf.concat([edge, 1 - edge], axis=1)
    return edge


def preparing_labels_edges_and_masks(n_way, n_shot, n_query):
    results = {}
    num_supports = n_way * n_shot
    num_queries = n_way * n_query
    num_samples = num_supports + num_queries

    support_edge_mask = tf.zeros([1, num_samples, num_samples])
    support_edge_mask = support_edge_mask.numpy()
    support_edge_mask[:, :num_supports, :num_supports] = 1
    query_edge_mask = 1 - support_edge_mask

    s_y = tf.tile(tf.reshape(
        tf.range(0, n_way), [n_way, 1]), [1, n_shot])
    s_y = tf.reshape(s_y, [1, num_supports])

    q_y = tf.tile(tf.reshape(
        tf.range(0, n_way), [n_way, 1]), [1, n_query])
    q_y = tf.reshape(q_y, [1, num_queries])

    full_label = tf.concat([s_y, q_y], axis=1)

    full_edge = label2edge_tf(full_label)

    # set init edge
    init_edge = tf.identity(full_edge)  # batch_size x 2 x num_samples x num_samples
    init_edge = init_edge.numpy()
    init_edge[:, :, num_supports:, :] = 0.5
    init_edge[:, :, :, num_supports:] = 0.5
    for i in range(num_queries):
        init_edge[:, 0, num_supports + i, num_supports + i] = 1.0
        init_edge[:, 1, num_supports + i, num_supports + i] = 0.0
    results['ys'] = s_y
    results['yq'] = q_y
    results['full_edge'] = full_edge
    results['init_edge'] = tf.convert_to_tensor(init_edge)
    results['support_edge_mask'] = tf.convert_to_tensor(support_edge_mask)
    results['query_edge_mask'] = tf.convert_to_tensor(query_edge_mask)
    return results


class EGNN_FSL(tf.keras.Model):
    def __init__(self, emb_size, in_features,
                 node_features,
                 edge_features,
                 num_layers,
                 dropout=0.0):
        super(EGNN_FSL, self).__init__()
        self.enc_module = EmbeddingImagenet_tf(emb_size)
        self.gnn_module = GraphNetwork_tf(in_features,
                 node_features,
                 edge_features,
                 num_layers,
                 dropout=dropout)

        self.edge_loss = tf.keras.losses.BinaryCrossentropy()

        self.node_loss = tf.keras.losses.CategoricalCrossentropy()
        self.num_layers = num_layers


    def preparing(self, n_way, n_shot, n_query):
        results = preparing_labels_edges_and_masks(n_way, n_shot, n_query)
        self.ys = results['ys']
        self.yq = results['yq']
        self.full_edge = results['full_edge']
        self.init_edge = results['init_edge']
        self.support_edge_mask = results['support_edge_mask']
        self.query_edge_mask = results['query_edge_mask']
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.ys_one_hot = tf.one_hot(self.ys, depth=n_way, dtype=tf.float32)


    def hit(self, logit, label):
        pred = tf.argmax(logit, axis=1)
        pred = tf.cast(pred, tf.float32)
        hit = tf.cast(tf.equal(pred, label), tf.float32)
        return hit


    def call(self, s, q, training=None, mask=None):
        s_shape = s.shape
        q_shape = q.shape
        n_way = s_shape[0]
        n_shot = s_shape[1]
        n_query = q_shape[1]
        assert self.n_way == n_way
        assert self.n_shot == n_shot
        assert self.n_query == n_query
        # s with shape [n_way, n_shot, W, H, C]
        # q with shape [n_way, n_query, W, H, C]


        # print("y", y)
        # print("y.shape", y.shape)
        #y_one_hot = tf.one_hot(y, depth=n_way, dtype=tf.float32)

        # print("s.shape", s.shape)

        s = tf.reshape(s, [-1, s_shape[2], s_shape[3], s_shape[4]])
        q = tf.reshape(q, [-1, q_shape[2], q_shape[3], q_shape[4]])
        x = tf.concat([s, q], axis=0)


        x = self.enc_module(x)
        x = tf.expand_dims(x, axis=0)
        #print("x after enc_module", x.shape)
        #print("self.init_edge", self.init_edge.shape)

        full_logit_layers = self.gnn_module(x, self.init_edge)
        #full_logit = full_logit_layers[-1]


        #TODO figure it out
        full_edge_loss_layers = [self.edge_loss(y_pred= 1 - full_logit_layer[:, 0], y_true=1.0 - self.full_edge[:, 0]) for
                                 full_logit_layer in full_logit_layers]
        #print("full_edge_loss_layers[0]", full_edge_loss_layers[0])
        #print("self.query_edge_mask", self.query_edge_mask.shape)
        #print("self.full_edge[:, 0]", self.full_edge[:, 0].shape)

        # weighted edge loss for balancing pos/neg
        pos_query_edge_loss_layers = [
            tf.reduce_sum(full_edge_loss_layer * self.query_edge_mask * self.full_edge[:, 0]) / tf.reduce_sum(
                self.query_edge_mask * self.full_edge[:, 0]) for full_edge_loss_layer in full_edge_loss_layers]
        neg_query_edge_loss_layers = [
            tf.reduce_sum(full_edge_loss_layer * self.query_edge_mask * (1 - self.full_edge[:, 0])) / tf.reduce_sum(
                self.query_edge_mask * (1 - self.full_edge[:, 0])) for full_edge_loss_layer in
            full_edge_loss_layers]

        query_edge_loss_layers = [pos_query_edge_loss_layer + neg_query_edge_loss_layer for
                                  (pos_query_edge_loss_layer, neg_query_edge_loss_layer) in
                                  zip(pos_query_edge_loss_layers, neg_query_edge_loss_layers)]

        # compute accuracy
        full_edge_accr_layers = [self.hit(full_logit_layer, 1.0 - self.full_edge[:, 0]) for full_logit_layer in
                                 full_logit_layers]
        query_edge_accr_layers = [tf.reduce_sum(full_edge_accr_layer * self.query_edge_mask) / tf.reduce_sum(
            self.query_edge_mask) for full_edge_accr_layer in full_edge_accr_layers]

        # compute node loss & accuracy (num_tasks x num_quries x num_ways)
        query_node_pred_layers = [tf.matmul(full_logit_layer[:, 0, n_way*n_shot:, :n_way*n_shot], self.ys_one_hot) for
                                  full_logit_layer in
                                  full_logit_layers]  # (num_tasks x num_quries x num_supports) * (num_tasks x num_supports x num_ways)
        query_node_accr_layers = [tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(query_node_pred_layer, axis=-1), tf.int32), self.yq), tf.float32))
                                  for query_node_pred_layer in query_node_pred_layers]

        total_loss_layers = query_edge_loss_layers

        # update model
        total_loss = []
        for l in range(self.num_layers - 1):
            total_loss += [tf.reshape(total_loss_layers[l], [-1]) * 0.5]
        total_loss += [tf.reshape(total_loss_layers[-1], [-1]) * 1.0]
        total_loss = tf.reduce_mean(tf.concat(total_loss, axis=0))

        return total_loss,  query_edge_accr_layers[-1], query_node_accr_layers[-1]