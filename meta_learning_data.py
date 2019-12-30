# -*- coding: utf-8 -*-
# @Time    : 19-7-1 下午7:26
# @Author  : HeJi
# @FileName: meta_learning_data.py
# @E-mail: hj@jimhe.cn

import numpy as np
import tensorflow as tf
# import torch
# from torchvision import transforms
# import torch.utils.data as data
# import random

print("meta_learning_data, augment with random flip, 12 30 22:30 using normalization original")

class MiniImageNet_Generator(object):

    def __init__(self, X, n_way=5, n_shot = 1, n_query=100, aug = False):
        self.X = X
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_classes = X.shape[0]
        self.n_examples_per_class = X.shape[1]
        self.base_shape = list(self.X.shape[2:])
        self.aug = aug

        self.mean = 113.77  # precomputed
        # self.std = np.std(list(self.x_train)+list(self.x_val))
        self.std = 70.1899  # precomputed

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.n_classes*self.n_examples_per_class

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        del index

        support, query = self.__data_generation()

        return support, query

    def preprocess_batch(self, x_batch):
        x_batch = (x_batch - self.mean) / self.std
        return x_batch

    def augment(self, x_batch):
        # print(x_batch.shape, axis, k)
        x_batch = tf.image.random_flip_left_right(x_batch)
        #x_batch = tf.image.random_flip_up_down(x_batch)
        return x_batch


    def __data_generation(self):

        use_labels = np.random.choice(
            self.n_classes, (self.n_way, 1, 1), replace=False)
        #use_labels = np.concatenate(use_labels, axis=0)
        use_indices = [np.random.choice(
            self.n_examples_per_class, (1, self.n_shot+self.n_query, 1),replace=False)
        for _ in range(self.n_way)]
        use_indices = np.concatenate(use_indices, axis=0)
        use_labels = np.tile(use_labels, [1, self.n_shot+self.n_query, 1])
        coords = np.concatenate([use_labels, use_indices], axis=-1)
        coords = np.reshape(coords, [-1, 2])

        data = self.X[coords[:, 0], coords[:, 1]]
        #if augment:
        #    data = self.rotate_batch(data)
        data = data.astype(np.float32)
        if self.aug:
            data = self.augment(data)
        #data = tf.convert_to_tensor(data, dtype=tf.float32)

        data = self.preprocess_batch(data)
        #data = data.astype(np.float32)
        data = tf.reshape(data, [self.n_way, self.n_shot+self.n_query] + self.base_shape)

        return tf.split(data, [self.n_shot, self.n_query], axis=1)


class MiniImageNet_Generator_as_egnn(object):

    def __init__(self, X, n_way=5, n_shot=1, n_query=100, aug=False):
        """
        the mini-imagenet data generator similiar with the egnn in cvpr2019
        :param X:
        :param n_way:
        :param n_shot:
        :param n_query:
        :param aug:
        """
        self.X = X
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_classes = X.shape[0]
        self.n_examples_per_class = X.shape[1]
        self.base_shape = list(self.X.shape[2:])
        self.aug = aug

        #self.mean = [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
        #self.std = [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]

        self.mean = [120.39586422, 115.59361427, 104.54012653]
        self.std = [70.68188272, 68.27635443, 72.54505529]

        self.mean = np.reshape(self.mean, [1, 1, 1, 3])
        self.std = np.reshape(self.std, [1, 1, 1, 3])


    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.n_classes * self.n_examples_per_class

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        del index

        support, query = self.__data_generation()

        return tf.cast(support, tf.float32), tf.cast(query, tf.float32)

    def preprocess_batch(self, x_batch):
        #x_batch = x_batch / 255.0
        x_batch = (x_batch - self.mean) / self.std
        return x_batch

    def augment(self, x_batch):
        '''
        x_batch = tf.image.pad_to_bounding_box(
            x_batch, offset_height=4, offset_width=4,
            target_height=92, target_width=92)
        x_batch = tf.image.random_crop(x_batch, size=[x_batch.shape[0], 84, 84, x_batch.shape[-1]])
        '''
        x_batch = tf.image.random_flip_left_right(x_batch)
        return x_batch

    def __data_generation(self):
        use_labels = np.random.choice(
            self.n_classes, (self.n_way, 1, 1), replace=False)
        # use_labels = np.concatenate(use_labels, axis=0)
        use_indices = [np.random.choice(
            self.n_examples_per_class, (1, self.n_shot + self.n_query, 1), replace=False)
            for _ in range(self.n_way)]
        use_indices = np.concatenate(use_indices, axis=0)
        use_labels = np.tile(use_labels, [1, self.n_shot + self.n_query, 1])
        coords = np.concatenate([use_labels, use_indices], axis=-1)
        coords = np.reshape(coords, [-1, 2])

        data = self.X[coords[:, 0], coords[:, 1]]
        data = data.astype(np.float32)
        if self.aug:
            data = self.augment(data)
        data = self.preprocess_batch(data)
        # data = data.astype(np.float32)
        data = tf.reshape(data, [self.n_way, self.n_shot + self.n_query] + self.base_shape)

        return tf.split(data, [self.n_shot, self.n_query], axis=1)

'''
class MiniImagenetLoader(data.Dataset):
    def __init__(self, root, partition='train'):
        super(MiniImagenetLoader, self).__init__()
        # set dataset information
        self.root = root
        self.partition = partition
        self.data_size = [3, 84, 84]

        # set normalizer
        mean_pix = [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
        std_pix = [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]
        # mean_pix = [120.39586422, 115.59361427, 104.54012653]
        # std_pix = [70.68188272, 68.27635443, 72.54505529]
        normalize = transforms.Normalize(mean=mean_pix, std=std_pix)

        # set transformer
        if self.partition == 'train':
            self.transform = transforms.Compose([transforms.RandomCrop(84, padding=4),
                                                 lambda x: np.asarray(x),
                                                 transforms.ToTensor(),
                                                 normalize])
        else:  # 'val' or 'test' ,
            self.transform = transforms.Compose([lambda x: np.asarray(x),
                                                 transforms.ToTensor(),
                                                 normalize])

        # load data
        self.data = self.load_dataset()

    def load_dataset(self):
        # load data
        dataset_path = "data/mini-imagenet/mini-imagenet-%s.npy" % self.partition
        return np.load(dataset_path)

    def get_task_batch(self,
                       num_tasks=5,
                       num_ways=20,
                       num_shots=1,
                       num_queries=1,
                       seed=None):

        if seed is not None:
            random.seed(seed)

        # init task batch data
        support_data, support_label, query_data, query_label = [], [], [], []
        for _ in range(num_ways * num_shots):
            data = np.zeros(shape=[num_tasks] + self.data_size,
                            dtype='float32')
            label = np.zeros(shape=[num_tasks],
                             dtype='float32')
            support_data.append(data)
            support_label.append(label)
        for _ in range(num_ways * num_queries):
            data = np.zeros(shape=[num_tasks] + self.data_size,
                            dtype='float32')
            label = np.zeros(shape=[num_tasks],
                             dtype='float32')
            query_data.append(data)
            query_label.append(label)

        # get full class list in dataset
        full_class_list = np.arange(self.data.shape[0])
        # print(full_class_list)

        # for each task
        for t_idx in range(num_tasks):
            # define task by sampling classes (num_ways)
            task_class_list = np.random.choice(full_class_list, num_ways, replace=False)

            # for each sampled class in task
            for c_idx in range(num_ways):
                # sample data for support and query (num_shots + num_queries)
                indices = np.random.choice(self.data.shape[1], (num_shots + num_queries,), replace=False)
                class_data_list = self.data[task_class_list[c_idx]][indices]
                # print(class_data_list.shape)

                # load sample for support set
                for i_idx in range(num_shots):
                    # set data
                    support_data[i_idx + c_idx * num_shots][t_idx] = self.transform(class_data_list[i_idx])
                    support_label[i_idx + c_idx * num_shots][t_idx] = c_idx

                # load sample for query set
                for i_idx in range(num_queries):
                    query_data[i_idx + c_idx * num_queries][t_idx] = self.transform(class_data_list[num_shots + i_idx])
                    query_label[i_idx + c_idx * num_queries][t_idx] = c_idx

        # convert to tensor (num_tasks x (num_ways * (num_supports + num_queries)) x ...)
        support_data = torch.stack([torch.from_numpy(data).float() for data in support_data], 1)
        support_label = torch.stack([torch.from_numpy(label).float() for label in support_label], 1)
        query_data = torch.stack([torch.from_numpy(data).float() for data in query_data], 1)
        query_label = torch.stack([torch.from_numpy(label).float() for label in query_label], 1)

        support_data = support_data.transpose(2, 4)
        query_data = query_data.transpose(2, 4)

        return [support_data.numpy(), support_label.numpy(), query_data.numpy(), query_label.numpy()]
'''
class CUB_Generator(object):

    def __init__(self, X, y, n_way=5, n_shot = 1, n_query=100, aug = True):
        self.X = X
        self.y = y
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_classes = X.shape[0]
        self.base_shape = list(self.X.shape[1:])

        self.all_labels = np.array(list(set(y.flatten())))
        self.n_classes = len(self.all_labels)

        self.label_indexs = {i: np.where(y == i)[0] for i in self.all_labels}
        self.n_examples_per_class = [len(self.label_indexs[i]) for i in range(self.n_classes)]
        self.cumsum = np.cumsum(self.n_examples_per_class)
        self.cumsum = np.delete(self.cumsum, -1)
        self.cumsum = np.insert(self.cumsum, 0, 0)
        self.aug = aug



    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.n_classes*self.n_examples_per_class

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        del index

        support, query = self.__data_generation()

        return support, query


    def augment(self, x_batch):
        # print(x_batch.shape, axis, k)
        x_batch = tf.image.random_flip_left_right(x_batch)
        return x_batch


    def __data_generation(self):

        use_labels = np.random.choice(
            self.n_classes, (self.n_way, ), replace=False)
        #use_labels = np.concatenate(use_labels, axis=0)
        use_indices = [self.cumsum[lbl] + np.random.choice(
            self.n_examples_per_class[lbl], (self.n_shot+self.n_query, ),replace=False)
        for lbl in use_labels]
        use_indices = np.concatenate(use_indices, axis=0)

        data = self.X[use_indices]

        data = data.astype(np.float32)
        #print("data.shape", data.shape)
        if self.aug:
            data = self.augment(data)

        data = tf.reshape(data, [self.n_way, self.n_shot+self.n_query] + self.base_shape)

        return tf.split(data, [self.n_shot, self.n_query], axis=1)


