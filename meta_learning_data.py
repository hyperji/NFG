# -*- coding: utf-8 -*-
# @Time    : 19-7-1 下午7:26
# @Author  : HeJi
# @FileName: meta_learning_data.py
# @E-mail: hj@jimhe.cn

import numpy as np
import tensorflow as tf
import os
from scipy.ndimage import rotate
#from tqdm import tqdm
import pandas as pd
import time
import matplotlib.pyplot as plt
#import imageio

class MiniImageNet_Generator(object):

    def __init__(self, X, n_way=5, n_shot = 1, n_query=100):
        self.X = X
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_classes = X.shape[0]
        self.n_examples_per_class = X.shape[1]
        self.base_shape = list(self.X.shape[2:])

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
        data = self.augment(data)
        #data = tf.convert_to_tensor(data, dtype=tf.float32)

        data = self.preprocess_batch(data)
        #data = data.astype(np.float32)
        data = tf.reshape(data, [self.n_way, self.n_shot+self.n_query] + self.base_shape)

        return tf.split(data, [self.n_shot, self.n_query], axis=1)



class CUB_Generator(object):

    def __init__(self, X, y, n_way=5, n_shot = 1, n_query=100):
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
        data = self.augment(data)

        data = tf.reshape(data, [self.n_way, self.n_shot+self.n_query] + self.base_shape)

        return tf.split(data, [self.n_shot, self.n_query], axis=1)


