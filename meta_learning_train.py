# -*- coding: utf-8 -*-
# @Time    : 19-7-2 下午4:50
# @Author  : HeJi
# @FileName: meta_learning_train.py
# @E-mail: hj@jimhe.cn

from meta_learning import Prototypical_Nets, NFG_Prototypical_Nets
from meta_learning_data import MiniImageNet_Generator, CUB_Generator
import numpy as np
import pickle
import tensorflow as tf
import argparse
from sklearn.preprocessing import LabelEncoder
import os
import glob
print("11:20,  16:00")

def get_args():
    parser = argparse.ArgumentParser()
    # 添加参数
    parser.add_argument('--n_epochs', type=int, default=400, help='n_epochs')
    parser.add_argument("--log_every_n_samples", type=int, default=100)
    parser.add_argument("--log_every_n_epochs", type=int, default=50)
    parser.add_argument("--ckpt", type=str, default="models")
    parser.add_argument("--start_learning_rate", type=float, default=1e-4)
    parser.add_argument("--dataset", type=str, default="mini-ImageNet") # or "CUB"
    parser.add_argument("--n_way_train", type=int, default=5)
    parser.add_argument("--n_shot_train", type=int, default=1)
    parser.add_argument("--data_augment", type = bool, default=False)
    parser.add_argument("--method", type=str, default="CNN_TPN") #or choice "CNN"
    parser.add_argument("--n_train_episodes", type=int, default=100)
    parser.add_argument("--n_test_episodes", type=int, default=500)
    parser.add_argument("--restore", type=bool, default=False)

    # 如: python xx.py --foo hello  > hello
    args = parser.parse_args()
    return args

def get_model_version(model_name):
    return int(model_name.split('_')[-1])

def get_the_latest_model_version(model_names):
    if len(model_names) == 0:
        return None
    else:
        return max(list(map(get_model_version, model_names)))

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')



@tf.function
def train_step(model, optimizer, S, Q, epoch):
    # tar_inp = tar[:, :-1]
    # tar_real = tar[:, 1:]
    initial_lr = 1e-3
    lr = max(0.5**(epoch//200)*initial_lr, 1e-5)
    print("lr", lr)
    optimizer.learning_rate = lr

    with tf.GradientTape() as tape:
        #logits = model(X, training = True, enc_padding_mask=None)
        #logits = model(X, training=True, enc_padding_mask=None)
        #print("model", model)
        logits, loss, acc = model(S, Q)
        #loss = loss_function(y, logits)
    #print(acc)
    gradients = tape.gradient(loss, model.trainable_variables)
    #print("grad", gradients)
    #print('trainable variable', model.trainable_variables)
    #print("opt learning rate", optimizer.learning_rate)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    #return acc
    train_accuracy(acc)

n_way_eval = 5
n_shot_eval = 5
n_query_eval = 15

n_way_test = 5
n_shot_test = 5
n_query_test = 15

n_examples = 350
im_width, im_height, channels = 84, 84, 3

h_dim = 64
z_dim = 64


if __name__ == "__main__":
    arg = get_args()

    log_every_n_samples = arg.log_every_n_samples
    log_every_n_epochs = arg.log_every_n_epochs

    n_epochs = arg.n_epochs
    n_episodes = arg.n_train_episodes
    n_way_train = arg.n_way_train
    n_shot_train = arg.n_shot_train
    n_query_train = 15


    if arg.dataset == "mini-ImageNet":
        train_in = open("./data/mini-imagenet/mini-imagenet-cache-train.pkl","rb")
        test_in = open("./data/mini-imagenet/mini-imagenet-cache-test.pkl", "rb")

        train = pickle.load(train_in)
        test = pickle.load(test_in)
        X_train = train["image_data"]
        X_train = X_train.reshape([64, 600, 84, 84, 3])

        X_test = test["image_data"]
        X_test = X_test.reshape([20, 600, 84, 84, 3])
        train_generator = MiniImageNet_Generator(
            X_train, n_way=n_way_train, n_shot=n_shot_train, n_query=n_query_train)

        test_generator = MiniImageNet_Generator(
            X_test, n_way=n_way_test, n_shot=n_shot_train, n_query=n_query_test)

        if n_shot_train == 1:
            another_shot = 5
            test_generator1 = MiniImageNet_Generator(
                X_test, n_way=n_way_test, n_shot=another_shot, n_query=n_query_test)
        elif n_shot_train == 5:
            another_shot = 1
            test_generator1 = MiniImageNet_Generator(
                X_test, n_way=n_way_test, n_shot=another_shot, n_query=n_query_test)
        else:
            raise NotImplementedError

    elif arg.dataset == "CUB":
        data_in = open("./data/CUB_200_2011/CUB_200_2011.pkl", "rb")
        data_in = pickle.load(data_in)
        X = data_in['X']
        X = tf.image.per_image_standardization(X)
        X = X.numpy()
        y = data_in['y']
        del data_in
        permutated_labels = np.random.permutation(200)
        train_labels = permutated_labels[:100]
        test_labels = permutated_labels[100:150]
        val_labels = permutated_labels[150:]

        train_masks = np.zeros_like(y)
        for l in train_labels:
            train_masks += y==l
        train_masks = train_masks > 0
        X_train = X[train_masks]
        y_train = y[train_masks]

        test_masks = np.zeros_like(y)
        for l in test_labels:
            test_masks += y == l
        test_masks = test_masks > 0
        X_test = X[test_masks]
        y_test = y[test_masks]

        lbl1 = LabelEncoder()
        y_train = lbl1.fit_transform(y_train)
        lbl2 = LabelEncoder()
        y_test = lbl2.fit_transform(y_test)
        train_generator = CUB_Generator(
            X_train, y_train, n_way=n_way_train, n_shot=n_shot_train, n_query=n_query_train)

        test_generator = CUB_Generator(
            X_test, y_test, n_way=n_way_test, n_shot=n_shot_train, n_query=n_query_test)

        test_generator1 = CUB_Generator(
            X_test, y_test, n_way=n_way_test, n_shot=1, n_query=n_query_test)
    print("X_train", X_train.shape)
    print("X_test", X_test.shape)

    print("n_way_train", n_way_train)
    print("n_shot_train", n_shot_train)
    print("n_query_train", n_query_train)


    if arg.method == 'CNN_ProtoNet':
        model = Prototypical_Nets(hidden_dim=h_dim, final_dim=z_dim)
    elif arg.method == 'NFG_ProtoNet':
        model = NFG_Prototypical_Nets(ksize=3, strides=2, d_neuron=64, dk=8, dv=64, Nh=8, dact=8, final_dim=64)
    else:
        raise NotImplementedError
    latest_version = 0

    if arg.restore:
        compat_names = arg.dataset + "_" + arg.method + "_" + str(arg.n_way_train) + "_" + str(arg.n_shot_train) + "_"
        compat_dir_path = os.path.join(arg.ckpt, compat_names)
        all_compatiable_models = glob.glob(compat_dir_path+'*')
        if len(all_compatiable_models) > 0:
            latest_version = get_the_latest_model_version(all_compatiable_models)
            restore_name = compat_names+str(latest_version)
            restore_path = os.path.join(arg.ckpt, restore_name)
            print("restore_path", restore_path)
            model.load_weights(restore_path)


    optimizer = tf.keras.optimizers.Adam()

    for eph in range(latest_version, n_epochs+latest_version):
        for episode in range(n_episodes):
            train_loss.reset_states()
            train_accuracy.reset_states()
            # inp -> portuguese, tar -> english
            data = train_generator[episode]
            #print("data[0].shape", data[0].shape)
            train_step(model, optimizer, data[0], data[1], epoch=eph)
            if (episode + 1) % log_every_n_samples == 0:
                # print(ls, ac)

                print('[epoch {}/{}, episode {}/{}] => loss: {:.5f}, acc: {:.5f}'.format(eph + 1, n_epochs,
                                                                                         episode + 1, n_episodes,
                                                                                         train_loss.result(),
                                                                                           train_accuracy.result()))
        accs = []
        if (eph + 1) % log_every_n_epochs == 0:
            for episode in range(200):
                test_data = test_generator[episode]
                _, _, acc = model(test_data[0], test_data[1])
                #print("test acc", acc)
                accs.append(acc)
            print("mean acc", np.mean(accs))
            save_name = arg.dataset + "_" + arg.method + "_" + str(arg.n_way_train) + "_" + str(
                arg.n_shot_train) + "_" + str(eph)
            save_dir_path = os.path.join(arg.ckpt, save_name)

            if not os.path.exists(save_dir_path):
                os.makedirs(save_dir_path)

            model.save_weights(os.path.join(save_dir_path, save_name))


    accs = []
    for episode in range(arg.n_test_episodes):

        test_data = test_generator[episode]
        _, _, acc = model(test_data[0], test_data[1])
        #print("test acc", acc)
        accs.append(acc)
    print("final mean acc shot %d"%(n_shot_train), np.mean(accs))

    accs = []
    for episode in range(arg.n_test_episodes):
        test_data = test_generator1[episode]
        _, _, acc = model(test_data[0], test_data[1])
        # print("test acc", acc)
        accs.append(acc)
    print("final mean acc shot %d"%(another_shot), np.mean(accs))




