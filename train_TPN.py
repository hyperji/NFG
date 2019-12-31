# -*- coding: utf-8 -*-
# @Time    : 19-7-2 下午4:50
# @Author  : HeJi
# @FileName: meta_learning_train.py
# @E-mail: hj@jimhe.cn

from meta_learning import TPN_stop_grad, Prototypical_Nets, RelationNets
from egnn_fsl import EGNN_FSL
from meta_learning_data import MiniImageNet_Generator, CUB_Generator, MiniImageNet_Generator_as_egnn
import numpy as np
import pickle
import tensorflow as tf
import argparse
from sklearn.preprocessing import LabelEncoder
import os
import glob
import gc
from utils import save_statistics
import shutil

print("12 30,  22:05, dropout 0.1 custom1 learning rate, add delete module add l2 norm only 3 layer, original norm")


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, init_lr, num_train_steps, warmup_steps=4000,
                 end_learning_rate=0, power=1, ):
        super(CustomSchedule, self).__init__()
        self.init_lr = init_lr

        self.num_train_steps = num_train_steps

        self.warmup_steps = warmup_steps

        self.end_learning_rate = end_learning_rate

        self.power = power

        self.the_val = (self.init_lr - self.end_learning_rate) * (1 - warmup_steps / self.num_train_steps) ** (
            self.power) + self.end_learning_rate

    def __call__(self, global_step):
        learning_rate = tf.constant(value=self.init_lr, shape=[], dtype=tf.float32)

        # Implements linear decay of the learning rate.
        # print("global_step", global_step)
        # print(learning_rate, self.end_learning_rate, self.num_train_steps, self.power)
        # global_step = np.min(global_step, self.num_train_steps)
        learning_rate = (learning_rate - self.end_learning_rate) * (1 - global_step / self.num_train_steps) ** (
            self.power) + self.end_learning_rate

        # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
        # learning rate will be `global_step/num_warmup_steps * init_lr`.
        if self.warmup_steps:
            global_steps_int = tf.cast(global_step, tf.int32)
            warmup_steps_int = tf.constant(self.warmup_steps, dtype=tf.int32)

            global_steps_float = tf.cast(global_steps_int, tf.float32)
            warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

            warmup_percent_done = global_steps_float / warmup_steps_float
            # print("warmup_percent_done", warmup_percent_done)
            # warmup_learning_rate = self.init_lr * warmup_percent_done
            warmup_learning_rate = self.the_val * warmup_percent_done

            is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
            learning_rate = (
                    (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

        return learning_rate


class CustomSchedule_v2(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=3000):
        super(CustomSchedule_v2, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def get_args():
    parser = argparse.ArgumentParser()
    # 添加参数
    parser.add_argument('--n_epochs', type=int, default=600, help='n_epochs')
    parser.add_argument("--log_every_n_samples", type=int, default=100)
    parser.add_argument("--log_every_n_epochs", type=int, default=10)
    parser.add_argument("--ckpt", type=str, default="models")
    parser.add_argument("--start_learning_rate", type=float, default=1.5 * 1e-3)
    parser.add_argument("--dataset", type=str, default="mini-ImageNet") # or "CUB"
    parser.add_argument("--n_way_train", type=int, default=5)
    parser.add_argument("--n_shot_train", type=int, default=1)
    parser.add_argument("--n_query_train", type=int, default=15)
    parser.add_argument("--data_augment", type = bool, default=False)
    parser.add_argument("--n_train_episodes", type=int, default=100)
    parser.add_argument("--n_test_episodes", type=int, default=600)
    parser.add_argument("--restore", type=bool, default=False)
    parser.add_argument("--rn", type=int, default=300)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.99)
    parser.add_argument("--encoder_type", type=str, default="NFG")
    parser.add_argument('--relation_type', type=str, default="NFG")
    parser.add_argument('--method', type=str, default="TPN")
    parser.add_argument("--power", type=int, default=2)
    parser.add_argument("--end_learning_rate", type=float, default=1e-6)
    parser.add_argument("--use_val_data", type=bool, default=True)

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

loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

@tf.function
def dual_train_step(model, FeatEncOpt, RelModOpt, S, Q):
    with tf.GradientTape() as tape:
        FeatEncLoss, RelModLoss, acc = model(S, Q, training = True)
    #FeatEncGrads = tape.gradient(FeatEncLoss, model.encoder.trainable_variables)
    #RelModGrads = tape.gradient(RelModLoss, model.relation.trainable_variables)
    gradients = tape.gradient(RelModLoss, model.trainable_variables)

    RelModOpt.apply_gradients(zip(gradients, model.trainable_variables))

    #FeatEncOpt.apply_gradients(zip(FeatEncGrads, model.encoder.trainable_variables))
    #RelModOpt.apply_gradients(zip(RelModGrads, model.relation.trainable_variables))
    train_loss(RelModLoss)
    # return acc
    train_accuracy(acc)

@tf.function
def train_step(model, optimizer, S, Q):
    # tar_inp = tar[:, :-1]
    # tar_real = tar[:, 1:]
    with tf.GradientTape() as tape:
        _, loss, acc = model(S, Q)
        #loss = loss_object(y_true=y_true, y_pred=logits)
    #print(acc)
    gradients = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    #return acc
    train_accuracy(acc)

n_way_val = 5
n_shot_val = 5
n_query_val = 15

n_way_test = 5
n_shot_test = 5
n_query_test = 15

n_examples = 350
im_width, im_height, channels = 84, 84, 3

h_dim = 64
z_dim = 64


if __name__ == "__main__":
    arg = get_args()
    print("arg", arg)

    log_every_n_samples = arg.log_every_n_samples
    log_every_n_epochs = arg.log_every_n_epochs

    n_epochs = arg.n_epochs
    n_episodes = arg.n_train_episodes
    n_way_train = arg.n_way_train
    n_shot_train = arg.n_shot_train
    n_query_train = arg.n_query_train

    #y_true = tf.reshape(tf.tile(tf.expand_dims(tf.range(n_way_train), 1), [1, n_query_train]), [-1])
    #y_true = tf.one_hot(y_true, depth=n_way_train)


    if arg.dataset == "mini-ImageNet":

        train_in = open("./data/mini-imagenet/mini-imagenet-cache-train.pkl","rb")
        test_in = open("./data/mini-imagenet/mini-imagenet-cache-test.pkl", "rb")
        val_in = open("./data/mini-imagenet/mini-imagenet-cache-val.pkl", 'rb')

        train = pickle.load(train_in)
        test = pickle.load(test_in)
        val = pickle.load(val_in)
        X_train = train["image_data"]
        X_train = X_train.reshape([64, 600, 84, 84, 3])

        X_val = val["image_data"]
        X_val = X_val.reshape([16, 600, 84, 84, 3])

        X_test = test["image_data"]
        X_test = X_test.reshape([20, 600, 84, 84, 3])

        #X_train = np.load("./data/mini-imagenet/mini-imagenet-train.npy")
        #X_val = np.load("./data/mini-imagenet/mini-imagenet-val.npy")
        #X_test = np.load("./data/mini-imagenet/mini-imagenet-test.npy")

        train_generator = MiniImageNet_Generator(
            X_train, n_way=n_way_train, n_shot=n_shot_train, n_query=n_query_train, aug=True)

        if arg.use_val_data:
            print("using validation data for validation")
            val_generator = MiniImageNet_Generator(X_val, n_way=n_way_train, n_shot=n_shot_train, n_query=n_query_val, aug=False)
        else:
            print("Not using validation data")
            val_generator = MiniImageNet_Generator(
                X_test, n_way=n_way_test, n_shot=n_shot_train, n_query=n_query_val, aug=False)

        test_generator = MiniImageNet_Generator(
                X_test, n_way=n_way_test, n_shot=n_shot_train, n_query=n_query_test, aug=False)

        if n_shot_train == 1:
            another_shot = 5
            test_generator1 = MiniImageNet_Generator(
                X_test, n_way=n_way_test, n_shot=another_shot, n_query=n_query_test, aug=False)
        elif n_shot_train == 5:
            another_shot = 1
            test_generator1 = MiniImageNet_Generator(
                X_test, n_way=n_way_test, n_shot=another_shot, n_query=n_query_test, aug=False)
        else:
            raise NotImplementedError

    elif arg.dataset == "CUB":
        data_in = open("./data/CUB_200_2011/CUB_200_2011.pkl", "rb")
        data_in = pickle.load(data_in)
        X = data_in['X']
        #X = X.numpy()
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

        val_masks = np.zeros_like(y)
        for l in val_labels:
            val_masks += y == l
        val_masks = val_masks > 0
        X_val = X[val_masks]
        y_val = y[val_masks]

        lbl1 = LabelEncoder()
        y_train = lbl1.fit_transform(y_train)
        lbl2 = LabelEncoder()
        y_test = lbl2.fit_transform(y_test)
        lbl3 = LabelEncoder()
        y_val = lbl3.fit_transform(y_val)
        train_generator = CUB_Generator(
            X_train, y_train, n_way=n_way_train, n_shot=n_shot_train, n_query=n_query_train, aug=True)

        if arg.use_val_data:
            print("using validation data for validation")
            val_generator = CUB_Generator(
                X_val, y_val, n_way=n_way_train, n_shot=n_shot_train, n_query=n_query_val, aug=False)
        else:
            print("Not using validation data")
            val_generator = CUB_Generator(
                X_test, y_test, n_way=n_way_test, n_shot=n_shot_train, n_query=n_query_val, aug=False)

        test_generator = CUB_Generator(
            X_test, y_test, n_way=n_way_test, n_shot=n_shot_train, n_query=n_query_test, aug=False)


        if n_shot_train == 1:
            another_shot = 5
            test_generator1 = CUB_Generator(
                X_test, y_test, n_way=n_way_test, n_shot=another_shot, n_query=n_query_test, aug=False)
        elif n_shot_train == 5:
            another_shot = 1
            test_generator1 = CUB_Generator(
                X_test, y_test, n_way=n_way_test, n_shot=another_shot, n_query=n_query_test, aug=False)
        else:
            raise NotImplementedError

    print("X_train", X_train.shape)
    print("X_test", X_test.shape)
    print("X_val", X_val.shape)

    print("n_way_train", n_way_train)
    print("n_shot_train", n_shot_train)
    print("n_query_train", n_query_train)

    if arg.method == "TPN":
        model = TPN_stop_grad(
        h_dim, z_dim, rn=arg.rn, k=arg.k, alpha=arg.alpha,
        encoder_type=arg.encoder_type, relation_type=arg.relation_type)
    elif arg.method == "ProtoNet":
        model = Prototypical_Nets(hidden_dim=h_dim, final_dim=z_dim, encoder_type=arg.encoder_type)
    elif arg.method == "RelNet":
        model = RelationNets(h_dim, z_dim, encoder_type = arg.encoder_type, relation_type = arg.relation_type)
    elif arg.method == "EGNN":
        model = EGNN_FSL(emb_size=128, in_features=None, node_features=96, edge_features=96, num_layers=3, dropout=0.1)
    else:
        raise NotImplementedError

    latest_version = 0

    if arg.restore:
        compat_names = arg.dataset + "_" + arg.method + "_" + arg.encoder_type + "_" + str(arg.n_way_train) + "_" + str(arg.n_shot_train) + "_"
        compat_dir_path = os.path.join(arg.ckpt, compat_names)
        all_compatiable_models = glob.glob(compat_dir_path+'*')
        if len(all_compatiable_models) > 0:
            latest_version = get_the_latest_model_version(all_compatiable_models)
            restore_name = compat_names+str(latest_version)
            restore_path = os.path.join(arg.ckpt, restore_name, restore_name)
            print("restore_path", restore_path)
            model.load_weights(restore_path)

    num_train_steps=n_epochs*n_episodes
    warmup_steps = min(num_train_steps // 10, 5000)
    print("n_train_steps", num_train_steps)
    print("warmup_steps", warmup_steps)

    learning_rate = CustomSchedule(
        init_lr=arg.start_learning_rate, num_train_steps=n_epochs*n_episodes, warmup_steps=warmup_steps,
        end_learning_rate=arg.end_learning_rate, power=arg.power)

    #learning_rate = CustomSchedule_v2(d_model=512, warmup_steps=warmup_steps)

    #learning_rate = 1e-3

    FeatEnc_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    RelMod_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    best_val_acc = 0
    val_accs = {}

    experiment_name = arg.dataset + "_" + arg.method + "_" + arg.encoder_type + "_" + str(arg.n_way_train) + "_" + str(
                    arg.n_shot_train)

    logs = "{}-way {}-shot learning problems".format(n_way_train, n_shot_train)
    save_statistics(experiment_name, ["Experimental details: {}".format(logs)])
    save_statistics(experiment_name, ["epoch", "train_acc", "val_acc", "test_acc"])
    if arg.method == 'EGNN':
        model.preparing(n_way = n_way_train, n_shot=n_shot_train, n_query=n_query_train)
    for eph in range(latest_version, n_epochs+latest_version):
        for episode in range(n_episodes):
            train_loss.reset_states()
            train_accuracy.reset_states()
            # inp -> portuguese, tar -> english
            data = train_generator[episode]
            #print("data[0].shape", data[0].shape)
            dual_train_step(model, FeatEnc_optimizer, RelMod_optimizer, data[0], data[1])
            #train_step(model, FeatEnc_optimizer, data[1], data[0])
            if (episode + 1) % log_every_n_samples == 0:
                # print(ls, ac)

                print('[epoch {}/{}, episode {}/{}] => loss: {:.5f}, acc: {:.5f}'.format(eph + 1, n_epochs,
                                                                                         episode + 1, n_episodes,
                                                                                         train_loss.result(),
                                                                                           train_accuracy.result()))
        accs = []
        if (eph + 1) % log_every_n_epochs == 0:
            if arg.method == 'EGNN':
                model.preparing(n_way=n_way_train, n_shot=n_shot_train, n_query=n_query_val)
            gc.collect()
            for episode in range(200):
                test_data = val_generator[episode]
                _, _, acc = model(test_data[0], test_data[1], training = False)
                #print("test acc", acc)
                accs.append(acc)
            mean_val_acc = np.mean(accs)
            print("mean acc", mean_val_acc)
            if mean_val_acc > best_val_acc:
                line_to_add = [eph+1, train_accuracy.result().numpy(), mean_val_acc, "None"]
                save_statistics(experiment_name, line_to_add)
                if arg.use_val_data:

                    compat_names = arg.dataset + "_" + arg.method + "_" + arg.encoder_type + "_" + str(
                        arg.n_way_train) + "_" + str(
                        arg.n_shot_train) + "_"
                    compat_dir_path = os.path.join(arg.ckpt, compat_names)
                    all_compatiable_models = glob.glob(compat_dir_path + '*')
                    if len(all_compatiable_models) > 0:
                        for delete_path in all_compatiable_models:
                            print("delete_path", delete_path)
                            shutil.rmtree(delete_path)

                    print("surpass the best_val_acc, saving model...")
                    save_name = arg.dataset + "_" + arg.method + "_" + arg.encoder_type + "_" + str(arg.n_way_train) + "_" + str(
                        arg.n_shot_train) + "_" + str(eph+1)
                    save_dir_path = os.path.join(arg.ckpt, save_name)

                    if not os.path.exists(save_dir_path):
                        os.makedirs(save_dir_path)

                    model.save_weights(os.path.join(save_dir_path, save_name))
                else:
                    print("Not using validation data, skipping saving...")
                best_val_acc = mean_val_acc
            if arg.method == 'EGNN':
                model.preparing(n_way=n_way_train, n_shot=n_shot_train, n_query=n_query_train)



    accs = []
    if arg.method == 'EGNN':
        model.preparing(n_way = n_way_test, n_shot=n_shot_train, n_query=n_query_test)
    for episode in range(arg.n_test_episodes):

        test_data = test_generator[episode]
        _, _, acc = model(test_data[0], test_data[1], training = False)
        #print("test acc", acc)
        accs.append(acc)
    print("final mean acc shot %d"%(n_shot_train), np.mean(accs))
    line_to_add = ["{}-shot acc".format(n_shot_train), "None", "None", np.mean(accs)]
    save_statistics(experiment_name, line_to_add)

    accs = []
    if arg.method == 'EGNN':
        model.preparing(n_way = n_way_test, n_shot=another_shot, n_query=n_query_test)
    for episode in range(arg.n_test_episodes):
        test_data = test_generator1[episode]
        _, _, acc = model(test_data[0], test_data[1], training = False)
        # print("test acc", acc)
        accs.append(acc)
    print("final mean acc shot %d"%(another_shot), np.mean(accs))

    line_to_add = ["{}-shot acc".format(another_shot), "None", "None", np.mean(accs)]
    save_statistics(experiment_name, line_to_add)

    if arg.use_val_data:
        print("restoring best valided model")
        compat_names = arg.dataset + "_" + arg.method + "_" + arg.encoder_type + "_" + str(arg.n_way_train) + "_" + str(
                        arg.n_shot_train) + "_"
        compat_dir_path = os.path.join(arg.ckpt, compat_names)
        all_compatiable_models = glob.glob(compat_dir_path + '*')
        if len(all_compatiable_models) > 0:
            latest_version = get_the_latest_model_version(all_compatiable_models)
            restore_name = compat_names + str(latest_version)
            restore_path = os.path.join(arg.ckpt, restore_name, restore_name)
            print("restore_path", restore_path)
            model.load_weights(restore_path)

        accs = []
        if arg.method == 'EGNN':
            model.preparing(n_way=n_way_test, n_shot=n_shot_train, n_query=n_query_test)
        for episode in range(arg.n_test_episodes):
            test_data = test_generator[episode]
            _, _, acc = model(test_data[0], test_data[1], training = False)
            # print("test acc", acc)
            accs.append(acc)
        print("the best valided model mean acc shot %d" % (n_shot_train), np.mean(accs))

        line_to_add = ["best valided {}-shot acc".format(n_shot_train), "None", "None", np.mean(accs)]
        save_statistics(experiment_name, line_to_add)

        accs = []
        if arg.method == 'EGNN':
            model.preparing(n_way=n_way_test, n_shot=another_shot, n_query=n_query_test)
        for episode in range(arg.n_test_episodes):
            test_data = test_generator1[episode]
            _, _, acc = model(test_data[0], test_data[1], training = False)
            # print("test acc", acc)
            accs.append(acc)
        print("the best valided model mean acc shot %d" % (another_shot), np.mean(accs))

        line_to_add = ["best valided {}-shot acc".format(another_shot), "None", "None", np.mean(accs)]
        save_statistics(experiment_name, line_to_add)



    print(model.summary())