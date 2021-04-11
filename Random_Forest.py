#-- coding: utf-8 --
#@Time : 2021/4/10 19:53
#@Author : HUANG XUYANG
#@Email : xhuang032@e.ntu.edu.sg
#@File : Random_Forest.py
#@Software: PyCharm

import sklearn.datasets as sk_dataset
import numpy as np
from Machine_Learning.Cart_Decision_Tree_on_Continuous_Value_by_Recursion import CartDecisionTree
import multiprocessing as mp
import os


class RandomForest:
    def __init__(self, n_tree, min_leaf, n_try, split_method='gini', pruning_prop=0.2):
        self.__n_tree = n_tree
        self.__min_leaf = min_leaf
        self.__n_class = None
        self.tree = []
        self.__split_method = split_method
        self.__pruning_prop = pruning_prop
        self.__n_try = n_try

    def train(self, data, label, n_class, multi_processing=True):
        self.__n_class = n_class
        if multi_processing:
            n_cores = mp.cpu_count() - 1
            p = mp.pool.Pool(n_cores)
            results = [p.apply_async(self.train_thread, args=(data, label, self.__n_tree // n_cores)) for _ in range(n_cores)]
            results = [p.get() for p in results]

            if self.__n_tree % n_cores != 0:
                results.append(self.train_thread(data, label, self.__n_tree % n_cores))
            [self.tree.extend(item) for item in results]

        else:
            for i in range(self.__n_tree):
                train, val = self.bagging(data, label)

                cart_dt = CartDecisionTree(self.__min_leaf, self.__split_method, self.__pruning_prop, self.__n_try)
                cart_dt.train(train[0], train[1], self.__n_class, val[0], val[1])
                self.tree.append(cart_dt)

    def train_thread(self, data, label, n_tree):
        print('Start pid', os.getpid())
        sub_tree = []
        for i in range(n_tree):
            train, val = self.bagging(data, label)
            cart_dt = CartDecisionTree(self.__min_leaf, self.__split_method, self.__pruning_prop, self.__n_try)
            cart_dt.train(train[0], train[1], self.__n_class, val[0], val[1])
            sub_tree.append(cart_dt)
        print('Finish pid', os.getpid())
        return sub_tree

    def predict(self, data):
        raw_result = [[] for _ in range(self.__n_class)]
        result_bin = []
        result = np.zeros(len(data))

        for i in range(self.__n_tree):
            _result = self.tree[i].predic(data)
            for j in range(self.__n_class):
                raw_result[j].extend(_result[j])

        for i in range(self.__n_class):
            result_bin.append(np.bincount(raw_result[i], minlength=self.__n_class))

        for i in range(len(data)):
            result[i] = np.argmax([result_bin[j][i] for j in range(self.__n_class)])

        return result

    def eval(self, data, label):
        raw_result = [[] for _ in range(self.__n_class)]
        result_bin = []
        result = np.zeros(len(data))

        for i in range(self.__n_tree):
            _result, _, _ = self.tree[i].eval(data, label)
            for j in range(self.__n_class):
                raw_result[j].extend(_result[j])

        for i in range(self.__n_class):
            result_bin.append(np.bincount(raw_result[i], minlength=len(data)))

        for i in range(len(data)):
            result[i] = np.argmax([result_bin[j][i] for j in range(self.__n_class)])

        acc = np.mean(result == label)

        return result, acc

    def bagging(self, data, label):
        train_rand_index = np.arange(len(data))
        np.random.shuffle(train_rand_index)
        val_rand_index = train_rand_index[int(0.65 * len(data)):]
        train_rand_index = train_rand_index[:int(0.65 * len(data))]
        train_rand_index = np.random.choice(train_rand_index, len(data))
        return (data[train_rand_index], label[train_rand_index]), (data[val_rand_index], label[val_rand_index])


def prepare_data(proportion):
    dataset = sk_dataset.load_iris()
    label = dataset['target']
    data = dataset['data']
    n_class = len(dataset['target_names'])

    shuffle_index = np.arange(len(label))
    np.random.shuffle(shuffle_index)

    train_number = int(proportion * len(label))
    train_index = shuffle_index[:train_number]
    val_index = shuffle_index[train_number:]
    data_train = data[train_index]
    label_train = label[train_index]
    data_val = data[val_index]
    label_val = label[val_index]
    return (data_train, label_train), (data_val, label_val), n_class


if __name__ == '__main__':
    minimum_leaf = 1
    num_tree = 500

    train, val, num_class = prepare_data(0.8)
    num_try = int(np.sqrt(train[0].shape[1]))
    rf = RandomForest(n_tree=num_tree, min_leaf=minimum_leaf, n_try=num_try, split_method='gini', pruning_prop=0.2)
    rf.train(train[0], train[1], num_class, multi_processing=True)
    _, train_acc = rf.eval(train[0], train[1])
    pred, val_acc = rf.eval(val[0], val[1])
    print('train_acc', train_acc)
    print('val_acc', val_acc)
