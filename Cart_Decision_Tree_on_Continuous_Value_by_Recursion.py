#-- coding: utf-8 --
#@Time : 2021/4/9 14:20
#@Author : HUANG XUYANG
#@Email : xhuang032@e.ntu.edu.sg
#@File : Cart_Decision_Tree_on_Continuous_Value_by_Recursion.py
#@Software: PyCharm

import numpy as np
import sklearn.datasets as sk_dataset

class TreeNode:
    """A decision tree node

    Attributes:
        feature_index: An integer of feature index, specify the decision feature.
        thr: A floating number of threshold to split the data.
        left: Left node.
        right: Right node.
    """
    def __init__(self, feature_index, thr):
        self.feature_index = feature_index
        self.thr = thr
        self.left = None
        self.right = None
        self.left_class = None
        self.right_class = None
        self.left_acc = None
        self.right_acc = None
        self.acc = None

    def split(self, data, label=None):
        """Split the input data.

        :param data: A 2-D Numpy array.
        :param label: A 1-D Numpy array.
        :return: [left split data, right split data], [left split data ground truth, right split data ground truth]
        """
        left_mask = data[:, self.feature_index] <= self.thr
        right_mask = ~left_mask

        left_leaf_data = data[left_mask]
        right_leaf_data = data[right_mask]

        if label is not None:
            left_leaf_label = label[left_mask]
            right_leaf_label = label[right_mask]
            return [left_leaf_data, right_leaf_data], [left_leaf_label, right_leaf_label]
        else:
            return [left_leaf_data, right_leaf_data]

    def split_predict(self, data, index, label=None):
        """

        :param data: A 2-D numpy array.
        :return: [left split data, right split data],
                 [left split data index, right split data index]
                 [left split data ground truth, right split data ground truth]
        """
        left_mask = data[:, self.feature_index] <= self.thr
        right_mask = ~left_mask

        left_leaf_data = data[left_mask]
        right_leaf_data = data[right_mask]

        left_leaf_data_index = index[left_mask]
        right_leaf_data_index = index[right_mask]
        if label is not None:
            left_leaf_label = label[left_mask]
            right_leaf_label = label[right_mask]
            return [left_leaf_data, right_leaf_data], [left_leaf_data_index, right_leaf_data_index], [left_leaf_label, right_leaf_label]
        else:
            return [left_leaf_data, right_leaf_data], [left_leaf_data_index, right_leaf_data_index]


class CartDecisionTree:
    """
    Attributes:
        root: A decision tree node class.
        __min_leaf: An integer of minimum leaf number.
        __n_class: An integer of class number.
        split_method: String, choose 'gini' or 'entropy'.
        __pruning_prop: A floating number, proportion of data number to prune DT.
        __pruning: Bool, if True prune the DT, or not.

    """
    def __init__(self, min_leaf, split_method='gini', pruning_prop=None, n_try=None):
        self.root = None
        self.__min_leaf = min_leaf
        self.__n_class = None
        c = Criterion()
        self.__train_node = getattr(c, split_method)
        self.__pruning_prop = pruning_prop
        self.__pruning = pruning_prop is not None
        self.__n_try = n_try

    def train(self, data, label, n_class, pruning_data=None, pruning_label=None):
        """Train a decision tree.

        Using recursion to train a Cart DT.

        :param data: A 2-D Numpy array.
        :param label: A 1-D Numpy array.
        :param n_class: An integer of class number.
        :return: No return.
        """
        self.__n_class = n_class
        assert (pruning_label is None) == (pruning_data is None), 'Please provide data and label both or not!'

        if self.__pruning:
            if pruning_data is None:
                pruning_data = data[:int(len(data) * self.__pruning_prop)]
                pruning_label = label[:int(len(label) * self.__pruning_prop)]
                data = data[int(len(data) * self.__pruning_prop):]
                label = label[int(len(label) * self.__pruning_prop):]

        def grow(_data, _label):
            # Train single node.
            _feature_index, _thr = self.__train_node(_data, _label, self.__n_try)
            _node = TreeNode(_feature_index, _thr)
            _split_data, _split_label = _node.split(_data, _label)

            if (len(_split_label[0]) == 0) | (len(_split_label[1]) == 0):
                return None
            _node.left_class = np.argmax(np.bincount(_split_label[0]))
            if not ((len(_split_label[0]) < self.__min_leaf) | (_split_label[0] == _split_label[0][0]).all()):
                _node.left = grow(_split_data[0], _split_label[0])

            _node.right_class = np.argmax(np.bincount(_split_label[1]))
            if not ((len(_split_label[1]) < self.__min_leaf) | (_split_label[1] == _split_label[1][0]).all()):
                _node.right = grow(_split_data[1], _split_label[1])

            return _node

        self.root = grow(data, label)
        if self.__pruning:
            self.__post_pruning(pruning_data, pruning_label)

    def __post_pruning(self, data, label):
        """Prune DT.

        :param data: A 2-D Numpy array.
        :param label: A 1-D Numpy array.
        :return:
        """
        index = np.arange(len(data))

        def __inference(root, _data, _index, _label):
            _split_data, _split_data_index, _split_label = root.split_predict(_data, _index, _label)
            if len(_label) == 0:
                root.acc = 0
            else:
                root.acc = np.mean(_label == np.argmax(np.bincount(_label)))
            if root.left is None:
                if len(_split_data[0]) != 0:
                    tmp_acc = np.mean(_split_label[0] == root.left_class)
                else:
                    tmp_acc = 0
                root.left_acc = tmp_acc
            else:
                __inference(root.left, _split_data[0], _split_data_index[0], _split_label[0])

            if root.right is None:
                if len(_split_data[1]) != 0:
                    tmp_acc = np.mean(_split_label[1] == root.right_class)
                else:
                    tmp_acc = 0
                root.right_acc = tmp_acc
            else:
                __inference(root.right, _split_data[1], _split_data_index[1], _split_label[1])

            if (root.left_acc is not None) & (root.right_acc is not None):
                child_acc = np.mean([root.right_acc, root.left_acc])
                if root.acc > child_acc:
                    del root.left
                    del root.right
                    root.left, root.right, root.left_acc, root.right_acc = None, None, None, None

        __inference(self.root, data, index, label)

    def predict(self, data):
        """Traverse DT get a result.

        :param data: A 2-D Numpy array.
        :return: Prediction.
        """
        result = [[] for _ in range(self.__n_class)]
        index = np.arange(len(data))

        def __inference(root, _data, _index):
            # if root is None:
            #     return None
            _split_data, _split_data_index = root.split_predict(_data, _index)
            if root.left is None:
                if len(_split_data[0]) != 0:
                    result[root.left_class].extend(_split_data_index[0])
            else:
                __inference(root.left, _split_data[0], _split_data_index[0])

            if root.right is None:
                if len(_split_data[1]) != 0:
                    result[root.left_class].extend(_split_data_index[1])
            else:
                __inference(root.right, _split_data[1], _split_data_index[1])

            return None
        __inference(self.root, data, index)
        return result

    def eval(self, data, label):
        """

        :param data: A 2-D Numpy array.
        :param label: A 1-D Numpy array.
        :return: Prediction, Prediction with label, Accuracy.
        """
        result = [[] for _ in range(self.__n_class)]
        result_label = [[] for _ in range(self.__n_class)]
        index = np.arange(len(data))

        def __inference(root, _data, _index, _label):
            # if root is None:
            #     return None
            _split_data, _split_data_index, _split_label = root.split_predict(_data, _index, _label)
            if root.left is None:
                if len(_split_data[0]) != 0:
                    result[root.left_class].extend(_split_data_index[0])
                    result_label[root.left_class].extend(_split_label[0])
            else:
                __inference(root.left, _split_data[0], _split_data_index[0], _split_label[0])

            if root.right is None:
                if len(_split_data[1]) != 0:
                    result[root.right_class].extend(_split_data_index[1])
                    result_label[root.right_class].extend(_split_label[1])
            else:
                __inference(root.right, _split_data[1], _split_data_index[1], _split_label[1])

            return None
        __inference(self.root, data, index, label)
        acc = []
        for i in range(self.__n_class):
            if len(result_label[i]) == 0:
                acc.append(0)
            else:
                acc.append(np.mean(np.array(result_label[i]) == i))
        acc = np.mean(acc)
        return result, result_label, acc

    def print_tree(self):
        def __inference(root):
            if root.left is None:
                print('leaf')
                print('left class', root.left_class)
                print('right class', root.left_class)
                print('root acc', root.acc, '\n')
            else:
                __inference(root.left)

            if root.right is None:
                print('leaf')
                print('left class', root.left_class)
                print('right class', root.left_class)
                print('root acc', root.acc, '\n')
            else:
                __inference(root.right)

            print('node')
            print('left class', root.left_class)
            print('right class', root.left_class)
            print('root acc', root.acc, '\n')

            return None

        __inference(self.root)


class Criterion:
    def gini(self, data, label, n_try=None):
        """Traverse all features and value, find the best split feature and threshold.

        :param data: A 2-D Numpy array.
        :param label: A 1-D Numpy array.
        :return: Best feature to split, best threshold to split.
        """
        if n_try is not None:
            rand_feature_index = np.arange(data.shape[1])
            np.random.shuffle(rand_feature_index)
            rand_feature_index = rand_feature_index[:n_try]
            data = data[:, rand_feature_index]
        best_gini = np.inf
        for i in range(data.shape[1]):
            sort_index = np.argsort(data[:, i])
            sub_data = data[sort_index]
            sub_label = label[sort_index]
            for j in range(1, data.shape[0]):
                tmp_gini_value = j / len(data) * self.__gini(sub_label[:j]) + \
                                 (len(data) - j) / len(data) * self.__gini(sub_label[j:])
                if tmp_gini_value < best_gini:
                    best_gini = tmp_gini_value
                    best_thr = np.mean([sub_data[j-1, i], sub_data[j, i]])
                    best_feature = rand_feature_index[i]
        return best_feature, best_thr

    def entropy(self, data, label, n_try=None):
        """Traverse all features and value, find the best split feature and threshold.

        Find the gain higher than average, pick the highest gain ratio one.

        :param data: A 2-D Numpy array.
        :param label: A 1-D Numpy array.
        :return: Best feature to split, best threshold to split.
        """
        gain = []
        gain_ratio = []
        ent_before = self.__ent(label)
        if n_try is not None:
            rand_feature_index = np.arange(data.shape[1])
            np.random.shuffle(rand_feature_index)
            rand_feature_index = rand_feature_index[:n_try]
            data = data[:, rand_feature_index]
        for i in range(data.shape[1]):
            sort_index = np.argsort(data[:, i])
            sub_label = label[sort_index]
            for j in range(1, data.shape[0]):
                tmp_gain = ent_before - \
                           (j / len(data) * self.__ent(sub_label[:j]) + (len(data) - j) / len(data) * self.__ent(sub_label[j:]))
                tmp_gain_ratio = tmp_gain / (- j / len(data) * np.log2(j / len(data)) -
                                             (len(data) - j) / len(data) * np.log2((len(data) - j) / len(data)))
                gain.append(tmp_gain)
                gain_ratio.append(tmp_gain_ratio)
        gain = np.array(gain)
        gain_ratio = np.array(gain_ratio)
        gain_ratio[gain < np.mean(gain)] = -np.inf
        best_index = np.argmax(gain_ratio)
        mat_index = np.unravel_index(best_index, [data.shape[1], data.shape[0] - 1])
        best_feature = rand_feature_index[mat_index[0]]
        sub_data = np.sort(data[:, mat_index[0]])
        best_thr = np.mean([sub_data[mat_index[1]], sub_data[mat_index[1] + 1]])
        return best_feature, best_thr

    def __gini(self, label):
        _label_class = list(set(label))
        gini_value = 1 - np.sum([(np.sum(label == i) / len(label)) ** 2 for i in _label_class])
        return gini_value

    def __ent(self, label):
        _label_class = list(set(label))
        ent_value = - np.sum([np.sum(label == i) / len(label) * np.log2(np.sum(label == i) / len(label)) for i in _label_class])
        return ent_value


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
    train, val, num_class = prepare_data(0.8)
    num_try = int(np.sqrt(train[0].shape[1]))
    cart_dt = CartDecisionTree(minimum_leaf, 'gini', pruning_prop=0.3, n_try=num_try)
    cart_dt.train(train[0], train[1], num_class)
    _, _, train_acc = cart_dt.eval(train[0], train[1])
    pred, pred_gt, val_acc = cart_dt.eval(val[0], val[1])
    print('train_acc', train_acc)
    print('val_acc', val_acc)
    # cart_dt.print_tree()
