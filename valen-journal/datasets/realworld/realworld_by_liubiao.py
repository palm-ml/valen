import os
import os.path
import sys
import torch
import numpy as np
import pickle
# import h5py
import scipy
from scipy.io import loadmat
import torch.utils.data as data
from copy import deepcopy
from sklearn.model_selection import KFold
from torch.utils.data import Dataset


class RealwordDataLoader:
    def __init__(self, mat_path):
        # self.n_splits = n_splits
        self.data = loadmat(mat_path)
        print(self.data.keys())
        if "data" in self.data.keys():
            self.features, self.targets, self.partial_targets = self.data['data'], self.data['target'], self.data[
            'partial_target']
        else:
            self.features, self.targets, self.partial_targets = self.data['features'], self.data['logitlabels'], self.data[
                'p_labels']
        if self.features.shape[0] != self.targets.shape[0]:
            self.targets = self.targets.transpose()
            self.partial_targets = self.partial_targets.transpose()
        if type(self.targets) != np.ndarray:
            self.targets = self.targets.toarray()
            self.partial_targets = self.partial_targets.toarray()

        # normalize
        print(self.features.shape, self.targets.shape, self.partial_targets.shape)
        self.features = (self.features - self.features.mean(axis=0, keepdims=True)) / self.features.std(axis=0,
                                                                                                        keepdims=True)
        # self.kfold = KFold(n_splits=n_splits, random_state=0, shuffle=True)
        # self.train_test_idx = [ (tr_idx, te_idx) for tr_idx, te_idx in self.kfold.split(self.features)]
        self.num_features, self.num_classes = self.features.shape[-1], self.targets.shape[-1]

    def get_data(self):
        # assert k>=0 and k<self.n_splits
        # self.tr_idx, self.te_idx = self.train_test_idx[k]
        # self.train_features, self.train_targets, self.train_labels = self.features[self.tr_idx], self.partial_targets[self.tr_idx], self.targets[self.tr_idx]
        # self.test_features, self.test_targets, self.test_labels = self.features[self.te_idx], self.partial_targets[self.te_idx], self.targets[self.te_idx]
        def to_sum_one(x):
            return x / x.sum(axis=1, keepdims=True)

        def to_torch(x):
            return torch.from_numpy(x).to(torch.float32)

        # self.train_final_labels, self.test_final_labels = map(to_sum_one, (self.train_targets, self.test_targets))
        # self.train_features, self.train_targets, self.train_final_labels, self.train_labels, self.test_features, self.test_targets, self.test_final_labels, self.test_labels = map(to_torch, (self.train_features, self.train_targets, self.train_final_labels, self.train_labels, self.test_features, self.test_targets, self.test_final_labels, self.test_labels))
        self.final_labels = to_sum_one(self.targets)
        self.features, self.partial_targets, self.final_labels, self.true_labels = map(to_torch, (
            self.features, self.partial_targets, self.final_labels, self.targets
        ))

        return self.features, self.partial_targets, self.final_labels, self.true_labels


class RealWorldData(data.Dataset):
    def __init__(self, realword_dataloader):
        # self.k = k
        # self.train = train_or_not
        self.dataset = realword_dataloader.get_data()
        self.features, self.partial_targets, self.final_labels, self.true_labels = self.dataset
        # self.test_features, self.test_targets, self.test_final_labels, self.test_labels = self.test_dataset

    def __getitem__(self, index):
        # if self.train:
        #     feature, target, final, true = self.train_features[index], self.train_targets[index], self.train_final_labels[index], self.train_labels[index]
        # else:
        #     feature, target, final, true = self.test_features[index], self.test_targets[index], self.test_final_labels[index], self.test_labels[index]
        feature, target, final, true = self.features[index], self.partial_targets[index], self.final_labels[index], \
        self.true_labels[index]

        return feature, target, final, true, index

    def __len__(self):
        # if self.train:
        #     return len(self.train_features)
        # else:
        #     return len(self.test_features)
        return len(self.features)


class My_Subset(Dataset):
    """
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset, indices):
        self.indices = indices
        self.features, self.partial_targets, self.final_labels, self.true_labels \
            = dataset.features[indices, :], dataset.partial_targets[indices, :], \
            dataset.final_labels[indices, :], dataset.true_labels[indices, :]

    def __getitem__(self, index):
        feature, target, final, true = self.features[index], self.partial_targets[index], self.final_labels[index], \
            self.true_labels[index]

        return feature, target, final, true, index

    def __len__(self):
        return len(self.indices)

# if __name__ == '__main__':
#     root = "/data1/qiaocy/PLL_DataSet/"
#     datalist = [
#         'birdac.mat',
#         # 'fgnet.mat',
#         'lost.mat',
#         'LYN.mat',
#         'MSRCv2.mat',
#         'spd.mat'
#     ]
#     for dataname in os.listdir(root):
#         if not dataname.endswith('.mat'):
#             continue
#
#         print(dataname)
#         data_reader = RealwordDataLoader(root + dataname)
#         full_dataset = RealWorldData(data_reader)
#         full_data_size = len(full_dataset)
#         test_size, valid_size = int(full_data_size * 0.1), int(full_data_size * 0.1)
#         train_size = full_data_size - test_size - valid_size
#         train_dataset, valid_dataset, test_dataset = \
#             torch.utils.data.random_split(full_dataset, [train_size, valid_size, test_size], torch.Generator().manual_seed(42))
#         for item in data:
#             print(item)
#             break

