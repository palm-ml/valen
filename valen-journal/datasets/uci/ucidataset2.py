import os
import os.path
import sys
import torch
import numpy as np
import pickle 
import h5py
import scipy
from scipy.io import loadmat
import torch.utils.data as data
from copy import deepcopy


class UCIDataset2(data.Dataset):
    def __init__(self, mat_path, train=True, k=0):
        self.train = train
        self.data = loadmat(mat_path)
        self.features, self.logitlabels, self.p_labels, self.tr, self.te = self.data['features'], self.data['logitlabels'], self.data['p_labels'], self.data['tr'], self.data['te']
        self.features = (self.features - self.features.mean(axis=0, keepdims=True))/self.features.std(axis=0, keepdims=True)
        print(self.tr.shape)
        try:
            print("tr_idx index 3")
            self.tr_idx = self.tr[0][k][0] - 1
            self.te_idx = self.te[0][k][0] - 1
        except:
            print("tr_idx index 2")
            self.tr_idx = self.tr[k] - 1
            self.te_idx = self.te[k] - 1
        print(self.tr_idx)
        if self.train:
            self.train_features, self.train_logitlabels, self.train_p_labels = self.features[self.tr_idx], self.logitlabels[self.tr_idx], self.p_labels[self.tr_idx]
            self.train_features, self.train_logitlabels, self.train_p_labels = map(torch.from_numpy, (self.train_features, self.train_logitlabels, self.train_p_labels))
            self.train_features, self.train_logitlabels, self.train_p_labels = self.train_features.to(torch.float32), self.train_logitlabels.to(torch.float32), self.train_p_labels.to(torch.float32)
        else:
            self.test_features, self.test_logitlabels, self.test_p_labels = self.features[self.te_idx], self.logitlabels[self.te_idx], self.p_labels[self.te_idx]
            self.test_features, self.test_logitlabels, self.test_p_labels = map(torch.from_numpy, (self.test_features, self.test_logitlabels, self.test_p_labels))
            self.test_features, self.test_logitlabels, self.test_p_labels = self.test_features.to(torch.float32), self.test_logitlabels.to(torch.float32), self.test_p_labels.to(torch.float32)
        self.num_features, self.num_classes = self.features.shape[-1], self.logitlabels.shape[-1]

    
    def __getitem__(self, index):
        if self.train:
            feature, p_labels, labels = self.train_features[index], self.train_p_labels[index], self.train_logitlabels[index]
        else:
            feature, p_labels, labels = self.test_features[index], self.test_p_labels[index], self.test_logitlabels[index]

        return feature, p_labels, labels, index

    
    def __len__(self):
        if self.train:
            return len(self.train_features)
        else:
            return len(self.test_features)


class UCIDataset(data.Dataset):
    def __init__(self, mat_path):
        self.data = loadmat(mat_path)
        self.features, self.logitlabels = self.data['features'], self.data['logitlabels']
        self.features = (self.features - self.features.mean(axis=0, keepdims=True))/self.features.std(axis=0, keepdims=True)
        self.num_features, self.num_classes = self.features.shape[-1], self.logitlabels.shape[-1]
        self.features, self.logitlabels = map(torch.from_numpy, (self.features, self.logitlabels))
        self.features, self.logitlabels = self.features.to(torch.float32), self.logitlabels.to(torch.float32)
        
    
    def __getitem__(self, index):

        feature, true = self.features[index], self.logitlabels[index]

        return feature, true, index

    
    def __len__(self):
        
        return len(self.features)
