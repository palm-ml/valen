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
from sklearn.model_selection import train_test_split
from utils.utils_algo import binarize_class, partialize

class UCIDataset(data.Dataset):
    def __init__(self, mat_path, train_or_not, partial_type='binomial', partial_rate=0.1, random_state=0):
        self.train = train_or_not
        self.data = loadmat(mat_path)
        self.features, self.labels = self.data['data'], self.data['label'][0]
        self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(self.features, self.labels, test_size=0.25) 
        self.train_data, self.train_labels, self.test_data, self.test_labels = map(torch.from_numpy, (self.train_data, self.train_labels, self.test_data, self.test_labels))
        self.train_data, self.test_data = self.train_data.to(torch.float32), self.test_data.to(torch.float32)
        self.train_labels, self.test_labels = self.train_labels.to(torch.long), self.test_labels.to(torch.long)
        if self.train:
            if partial_rate != 0.0:
                y = binarize_class(self.train_labels)
                self.train_final_labels, self.average_class_label = partialize(y, self.train_labels, partial_type, partial_rate)
            self.train_targets = deepcopy(self.train_final_labels)
            self.train_final_labels = self.train_final_labels / torch.sum(self.train_final_labels, dim=1, keepdim=True)
            self.train_label_distribution = deepcopy(self.train_final_labels)

        self.num_features = self.train_data.shape[-1]
        self.num_classes = torch.max(self.train_labels)+1
        
    
    def __getitem__(self, index):
        if self.train:
            feature, target, final, true, distr = self.train_data[index], self.train_targets[index], self.train_final_labels[index], self.train_labels[index], self.train_label_distribution[index]
        else:
            feature, target, final, true, distr = self.test_data[index], self.test_labels[index], self.test_labels[index], self.test_labels[index], self.test_labels[index]

        return feature, target, final, true, distr, index

    
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)