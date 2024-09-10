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


class TrueDataReader:
    def __init__(self, mat_file):
        # 判断是否是.mat文件
        if mat_file[-4:]!='.mat':
            raise Exception('Not a \'.mat\' file')
        # 读取.mat文件
        try:
            self.read_method = 1
            self.mat_data = h5py.File(mat_file, 'r')
            print('Use h5py reader to read: - ',os.path.basename(mat_file))
            self.features = self.correct(self.mat_data['data'][()])
            self.tr_idx = self.read_idx(self.mat_data['tr_idx'][()])
            self.te_idx = self.read_idx(self.mat_data['te_idx'][()])
            try:
                self.partial_labels = self.correct(self.mat_data['partial_target'][()])
                self.labels = self.correct(self.mat_data['target'][()])
            except:
                print('read group.')
                # .mat中出现group组
                pl_row = self.mat_data['partial_target']['ir'][()]
                pl_col = self.mat_data['partial_target']['jc'][()]
                self.partial_labels = self.correct(self.mat_data['partial_target']['data'][()], coordinate=(pl_row, pl_col))
                l_row = self.mat_data['target']['ir'][()]
                l_col = self.mat_data['target']['jc'][()]
                self.labels = self.correct(self.mat_data['target']['data'][()], coordinate=(l_row, l_col))
        except:
            self.read_method = 0
            self.mat_data = loadmat(mat_file)
            print('Use scipy reader to read: -',os.path.basename(mat_file))
            self.features = self.correct(self.mat_data['data'])
            self.labels = self.correct(self.mat_data['target'])
            self.partial_labels = self.correct(self.mat_data['partial_target'])
            self.tr_idx = self.mat_data['tr_idx']
            self.te_idx = self.mat_data['te_idx']
        self.normalize_idx()
        # 数据信息
        self.n_features = self.features.shape[-1]
        self.n_classes = self.labels.shape[-1]
        # 当前的训练集和测试集
        self.train_data, self.test_data = None, None

        

        
    # 行列校正等
    def correct(self, data, coordinate=None, shape=None):
        if type(data) != np.ndarray:
            try:
                data = data.toarray()
            except:
                data = np.ndarray(data)
        try:
            assert len(data.shape) == 2
            if data.shape[0]>=data.shape[1]:
                return data
            else:
                return np.transpose(data)
        except:
            row, col = coordinate
            data = scipy.sparse.csr_matrix((data, row, col))
            data = data.toarray()
            return self.correct(data)
    
    # 读取k折交叉验证的数据划分
    def read_idx(self, x):
        idx = [] 
        _, row = x.shape           
        for i in range(0, row):
            idx.append(self.mat_data[x[0,i]][:].T[0])
        return idx
    
    # 将idx转化成统一的list格式
    def normalize_idx(self):
        tr_idx = []
        te_idx = []
        for k in range(0, 10):
            if self.read_method:
                tr_idx.append(list(map(lambda x: int(x)-1, self.tr_idx[k])))
                te_idx.append(list(map(lambda x: int(x)-1, self.te_idx[k])))
            else:
                tr_idx.append(list(map(lambda x: int(x)-1, self.tr_idx[k][0][0])))
                te_idx.append(list(map(lambda x: int(x)-1, self.te_idx[k][0][0])))
        self.tr_idx = tr_idx
        self.te_idx = te_idx

    # 获得数据
    def getdata(self, features_f, labels_f):
        features, partial_labels, labels = self.features, self.partial_labels, self.labels
        if features_f != None:
            features = features_f(features)
        if labels_f != None:
            partial_labels, labels = map(labels_f, (partial_labels, labels))
        return features, partial_labels, labels

    def k_cross_validation(self, k=0):
        tr_idx = self.tr_idx[k]
        te_idx = self.te_idx[k]
        self.train_data = (self.features[tr_idx], self.partial_labels[tr_idx], self.labels[tr_idx])
        self.test_data = (self.features[te_idx], self.partial_labels[te_idx], self.labels[te_idx])
        return self.train_data, self.test_data


class TrueData(data.Dataset):
    def __init__(self, mat_path, train_or_not, k_fold_order=0):
        self.train = train_or_not
        self.train_dataset, self.test_dataset = TrueDataReader(mat_path).k_cross_validation(k_fold_order)
        self.train_data, self.train_final_labels, self.train_labels = map(torch.from_numpy, self.train_dataset)
        self.train_data = self.train_data.to(torch.float32)
        self.train_final_labels = self.train_final_labels.to(torch.float32)
        self.train_labels = self.train_labels.to(torch.float32) 
        self.train_label_distribution = deepcopy(self.train_final_labels)
        self.mean = self.train_data.mean(axis=0, keepdim=True)
        self.std = self.train_data.std(axis=0, keepdim=True)
        self.train_data = (self.train_data - self.mean)/self.std
        self.train_data.to(torch.float32)
        self.train_final_labels = self.train_final_labels.to(torch.float32)
        self.train_labels = self.train_labels.to(torch.float32) 

        self.test_data, self.test_final_labels, self.test_labels = self.test_dataset
        self.test_data, self.test_final_labels, self.test_labels = map(torch.from_numpy, self.test_dataset)
        self.test_data = self.test_data.to(torch.float32)
        self.test_final_labels = self.test_final_labels.to(torch.float32)
        self.test_labels = self.test_labels.to(torch.float32) 
        self.test_data = (self.test_data - self.mean)/self.std
        self.num_features = self.train_data.shape[-1]
        self.num_classes = self.train_labels.shape[-1]
        
    
    def __getitem__(self, index):
        if self.train:
            feature, target, true, distr = self.train_data[index], self.train_final_labels[index], self.train_labels[index], self.train_label_distribution[index]
        else:
            feature, target, true, distr = self.test_data[index], self.test_labels[index], self.test_labels[index], self.test_labels[index]

        return feature, target, true, distr, index

    
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

        
        


    

    



if __name__ == '__main__':
    root = '/data1/qiaocy/Cimap_wj_dataset/REAL/'
    for dataname in os.listdir(root):
        if not dataname.endswith('.mat'):
            continue
        data = TrueData(root + dataname, train_or_not=True)
        for item in data:
            print(item)
            break
