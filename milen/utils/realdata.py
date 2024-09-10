import numpy as np
import torch
import torch.utils.data as data
import scipy.io as sio
import os
from sklearn.preprocessing import scale

class Sample(data.Dataset):

    def __init__(self, name, idx=0, train_or_not=True, random_state=0):
        
        matfn = os.path.join('./', (name + '.mat'))
        matdata = sio.loadmat(matfn)
        self.train = train_or_not 

        datas = torch.from_numpy(scale(matdata['data'])).float()
        labels = torch.squeeze(torch.from_numpy(matdata['label_matrix'])).long()
        partial_labels = torch.from_numpy(matdata['partial_target']).float()
        test_num = int(datas.size(0)/5)
        mask_data = torch.zeros(datas.size()).byte()
        mask_data[torch.arange(idx*test_num, idx*test_num+test_num),:] = 1
        mask_label = torch.zeros(labels.size()).byte()
        mask_label[torch.arange(idx*test_num, idx*test_num+test_num)] = 1
        mask_partial_label = torch.zeros(partial_labels.size()).byte()
        mask_partial_label[torch.arange(idx*test_num, idx*test_num+test_num),:] = 1
            
        if self.train:
            self.train_data = torch.masked_select(datas, 1-mask_data).reshape(-1, datas.size(1))
            self.train_final_labels = torch.masked_select(partial_labels, 1-mask_partial_label).reshape(-1, partial_labels.size(1))
            
        else:
            self.test_data = torch.masked_select(datas, mask_data).reshape(-1, datas.size(1))
            self.test_labels = torch.masked_select(labels, mask_label).reshape(-1, labels.size(1))


    def __getitem__(self, index):

        if self.train:
            img, target  = self.train_data[index], self.train_final_labels[index]
        else:
            img, target  = self.test_data[index], self.test_labels[index]

        return img, target


    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

