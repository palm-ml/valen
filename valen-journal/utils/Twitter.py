import os
import os.path
import sys
import torch
import numpy as np
import pickle
# import h5py
import scipy
from scipy.io import loadmat
# import torch.utils.data as data
from torch.utils.data import Dataset
from copy import deepcopy
from sklearn.model_selection import KFold

from PIL import Image
import torch
import torchvision.transforms as transforms
import random
from augment.autoaugment_extra import CIFAR10Policy
from augment.cutout import Cutout
import scipy.io as sio
import h5py
import re
import torchvision
 
# 预处理数据

# print(names)
# print(label_dis)
 
# 转换成numpy数组
# np_data = np.array(data[:,2])
 
# # 转换成Tensor
# tensor_data = torch.tensor(np_data)
# lk = [[1,2,3],[4,5,6],[7,8,9]]
# print(lk[:][1:])

# print(data[:])
# print(tensor_data)

def set_seed(seed):
    torch.manual_seed(seed) # 为CPU设置随机种子
    torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.	
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def find_numbers_before_char(input_str, char):
    pattern = r'(\d+)(?=' + re.escape(char) + ')'
    match = re.search(pattern, input_str)
    return match.group(1) if match else None

root = "/home/qiaocy/code/VALEN_MILEN/VALEN_MSE/VALEN"

class Twitter(Dataset):
    def __init__(self, txt_path, root_path,  target_transform = None):

        # data_f_d = sio.loadmat('/home/zhaoyuchen/VILEN_MILEN/VALEN_IDPLL/VALEN_IDPLL/a_Twitter/Flickr_LDL_config.mat')
        self.target_size = [100, 100]

        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            self.img_list = [line.strip().split()[0] for line in lines]
            # print(self.label_dis)
            self.label_list = [int(find_numbers_before_char((line.strip().split()[0]), ".")) for line in lines]
            # print(self.img_list)
            self.loader = Image.open
            self.img_transform = target_transform
            self.root_path = root_path
        mat_path = root + '/a_Twitter/Twitter_LDL_config.mat'
        with h5py.File(mat_path, 'r') as data_f_d:
            # print(data_f_d['vote'][:])
            # print(self.label_list.shape)
            self.label_list = torch.tensor(self.label_list)
            # print(type(self.label_list))
            # self.label_dis = data_f_d['vote'][:][self.label_list, :].transpose()
            self.label_dis = ((data_f_d['vote'][:]).transpose())[self.label_list - 1, :]
            # print(self.label_dis)
            # print(self.label_dis.shape)
        
 
    def __getitem__(self, index):
        img_path = self.img_list[index]
        label = torch.tensor(np.array(self.label_dis[index]).astype(float))
        root_path = self.root_path
        # img = self.loader(os.path.join(root_path , img_path[:img_path.find('.jpg')] + '_aligned.jpg'))
        img = self.loader(os.path.join(root_path , img_path))
        # img = img.convert("RGB")
        # print(img_path)
        # print(torch.from_numpy(np.array(img)).shape)
        # if self.img_transform is not None:
        #     img = self.img_transform(img)
        # print(np.array(img).shape)
        
        transform_tool = transforms.Compose(
            [transforms.Resize(self.target_size),
            transforms.ToTensor(),  ])
        img = transform_tool(img)

        if(img.size()[0] == 1):
            img = img.repeat(3,1,1)
            # print("repeat", img.size())      
            # print("second")

        if self.img_transform is not None:
            img = self.img_transform(img)  
        return img, label

    def __len__(self):
        return len(self.img_list)
    
    
# def to_logis(label, k):
#     label = torch.tensor([[0.1, 0.2, 0.3], [0.2, 0.3, 0.1]])
#     # 找到每行的最大值及其索引
#     _, topk_indices = label.topk(k, dim=1, largest=True, sorted=True)
#     # 创建一个全为0的tensor，大小与x相同
#     zeros = torch.zeros_like(label)  # 使用torch.uint8类型以节省内存
#     # 将最大值位置设为1
#     ones = zeros.scatter_(1, topk_indices, 1)
#     return ones


def to_logis(train_labels, device):
    set_seed(35)
    # print(train_labels)
    K = train_labels.shape[1]
    # print(K)
    rate = 0.1
    avg_C = 0
    n = train_labels.shape[0]
    train_p_Y_list = []
    _, max_indices = torch.max(train_labels, dim=1)
    
    # train_p_Y = train_Y[i * batch_size:(i + 1) * batch_size].clone().detach()
    train_p_Y = torch.zeros((len(train_labels), K)).to(device)
    train_p_Y[torch.arange(len(train_labels)), max_indices] = 1.0
    partial_rate_array = train_labels.detach().clone()
    partial_rate_array[torch.arange(len(train_labels)), max_indices] = 0

    # partial_rate_array[torch.where(train_Y[i * batch_size:(i + 1) * batch_size] == 1)] = 0
    partial_rate_array = partial_rate_array / torch.max(partial_rate_array, dim=1, keepdim=True)[0]
    # partial_rate_array = partial_rate_array / torch.sum(partial_rate_array, dim=1, keepdim=True)
    partial_rate_array = partial_rate_array / partial_rate_array.mean(dim=1, keepdim=True) * rate
    partial_rate_array[partial_rate_array > 1.0] = 1.0
    partial_rate_array = torch.nan_to_num(partial_rate_array, nan=0)
    # debug_value = partial_rate_array.cpu().numpy()
    # partial_rate_array[partial_rate_array < 0.0] = 0.0
    m = torch.distributions.binomial.Binomial(total_count = 1, probs = partial_rate_array)
    # z = m.sample()
    false_indices = torch.tensor([])
    z = m.sample()
    # print(z.size())
    while(True):
        row_sums = z.sum(dim=1, keepdim = False)
        if row_sums.ge(1).all():
            print("OK")
            break
        else:
            print("NO")
            false_indices = torch.nonzero(row_sums.lt(1))
            print(false_indices.size())
            z2 = m.sample()
            z[false_indices, :] = z2[false_indices, :]
            

    
    train_p_Y[torch.where(z == 1)] = 1.0
    train_p_Y_list.append(train_p_Y)
    

    train_p_Y = torch.cat(train_p_Y_list, dim=0)
    assert train_p_Y.shape[0] == n
    avg_C = torch.sum(train_p_Y) / train_p_Y.size(0)
    train_p_Y_sum = train_p_Y.sum(dim=1, keepdim=True)

    return train_p_Y.cpu(), avg_C.item()


def load_Twitter(ds, batch_size, device, has_eval_train=True):
    GENERATE_SEED=52
    print("loading" + ds)
    # target_size = [100,100]
    target_size = [100,100]
    train_transform = None

    test_transform = transforms.Compose(
            [
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ])

    
    temp_train = Twitter(root + '/a_Twitter/twitter_train.txt', root + '/a_Twitter/images', target_transform = train_transform)
    data_size = len(temp_train)
    print("total: ", data_size)
    temp_valid = Twitter(root + '/a_Twitter/twitter_train.txt', root + '/a_Twitter/images',  target_transform = test_transform)

    train_dataset, _ = torch.utils.data.random_split(temp_train,
                                                                    [int(data_size * 0.9), data_size - int(data_size * 0.9)],
                                                                    torch.Generator().manual_seed(GENERATE_SEED))
    eval_train_dataset, valid_dataset = torch.utils.data.random_split(temp_valid,
                                                                    [int(data_size * 0.9), data_size - int(data_size * 0.9)],
                                                                    torch.Generator().manual_seed(GENERATE_SEED))  
    
    full_train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = (int)(len(train_dataset)), shuffle = False, num_workers = 8)
    for data, targets in full_train_loader:
        traindata, trainlabels = data, targets
        # print(traindata.size(), trainlabels.size())

    #trainlabel is label distribution， logis_label is partial label set
    set_seed(GENERATE_SEED)
    # train_loader_for_partial_labels = torch.utils.data.DataLoader(dataset=train_dataset_for_partial_labels, batch_size=batch_size*4, shuffle=False, num_workers=8)

    logis_label, avg_C = to_logis(trainlabels, device)
    sio.savemat('tensor.mat', {'logis_label': logis_label})


    # dis_label = trainlabels * logis_label / torch.sum(trainlabels * logis_label, dim = 1, keepdim = True)
    dis_label = trainlabels
    print("average:", avg_C)
    train_dataset = Twitter_Augmentention(traindata, logis_label, dis_label.float())
    valid_loader = torch.utils.data.DataLoader(dataset = valid_dataset, batch_size = batch_size, shuffle = False, num_workers = 8)
    eval_train_loader = torch.utils.data.DataLoader(dataset = eval_train_dataset, batch_size = batch_size, shuffle = False, num_workers = 8)
    test_loader = None

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                                                batch_size=batch_size, 
                                                                shuffle=True,
                                                                # shuffle=False, 
                                                                num_workers=8,
                                                                drop_last=True)
    dim = 100 * 100 * 3
    K = len(trainlabels[0])
    if has_eval_train:
        return train_loader, valid_loader, test_loader, dim, K, eval_train_loader
    else:
        return train_loader, valid_loader, test_loader, dim, K



class Twitter_Augmentention(Dataset):
    def __init__(self, images, logis_label, trainlabels):
        self.images = images
        self.given_label_matrix = logis_label
        self.trainlabels = (trainlabels * logis_label) / (trainlabels * logis_label).sum(dim=1, keepdim=True)
        # user-defined label (partial labels)
        # self.trainlabels = trainlabels * logis_label / torch.sum(trainlabels * logis_label, dim = 1, keepdim = True)
        print("logistic", self.given_label_matrix) 
        print("true_dis", self.trainlabels)
        self.true_labels = torch.argmax(trainlabels, dim = 1)
        # PiCO augmentation
        # self.weak_transform = transforms.Compose(
        #     [
        #     transforms.ToPILImage(),
        #     transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomApply([
        #         transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        #     ], p=0.8),
        #     transforms.RandomGrayscale(p=0.2),
        #     transforms.ToTensor(), 
        #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        # self.strong_transform = transforms.Compose(
        #     [
        #     transforms.ToPILImage(),
        #     transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        #     transforms.RandomHorizontalFlip(),
        #     RandomAugment(3, 5),
        #     transforms.ToTensor(), 
        #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

        # PLCR
        self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                ])
        self.weak_transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(100, 4, padding_mode='reflect'),
                    # transforms.ToTensor(),
                    Cutout(n_holes=1, length=16),
                    transforms.ToPILImage(),
                    CIFAR10Policy(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
        self.strong_transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(100, 4, padding_mode='reflect'),
                    # transforms.ToTensor(),
                    Cutout(n_holes=1, length=16),
                    transforms.ToPILImage(),
                    CIFAR10Policy(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])


    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, index):
        # set_seed(52)
        each_image_o = self.transform(self.images[index])
        each_image_w = self.weak_transform(self.images[index])
        each_image_s = self.strong_transform(self.images[index])
        each_label = self.given_label_matrix[index]
        each_true_labels = self.true_labels[index]
        
        return each_image_o, each_image_w, each_image_s, each_label, each_true_labels, index