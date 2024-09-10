import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from scipy.special import comb
import itertools
from utils.gen_partial_dataset import gen_partial_dataset
from utils.utils_algo import getnewList, binarize_class
	

def prepare_datasets(dataname, batch_size):
    if dataname == 'mnist':
        ordinary_train_dataset = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
        test_dataset = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
    elif dataname == 'kmnist':
        ordinary_train_dataset = dsets.KMNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
        test_dataset = dsets.KMNIST(root='./data', train=False, transform=transforms.ToTensor())
    elif dataname == 'fashion':
        ordinary_train_dataset = dsets.FashionMNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
        test_dataset = dsets.FashionMNIST(root='./data', train=False, transform=transforms.ToTensor())
    elif dataname == 'cifar10':
        train_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        test_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        ordinary_train_dataset = dsets.CIFAR10(root='./data', train=True, transform=train_transform, download=True)
        test_dataset = dsets.CIFAR10(root='./data', train=False, transform=test_transform)
    
    full_train_loader = torch.utils.data.DataLoader(dataset=ordinary_train_dataset, batch_size=len(ordinary_train_dataset.data), shuffle=True, num_workers=8)
    full_test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=len(test_dataset.data), shuffle=False, num_workers=8)

    for i, (data, labels) in enumerate(full_train_loader):
        pass
    y = binarize_class(labels) 
    ordinary_train_dataset = gen_partial_dataset(data, y.float())
    for i, (data, labels) in enumerate(full_test_loader):
        pass
    y = binarize_class(labels) 
    test_dataset = gen_partial_dataset(data, y.float())

    train_loader = torch.utils.data.DataLoader(dataset=ordinary_train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    num_classes = 10
    return full_train_loader, train_loader, test_loader, ordinary_train_dataset, test_dataset, num_classes
    

def prepare_train_loaders(dataname, full_train_loader, batch_size, partial_type):
    for i, (data, labels) in enumerate(full_train_loader):
        pass
    
    if partial_type == 'uset':
        partialY = generate_uniformset(data, labels)
    if partial_type == 'ulabel':
        partialY = generate_uniformlabel(data, labels)
    if partial_type == 'nset':
        partialY = generate_nonuniformset(data, labels)
    if partial_type == 'ccnlabel1':
        partialY = generate_ccnlabel_1(data, labels)
    if partial_type == 'ccnlabel5':
        partialY = generate_ccnlabel_5(data, labels)

    if partial_type == 'noise+partial':
        partialY = generate_noisepartial(data, labels)
    if partial_type == 'partial+noise':
        partialY = generate_partialnoise(data, labels)

    partial_matrix_dataset = gen_partial_dataset(data, partialY.float())
    partial_matrix_train_loader = torch.utils.data.DataLoader(dataset=partial_matrix_dataset, batch_size=batch_size, shuffle=True)#, num_workers=8)
    dim = int(data.reshape(-1).shape[0]/data.shape[0])

    return partial_matrix_train_loader, data, partialY, dim


def generate_uniformset(dataname, train_labels): 
    if torch.min(train_labels) > 1:
        raise RuntimeError('testError')
    elif torch.min(train_labels) == 1:
        train_labels = train_labels - 1
        
    K = torch.max(train_labels) - torch.min(train_labels) + 1
    n = train_labels.shape[0]
    cardinality = (2**K - 2).float()
    number = torch.tensor([comb(K, i+1) for i in range(K-1)]).float()
    frequency_dis = number / cardinality
    prob_dis = torch.zeros(K-1)
    for i in range(K-1):
        if i == 0:
            prob_dis[i] = frequency_dis[i]
        else:
            prob_dis[i] = frequency_dis[i]+prob_dis[i-1]

    random_n = torch.from_numpy(np.random.uniform(0, 1, n)).float()
    mask_n = torch.ones(n)
    partialY = torch.zeros(n, K)
    partialY[torch.arange(n), train_labels] = 1.0
    
    temp_num_partial_train_labels = 0 
    
    for j in range(n): # for each instance
        for jj in range(K-1):
            if random_n[j] <= prob_dis[jj] and mask_n[j] == 1:
                temp_num_partial_train_labels = jj+1 # number of candidate labels
                mask_n[j] = 0
                
        temp_num_fp_train_labels = temp_num_partial_train_labels - 1 # number of negative labels
        candidates = torch.from_numpy(np.random.permutation(K.item())).long()
        candidates = candidates[candidates!=train_labels[j]]
        temp_fp_train_labels = candidates[:temp_num_fp_train_labels]
        # temp_comp_train_labels = candidates[temp_num_fp_train_labels:]
        
        partialY[j, temp_fp_train_labels] = 1.0
    return partialY


# def generate_nonuniformset(dataname, train_labels): 
#     if torch.min(train_labels) > 1:
#         raise RuntimeError('testError')
#     elif torch.min(train_labels) == 1:
#         train_labels = train_labels - 1
        
#     K = torch.max(train_labels) - torch.min(train_labels) + 1
#     n = train_labels.shape[0]

#     # np.random.seed(0)
#     frequency_dis = np.sort(np.random.uniform(1e-4, 1, 2**(K-1)-2))
#     while len(set(frequency_dis))<2**(K-1)-2:
#         frequency_dis = np.sort(np.random.uniform(1e-4, 1, 2**(K-1)-2))
#     prob_dis = torch.ones(2**(K-1)-1)
#     for i in range(2**(K-1)-2):
#         prob_dis[i] = frequency_dis[i]

#     # np.random.seed(0)
#     random_n = torch.from_numpy(np.random.uniform(0, 1, n)).float()
#     mask_n = torch.ones(n)
#     partialY = torch.zeros(n, K)
#     partialY[torch.arange(n), train_labels] = 1.0

#     d = {}
#     for i in range(K):
#         value = []
#         for ii in range(1, K-1):
#             candidates = torch.arange(K).long()
#             candidates = candidates[candidates!=i].numpy().tolist()
#             value.append(list(itertools.combinations(candidates, ii)))
#         d[i] = getnewList(value)
    
#     temp_fp_train_labels = []
#     for j in range(n): # for each instance
#         if random_n[j] <= prob_dis[0] and mask_n[j] == 1:
#             mask_n[j] = 0
#             continue
#         for jj in range(1, 2**(K-1)-1):
#             if random_n[j] <= prob_dis[jj] and mask_n[j] == 1:
#                 temp_fp_train_labels = d[train_labels[j].item()][jj-1]
#                 break
#         partialY[j, temp_fp_train_labels] = 1.0

#     return partialY


def generate_uniformlabel(dataname, train_labels):
    K = torch.max(train_labels) - torch.min(train_labels) + 1
    n = train_labels.shape[0]
    partialY = torch.zeros(n, K)
    partialY[torch.arange(n), train_labels] = 1.0

    p = 0.1

    for i in range(n):
        # partialY[i, np.where(np.random.binomial(1, flip_prob, K)==1)] = 1.0
        partialY[i, np.where(np.random.binomial(1, p, K)==1)] = 1.0
    return partialY


# def generate_nonuniformlabel(dataname, train_labels):
#     K = torch.max(train_labels) - torch.min(train_labels) + 1
#     n = train_labels.shape[0]
#     partialY = torch.zeros(n, K)
#     partialY[torch.arange(n), train_labels] = 1.0

#     # np.random.seed(0)
#     P = np.random.rand(K, K)
#     for i in range(n):
#         partialY[i, np.where(np.random.binomial(1, P[train_labels[i],:])==1)] = 1.0
#     return partialY


# def generate_ccnlabel_1(dataname, train_labels):
#     K = (torch.max(train_labels) - torch.min(train_labels) + 1).item()
#     n = train_labels.shape[0]
#     partialY = torch.zeros(n, K)
#     partialY[torch.arange(n), train_labels] = 1.0

#     p = 0.5

#     # np.random.seed(0)
#     P = np.eye(K)
#     for idx in range(0, K):
#         P[idx, (idx+1)%K] = p
#     for i in range(n):
#         partialY[i, np.where(np.random.binomial(1, P[train_labels[i],:])==1)] = 1.0
#     return partialY


def generate_ccnlabel_5(dataname, train_labels):
    K = (torch.max(train_labels) - torch.min(train_labels) + 1).item()
    n = train_labels.shape[0]
    partialY = torch.zeros(n, K)
    partialY[torch.arange(n), train_labels] = 1.0

    p = 0.5

    # np.random.seed(0)
    P = np.eye(K)
    for idx in range(0, K):
        if (idx+1)%K+5<K:
            P[idx, (idx+1)%K:(idx+1)%K+5] = p
        else:
            P[idx, (idx+1)%K:(idx+1)%K+5] = p
            P[idx, 0:(idx+1)%K+5-K] = p
    for i in range(n):
        partialY[i, np.where(np.random.binomial(1, P[train_labels[i],:])==1)] = 1.0
    return partialY


def generate_partialnoise(dataname, train_labels):
    if torch.min(train_labels) > 1:
        raise RuntimeError('testError')
    elif torch.min(train_labels) == 1:
        train_labels = train_labels - 1
        
    K = torch.max(train_labels) - torch.min(train_labels) + 1
    n = train_labels.shape[0]
    cardinality = (2**K - 2).float()
    number = torch.tensor([comb(K, i+1) for i in range(K-1)]).float()
    frequency_dis = number / cardinality
    prob_dis = torch.zeros(K-1)
    for i in range(K-1):
        if i == 0:
            prob_dis[i] = frequency_dis[i]
        else:
            prob_dis[i] = frequency_dis[i]+prob_dis[i-1]

    random_n = torch.from_numpy(np.random.uniform(0, 1, n)).float()
    mask_n = torch.ones(n)
    partialY = torch.zeros(n, K)
    partialY[torch.arange(n), train_labels] = 1.0
    
    temp_num_partial_train_labels = 0 
    
    for j in range(n): # for each instance
        for jj in range(K-1):
            if random_n[j] <= prob_dis[jj] and mask_n[j] == 1:
                temp_num_partial_train_labels = jj+1 # number of candidate labels
                mask_n[j] = 0
                
        temp_num_fp_train_labels = temp_num_partial_train_labels - 1 # number of negative labels
        candidates = torch.from_numpy(np.random.permutation(K.item())).long()
        candidates = candidates[candidates!=train_labels[j]]
        temp_fp_train_labels = candidates[:temp_num_fp_train_labels]
        # temp_comp_train_labels = candidates[temp_num_fp_train_labels:]
        
        partialY[j, temp_fp_train_labels] = 1.0
    
    cp = 0.3
    complementary = torch.ones(1, K)
    for i in range(n):
        flag = np.random.binomial(1, cp)
        if flag==1:
            partialY[i,:] = complementary - partialY[i,:]

    return partialY


def generate_noisepartial(dataname, train_labels):
    K = (torch.max(train_labels) - torch.min(train_labels) + 1).item()
    n = train_labels.shape[0]

    noise_type = 'symmetric'
    noise_rate = 0.3
    train_noisy_labels = noisify(train_labels, noise_type, noise_rate, K, n)

    partialY = torch.zeros(n, K)
    partialY[torch.arange(n), train_noisy_labels] = 1.0
    p = 0.1
    for i in range(n):
        partialY[i, np.where(np.random.binomial(1, p, K)==1)] = 1.0
    return partialY


def noisify(train_labels, noise_type, noise_rate, K, n):
    if noise_type == 'pairflip':
        train_noisy_labels = noisify_pairflip(train_labels, noise_rate, K, n)
    if noise_type == 'symmetric':
        train_noisy_labels = noisify_multiclass_symmetric(train_labels, noise_rate, K, n)
    return train_noisy_labels


def noisify_multiclass_symmetric(y_train, noise, nb_classes, number):
    n = noise
    P = np.ones((nb_classes, nb_classes))
    P = (n / (nb_classes - 1)) * P
    for i in range(0, nb_classes):
        P[i, i] = 1. - n
        
    y_train_noisy = y_train.clone()
    for idx in np.arange(number):
        i = y_train[idx]
        flipped = np.random.multinomial(1, P[i, :], 1)[0]
        y_train_noisy[idx] = torch.from_numpy(np.where(flipped == 1)[0]).long()

    return y_train_noisy


# def noisify_pairflip(y_train, noise, nb_classes, number):
#     P = np.eye(nb_classes)
#     n = noise
#     for i in range(0, nb_classes-1):
#         P[i, i], P[i, i + 1] = 1. - n, n
#     P[nb_classes-1, nb_classes-1], P[nb_classes-1, 0] = 1. - n, n

#     y_train_noisy = y_train.clone()
#     for idx in np.arange(number):
    #     i = y_train[idx]
    #     flipped = np.random.multinomial(1, P[i, :], 1)[0]
    #     y_train_noisy[idx] = torch.from_numpy(np.where(flipped == 1)[0]).long()

    # return y_train







