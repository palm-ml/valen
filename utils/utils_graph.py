import os 
import os.path as p 
import numpy as np 
import torch
import pickle
import random
from torch.utils.data import Dataset
import scipy.io as scio
import scipy.sparse as sp
from sklearn.metrics import euclidean_distances
from functools import partial


def setup_seed(seed):
    # 设置随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def to_logits(y):
    '''
        将y中每一行值最大的位置赋值为1, 其余位置为0
    '''
    y_ = np.zeros_like(y, dtype='float32')
    col = np.argmax(y, axis=1)
    row = [ i for i in range(0, len(y))]
    y_[row, col] = 1

    return y_

def DistancesMatrix(X_, Y_, device=torch.device('cpu')):
    X, Y = X_.clone().detach().to(device), Y_.clone().detach().to(device)
    X_sum = torch.sum(X*X, axis=1, keepdims=True)
    Y_sum = torch.sum(Y*Y, axis=1, keepdims=True)
    Ones = torch.ones((1, X_sum.shape[0]), dtype=torch.float32).to(device)
    distances_matrix = X_sum@Ones+(Y_sum@Ones).T-2*(X@Y.T)
    return torch.sqrt(torch.abs(distances_matrix))


def gen_adj_matrix(features, k=10, device=torch.device('cpu')):
    N = features.size(0)
    adj_m = torch.zeros((N, N), dtype=torch.float32)
    # 计算距离矩阵
    dm = DistancesMatrix(features, features, device)
    max_val = torch.max(dm) + 1
    # 将对角线赋值为最大值
    map_list = [ i for i in range(0, N)]
    dm[map_list, map_list] = max_val
    # 找出每一行最小值的位置然后赋值为最大值，迭代10次，找出离该向量最近的10个值
    for _ in range(0, k):
        min_list = torch.argmin(dm, axis = 1)
        dm[map_list, min_list] = max_val
        adj_m[map_list, min_list] = 1
    return adj_m

def adj_normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def gen_adj_matrix2(X, k=10, path=""):
    if os.path.exists(path):
        print("Found adj matrix file and Load.")
        adj_m = np.load(path)
        print("Adj matrix Finished.")
    else:
        print("Not Found adj matrix file and Compute.")
        dm = euclidean_distances(X, X)
        adj_m = np.zeros_like(dm)
        row = np.arange(0, X.shape[0])
        dm[row, row] = np.inf
        for _ in range(0, k):
            col = np.argmin(dm, axis=1)
            dm[row, col] = np.inf
            adj_m[row, col] = 1.0
        np.save(path, adj_m)
        print("Adj matrix Finished.")
    adj_m = sp.coo_matrix(adj_m)
    adj_m = adj_normalize(adj_m + sp.eye(adj_m.shape[0]))
    adj_m = sparse_mx_to_torch_sparse_tensor(adj_m)
    return adj_m








def preprocess_graph(adj):
    adj = sp.csr_matrix(adj)
    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    adj = adj + adj.T
    adj_label = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_norm = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    return torch.sigmoid(torch.from_numpy(adj_label.todense()).to(torch.float32)), torch.from_numpy(adj_norm.todense()).to(torch.float32)


def sharpen(x, T=2):
    return x.pow(T)/x.pow(T).sum()


def correct_squence(d, idx, device):
    zipped = list(zip(idx.clone().detach(), d.clone().detach()))
    zipped.sort()
    print(zipped[0:3])
    d = []
    for item in zipped:
        d.append(item[0])
    return torch.Tensor(d).to(device)

