import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder
import os
import hashlib 
import errno

from models_ins.resnet34 import Resnet34
from partial_models_ins.resnet import resnet
from partial_models_ins.resnext import resnext
from partial_models_ins.linear_mlp_models import mlp_model

def dot_product_decode(Z):
    A_pred = torch.sigmoid(torch.matmul(Z,Z.t()))
    return A_pred
    
def binarize_class(y):  
    label = y.reshape(len(y), -1)
    enc = OneHotEncoder(categories='auto') 
    enc.fit(label)
    label = enc.transform(label).toarray().astype(np.float32)     
    label = torch.from_numpy(label)
    return label


def partialize(y, t="binomial", p=0.5):
    y0 = torch.argmax(y, axis=1)
    new_y = y.clone()
    n, c = y.shape[0], y.shape[1]
    avgC = 0

    if t=='binomial':
        for i in range(n):
            row = new_y[i, :] 
            row[np.where(np.random.binomial(1, p, c)==1)] = 1
            while torch.sum(row) == 1:
                row[np.random.randint(0, c)] = 1
            avgC += torch.sum(row)
            

    if t=='pair':
        P = np.eye(c)
        for idx in range(0, c-1):
            P[idx, idx], P[idx, idx+1] = 1, p
        P[c-1, c-1], P[c-1, 0] = 1, p
        for i in range(n):
            row = new_y[i, :]
            idx = y0[i] 
            row[np.where(np.random.binomial(1, P[idx, :], c)==1)] = 1
            avgC += torch.sum(row)

    avgC = avgC / n    
    return new_y, avgC


def partialize2(y, y0, t="binomial", p=0.5):
    new_y = y.clone()
    n, c = y.shape[0], y.shape[1]
    avgC = 0

    if t=='binomial':
        for i in range(n):
            row = new_y[i, :] 
            row[np.where(np.random.binomial(1, p, c)==1)] = 1
            while torch.sum(row) == 1:
                row[np.random.randint(0, c)] = 1
            avgC += torch.sum(row)
            

    if t=='pair':
        P = np.eye(c)
        for idx in range(0, c-1):
            P[idx, idx], P[idx, idx+1] = 1, p
        P[c-1, c-1], P[c-1, 0] = 1, p
        for i in range(n):
            row = new_y[i, :]
            idx = y0[i] 
            row[np.where(np.random.binomial(1, P[idx, :], c)==1)] = 1
            avgC += torch.sum(row)

    avgC = avgC / n    
    return new_y, avgC


def check_integrity(fpath, md5): 
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): 
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False
    return True


def download_url(url, root, filename, md5):
    import urllib.request

    root = os.path.expanduser(root)
    fpath = os.path.join(root, filename)

    try:
        os.makedirs(root)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        print('Downloading ' + url + ' to ' + fpath)
        urllib.request.urlretrieve(url, fpath) 

def to_logits(y):
    '''
        将y中每一行值最大的位置赋值为1, 其余位置为0
    '''
    y_ = torch.zeros_like(y)
    col = torch.argmax(y, axis=1)
    row = [ i for i in range(0, len(y))]
    y_[row, col] = 1

    return y_


# def feature_partialize(train_X, train_Y, model, weight_path, device, rate=0.4):
#     with torch.no_grad():
#         model = model.to(device)
#         model.load_state_dict(torch.load(weight_path, map_location=device))
#         avg_C = 0
#         train_X, train_Y = train_X.to(device), train_Y.to(device)
#         _, outputs = model(train_X)
#         train_p_Y = train_Y.clone().detach()
#         partial_rate_array = F.softmax(outputs, dim=1).clone().detach()
#         partial_rate_array[torch.where(train_Y==1)] = 0
#         partial_rate_array = partial_rate_array / torch.max(partial_rate_array, dim=1, keepdim=True)[0]
#         partial_rate_array = partial_rate_array / partial_rate_array.mean(dim=1, keepdim=True) * rate
#         partial_rate_array[partial_rate_array > 1.0] = 1.0
#         m = torch.distributions.binomial.Binomial(total_count=1, probs=partial_rate_array)
#         z = m.sample()
#         train_p_Y[torch.where(z == 1)] = 1.0
#     avg_C = torch.sum(train_p_Y) / train_p_Y.size(0)
#     return train_p_Y, avg_C.item()

# def feature_partialize2(train_X, train_Y, model, weight_path, device, rate=0.4, batch_size=2000):
#     with torch.no_grad():
#         model = model.to(device)
#         model.load_state_dict(torch.load(weight_path, map_location=device))
#         avg_C = 0
#         train_X, train_Y = train_X.to(device), train_Y.to(device)
#         train_p_Y_list = []
#         step = train_X.size(0) // batch_size
#         for i in range(0, step):
#             _, outputs = model(train_X[i*batch_size:(i+1)*batch_size])
#             train_p_Y = train_Y[i*batch_size:(i+1)*batch_size].clone().detach()
#             partial_rate_array = F.softmax(outputs, dim=1).clone().detach()
#             partial_rate_array[torch.where(train_Y[i*batch_size:(i+1)*batch_size]==1)] = 0
#             partial_rate_array = partial_rate_array / torch.max(partial_rate_array, dim=1, keepdim=True)[0]
#             partial_rate_array = partial_rate_array / partial_rate_array.mean(dim=1, keepdim=True) * rate
#             partial_rate_array[partial_rate_array > 1.0] = 1.0
#             m = torch.distributions.binomial.Binomial(total_count=1, probs=partial_rate_array)
#             z = m.sample()
#             train_p_Y[torch.where(z == 1)] = 1.0
#             train_p_Y_list.append(train_p_Y)
#         train_p_Y = torch.cat(train_p_Y_list, dim=0)
#         assert train_p_Y.shape[0] == train_X.shape[0]
#     avg_C = torch.sum(train_p_Y) / train_p_Y.size(0)
#     return train_p_Y, avg_C.item()

def accuracy_check(model, loader, device):
    with torch.no_grad():
        total, num_samples = 0, 0
        for images, labels in loader:
            labels, images = labels.to(device), images.to(device)
            _, outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            # _, y = torch.max(labels.data, 1)
            # print(predicted, labels)
            total += (predicted == labels).sum().item()
            num_samples += labels.size(0)

    return 100*(total/num_samples)

def generate_instance_dependent_candidate_labels(ds, train_loader, train_labels, device):
    if ds == 'cifar10':
        weight_path = '/home/qiaocy/data/IDGP/partial_weights/checkpoint_c10_resnet.pt'
        model = resnet(depth=32, num_classes=10)
        rate = 0.4
    elif ds == 'mnist':
        weight_path = '/home/qiaocy/data/IDGP/partial_weights/checkpoint_mnist_mlp.pt'
        model = mlp_model(input_dim=28*28, output_dim=10)
        rate = 0.4
    elif ds == 'kmnist':
        weight_path = '/home/qiaocy/data/IDGP/partial_weights/checkpoint_kmnist_mlp.pt'
        model = mlp_model(input_dim=28*28, output_dim=10)
        rate = 0.4
    elif ds == 'fmnist':
        weight_path = '/home/qiaocy/data/IDGP/partial_weights/checkpoint_fashion_mlp.pt'
        model = mlp_model(input_dim=28*28, output_dim=10)
        rate = 0.4
    elif ds == 'cifar100':
        weight_path = '/home/qiaocy/data/IDGP/partial_weights/c100_resnext.pt'
        model = resnext(cardinality=16, depth=29, num_classes=100)
        rate = 0.04
    elif ds == 'cub200':
        weight_path = '/home/qiaocy/data/IDGP/partial_weights/cub200_256.pt'
        model = Resnet34(200)
        rate = 0.04

    with torch.no_grad():
        model = model.to(device)
        model.load_state_dict(torch.load(weight_path, map_location=device))
        if weight_path == '/home/qiaocy/data/IDGP/partial_weights/cub200_256.pt':
            model = model.model
        # model.eval()
        avg_C = 0
        K = (torch.max(train_labels) - torch.min(train_labels) + 1).item()
        n = train_labels.shape[0]
        train_p_Y_list = []
        
        for images, labels in train_loader:
            labels, images = labels.to(device), images.to(device)
            outputs = model(images)
            # train_p_Y = train_Y[i * batch_size:(i + 1) * batch_size].clone().detach()
            train_p_Y = torch.zeros((len(images), K))
            train_p_Y[torch.arange(len(images)), labels] = 1.0
            partial_rate_array = F.softmax(outputs, dim=1).clone().detach()
            # partial_rate_array[torch.where(train_Y[i * batch_size:(i + 1) * batch_size] == 1)] = 0
            partial_rate_array[torch.arange(labels.shape[0]), labels] = 0
            partial_rate_array = partial_rate_array / torch.max(partial_rate_array, dim=1, keepdim=True)[0]
            # partial_rate_array = partial_rate_array / torch.sum(partial_rate_array, dim=1, keepdim=True)
            partial_rate_array = partial_rate_array / partial_rate_array.mean(dim=1, keepdim=True) * rate
            partial_rate_array[partial_rate_array > 1.0] = 1.0
            partial_rate_array = torch.nan_to_num(partial_rate_array, nan=0)
            # debug_value = partial_rate_array.cpu().numpy()
            # partial_rate_array[partial_rate_array < 0.0] = 0.0
            m = torch.distributions.binomial.Binomial(total_count=1, probs=partial_rate_array)
            z = m.sample()
            train_p_Y[torch.where(z == 1)] = 1.0
            train_p_Y_list.append(train_p_Y)
        train_p_Y = torch.cat(train_p_Y_list, dim=0)
        assert train_p_Y.shape[0] == n
    avg_C = torch.sum(train_p_Y) / train_p_Y.size(0)
    train_p_Y_sum = train_p_Y.sum(dim=1, keepdim=True)
    del model
    return train_p_Y.cpu(), avg_C.item()

    
