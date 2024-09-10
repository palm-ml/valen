from email.mime import image
import numpy as np
import torch
import math
import torch.nn.functional as F
from utils.utils_loss import mae_loss, mse_loss, cce_loss, gce_loss, phuber_ce_loss, focal_loss, pll_estimator
from sklearn.preprocessing import OneHotEncoder
from models_ins.resnet34 import Resnet34
from partial_models_ins.resnet import resnet
from partial_models_ins.resnext import resnext
from partial_models_ins.linear_mlp_models import mlp_model

def binarize_class(y):  
    label = y.reshape(len(y), -1)
    enc = OneHotEncoder(categories='auto') 
    enc.fit(label)
    label = enc.transform(label).toarray().astype(np.float32)     
    label = torch.from_numpy(label)
    return label

def accuracy_check(loader, model, device):
    with torch.no_grad():
        total, num_samples = 0, 0
        for images, labels in loader:
            labels, images = labels.to(device), images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            # _, y = torch.max(labels.data, 1)
            # print(predicted, labels)
            total += (predicted == labels).sum().item()
            num_samples += labels.size(0)

    return 100*(total/num_samples)

# def accuracy_check0(loader, model, device):
#     with torch.no_grad():
#         total, num_samples = 0, 0
#         truew = 0.0
#         for images, labels in loader:
#             labels, images = labels.to(device), images.to(device)
#             outputs = model(images)
#             outsoft = F.softmax(outputs, dim=1)
#             w, predicted = torch.max(outsoft.data, 1)
#             _, y = torch.max(labels.data, 1)
#             total += (predicted == y).sum().item()
#             num_samples += labels.size(0)
            
#             truew += w[predicted == y].sum().item()

#     return 100*(total/num_samples), (truew/total)

def getnewList(newlist):
	d = []
	for element in newlist:
		if not isinstance(element,list):
			d.append(element)
		else:
			d.extend(getnewList(element))
	
	return d
	
def generate_unreliable_candidate_labels(train_labels, partial_rate, noisy_rate):
    K = (torch.max(train_labels) - torch.min(train_labels) + 1).item()
    n = train_labels.shape[0]

    # Categorical Distribution
    Categorical_Matrix = torch.ones(n, K) * (noisy_rate / (K-1))
    Categorical_Matrix[torch.arange(n), train_labels] = 1 - noisy_rate
    noisy_label_sampler = torch.distributions.Categorical(probs=Categorical_Matrix)
    noisy_labels = noisy_label_sampler.sample()

    # Bernoulli Distribution
    Bernoulli_Matrix = torch.ones(n, K) * partial_rate
    Bernoulli_Matrix[torch.arange(n), train_labels] = 0
    incorrect_labels = torch.zeros(n, K)
    for i in range(n):
        incorrect_labels_sampler = torch.distributions.Bernoulli(probs=Bernoulli_Matrix[i])
        incorrect_labels_row = incorrect_labels_sampler.sample()
        while incorrect_labels_row.sum() < 1:
            incorrect_labels_row = incorrect_labels_sampler.sample()
        incorrect_labels[i] = incorrect_labels_row.clone().detach()
    # check
    partial_labels = incorrect_labels.clone().detach()
    partial_labels[torch.arange(n), noisy_labels] = 1.0
    return partial_labels

def generate_instance_independent_candidate_labels(train_labels, partial_rate):
    K = (torch.max(train_labels) - torch.min(train_labels) + 1).item()
    n = train_labels.shape[0]

    # Bernoulli Distribution
    Bernoulli_Matrix = torch.ones(n, K) * partial_rate
    Bernoulli_Matrix[torch.arange(n), train_labels] = 0
    incorrect_labels = torch.zeros(n, K)
    for i in range(n):
        incorrect_labels_sampler = torch.distributions.Bernoulli(probs=Bernoulli_Matrix[i])
        incorrect_labels_row = incorrect_labels_sampler.sample()
        while incorrect_labels_row.sum() < 1:
            incorrect_labels_row = incorrect_labels_sampler.sample()
        incorrect_labels[i] = incorrect_labels_row.clone().detach()
    # check
    partial_labels = incorrect_labels.clone().detach()
    partial_labels[torch.arange(n), train_labels] = 1.0
    avgC = partial_labels.sum() / n
    return partial_labels, avgC

def generate_instance_dependent_candidate_labels(ds, train_loader, train_labels, device):
    if ds == 'cifar10':
        weight_path = '../../../data/IDGP/partial_weights/checkpoint_c10_resnet.pt'
        model = resnet(depth=32, num_classes=10)
        rate = 0.4
    elif ds == 'mnist':
        weight_path = '../../../data/IDGP/partial_weights/checkpoint_mnist_mlp.pt'
        model = mlp_model(input_dim=28*28, output_dim=10)
        rate = 0.4
    elif ds == 'kmnist':
        weight_path = '../../../data/IDGP/partial_weights/checkpoint_kmnist_mlp.pt'
        model = mlp_model(input_dim=28*28, output_dim=10)
        rate = 0.4
    elif ds == 'fmnist':
        weight_path = '../../../data/IDGP/partial_weights/checkpoint_fashion_mlp.pt'
        model = mlp_model(input_dim=28*28, output_dim=10)
        rate = 0.4
    elif ds == 'cifar100':
        weight_path = '../../../data/IDGP/partial_weights/c100_resnext.pt'
        model = resnext(cardinality=16, depth=29, num_classes=100)
        rate = 0.04
    elif ds == 'cub200':
        weight_path = '../../../data/IDGP/partial_weights/cub200_256.pt'
        model = Resnet34(200)
        rate = 0.04

    with torch.no_grad():
        model = model.to(device)
        model.load_state_dict(torch.load(weight_path, map_location=device))
        if weight_path == '../../../data/IDGP/partial_weights/cub200_256.pt':
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


    
