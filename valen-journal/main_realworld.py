# import
import argparse
from copy import deepcopy
from operator import index
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet
import os
import pickle
import time
import random
from utils.args import extract_args
from utils.data_factory import extract_data, partialize, create_train_loader
from utils.model_factory import create_model
from utils.utils_graph import gen_adj_matrix2
from utils.utils_loss import partial_loss, alpha_loss, kl_loss, revised_target
from utils.utils_algo import dot_product_decode
from utils.metrics import evaluate_benchmark, evaluate_realworld
from utils.utils_log import Monitor, TimeUse
from models.linear import linear
from models.mlp import mlp, mlp_phi
from datasets.realworld.realworld_by_liubiao import RealwordDataLoader, RealWorldData, My_Subset

# settings
# run device gpu:x or cpu
args = extract_args()
device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
    torch.manual_seed(seed) # 为CPU设置随机种子
    torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_seed(args.seed)


def warm_up_realworld(config, model, train_loader, test_X, test_Y):
    opt = torch.torch.optim.SGD(list(model.parameters()), lr=config.lr, weight_decay=config.wd, momentum=0.9)
    partial_weight = train_loader.dataset.train_p_Y.clone().detach().to(device)
    partial_weight = partial_weight / partial_weight.sum(dim=1, keepdim=True)
    print("Begin warm-up, warm up epoch {}".format(config.warm_up))
    for _ in range(0, config.warm_up):
        for features, targets, trues, indexes in train_loader:
            features, targets, trues = map(lambda x: x.to(device), (features, targets, trues))
            outputs = model(features)
            L_ce, new_labels = partial_loss(outputs, partial_weight[indexes,:].clone().detach(), None)
            partial_weight[indexes,:] = new_labels.clone().detach()
            opt.zero_grad()
            L_ce.backward()
            opt.step()
    test_acc = evaluate_realworld(model, test_X, test_Y, device)
    print("After warm up, test acc: {:.2f}".format(test_acc))
    return model, partial_weight

def train_realworld(config):
    # random seed
    root = "../data/realworld/"
    batch_size = 100
    dataset = config.ds
    data_reader = RealwordDataLoader(root + dataset + '.mat')
    full_dataset = RealWorldData(data_reader)
    full_data_size = len(full_dataset)
    test_size, valid_size = int(full_data_size * 0.2), int(full_data_size * 0.2)
    train_size = full_data_size - test_size - valid_size
    train_dataset, valid_dataset, test_dataset = \
        torch.utils.data.random_split(full_dataset, [train_size, valid_size, test_size],
                                      torch.Generator().manual_seed(42))
    train_idx, valid_idx, test_idx = train_dataset.indices, valid_dataset.indices, test_dataset.indices
    train_dataset, valid_dataset, test_dataset = \
        My_Subset(full_dataset, train_idx), My_Subset(full_dataset, valid_idx), My_Subset(full_dataset, test_idx)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=True)
    train_X, train_Y, train_p_Y = train_loader.dataset.features, train_loader.dataset.true_labels, train_loader.dataset.partial_targets
    valid_X, valid_Y = valid_loader.dataset.features, valid_loader.dataset.true_labels
    test_X, test_Y = test_loader.dataset.features, test_loader.dataset.true_labels
    num_samples = train_X.shape[0]
    num_features = train_X.shape[-1]
    num_classes = train_Y.shape[-1]
    train_X = train_X.view((num_samples, -1))

    with TimeUse("Create Model"):
        net, enc, dec = create_model(args, num_features=num_features, num_classes=num_classes)
        net, enc, dec = map(lambda x: x.to(device), (net, enc, dec))
    print("Net:\n",net)
    print("Encoder:\n",enc)
    print("Decoder:\n",dec)
    print("The Training Set has {} samples and {} classes".format(num_samples, num_features))
    print("Average Candidate Labels is {:.4f}".format(train_p_Y.sum().item()/num_samples))
    train_loader = create_train_loader(train_X, train_Y, train_p_Y, batch_size=config.bs)
    # warm up
    net, o_array = warm_up_realworld(config, net, train_loader, test_X, test_Y)
    # compute adj matrix
    print("Compute adj maxtrix or Read.")
    with TimeUse("Adj Maxtrix"):
        adj = gen_adj_matrix2(train_X.cpu().numpy(), k=config.knn, path=os.path.abspath("middle/adjmatrix/"+args.dt+"/"+args.ds+".npy"))
    with TimeUse("Adj to Dense"):
        A = adj.to_dense()
    with TimeUse("Adj to Device"):
        adj = adj.to(device)
    # compute gcn embedding
    with TimeUse("Spmm"):
        embedding = train_X.to(device)
    prior_alpha = torch.Tensor(1, num_classes).fill_(1.0).to(device)
    # training
    opt = torch.optim.SGD(list(net.parameters())+list(enc.parameters())+list(dec.parameters()), lr=args.lr, weight_decay=args.wd, momentum=0.9)
    d_array = deepcopy(o_array)

    from utils.earlystopping import EarlyStopping
    save_path = "checkpoints_realworld/{}/{}/".format(args.ds, args.lo)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    early = EarlyStopping(patience=50,
                          path=os.path.join(save_path, "{}_lo={}_seed={}.pt".format(args.ds, args.lo, args.seed)))

    best_epoch, best_val, best_test = -1, -1, -1

    for epoch in range(0, args.ep):
        for features, targets, trues, indexes in train_loader:
            features, targets, trues = map(lambda x: x.to(device), (features, targets, trues))
            outputs = net(features)
            alpha = enc(embedding[indexes, :])
            s_alpha = F.softmax(alpha, dim=1)
            revised_alpha = torch.zeros_like(targets)
            revised_alpha[o_array[indexes,:]>0] = 1.0
            s_alpha = s_alpha * revised_alpha
            s_alpha_sum = s_alpha.clone().detach().sum(dim=1, keepdim=True)
            s_alpha = s_alpha / s_alpha_sum + 1e-2
            L_d, new_d = partial_loss(alpha, o_array[indexes,:], None)
            alpha = torch.exp(alpha/4)
            alpha = F.hardtanh(alpha, min_val=1e-2, max_val=30)
            L_alpha = alpha_loss(alpha, prior_alpha)
            dirichlet_sample_machine = torch.distributions.dirichlet.Dirichlet(s_alpha)
            d = dirichlet_sample_machine.rsample()
            x_hat = dec(d)
            x_hat = x_hat.view(features.shape)
            A_hat = F.softmax(dot_product_decode(d), dim=1)
            L_recx = F.mse_loss(x_hat, features)
            L_recy = 0.01 * F.binary_cross_entropy_with_logits(d, targets)
            L_recA = F.mse_loss(A_hat, A[indexes,:][:,indexes].to(device))
            L_rec = L_recx + L_recy + L_recA
            L_o, new_o = partial_loss(outputs, d_array[indexes,:], None)
            L = config.alpha*L_rec + config.beta*L_alpha + config.gamma * L_d + config.theta * L_o
            opt.zero_grad()
            L.backward()
            opt.step()
            new_d = revised_target(d, new_d)
            new_d = config.correct * new_d + (1 - config.correct) * o_array[indexes,:]
            d_array[indexes,:] = new_d.clone().detach()
            o_array[indexes,:] = new_o.clone().detach()
        valid_acc = evaluate_realworld(net, valid_X, valid_Y, device)
        test_acc = evaluate_realworld(net, test_X, test_Y, device)
        print("Epoch {:>3d}, valid acc: {:.2f}, test acc: {:.2f}. ".format(epoch+1, valid_acc, test_acc))
        if epoch >= 1:
            early(valid_acc, net, epoch)
        if early.early_stop:
            break
        if valid_acc > best_val:
            best_val = valid_acc
            best_epoch = epoch
            best_test = test_acc
    print("Best Epoch {:>3d}, Best valid acc: {:.2f}, test acc: {:.2f}. ".format(best_epoch, best_val, best_test))

    return best_test


def train_realworld2(config):
    # random seed
    root = "../data/realworld/"
    batch_size = 100
    dataset = config.ds
    data_reader = RealwordDataLoader(root + dataset + '.mat')
    full_dataset = RealWorldData(data_reader)
    full_data_size = len(full_dataset)
    test_size, valid_size = int(full_data_size * 0.2), int(full_data_size * 0.2)
    train_size = full_data_size - test_size - valid_size
    train_dataset, valid_dataset, test_dataset = \
        torch.utils.data.random_split(full_dataset, [train_size, valid_size, test_size],
                                      torch.Generator().manual_seed(42))
    train_idx, valid_idx, test_idx = train_dataset.indices, valid_dataset.indices, test_dataset.indices
    train_dataset, valid_dataset, test_dataset = \
        My_Subset(full_dataset, train_idx), My_Subset(full_dataset, valid_idx), My_Subset(full_dataset, test_idx)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader  = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=True)
    train_X, train_Y, train_p_Y = train_loader.dataset.features, train_loader.dataset.true_labels, train_loader.dataset.partial_targets
    valid_X, valid_Y = valid_loader.dataset.features, valid_loader.dataset.true_labels
    test_X, test_Y = test_loader.dataset.features, test_loader.dataset.true_labels
    num_samples = train_X.shape[0]
    num_features = train_X.shape[-1]
    num_classes = train_Y.shape[-1]
    train_X = train_X.view((num_samples, -1))
    with TimeUse("Create Model"):
        net, enc, dec = create_model(args, num_features=num_features, num_classes=num_classes)
        net, enc, dec = map(lambda x: x.to(device), (net, enc, dec))
    print("Net:\n",net)
    print("Encoder:\n",enc)
    print("Decoder:\n",dec)
    print("The Training Set has {} samples and {} classes".format(num_samples, num_features))
    print("Average Candidate Labels is {:.4f}".format(train_p_Y.sum().item()/num_samples))
    train_loader = create_train_loader(train_X, train_Y, train_p_Y, batch_size=config.bs)
    # warm up
    net, o_array = warm_up_realworld(config, net, train_loader, test_X, test_Y)
    # compute adj matrix
    print("Compute adj maxtrix or Read.")
    with TimeUse("Adj Maxtrix"):
        adj = gen_adj_matrix2(train_X.cpu().numpy(), k=config.knn, path=os.path.abspath("middle/adjmatrix/"+args.dt+"/"+args.ds+".npy"))
    with TimeUse("Adj to Dense"):
        A = adj.to_dense()
    with TimeUse("Adj to Device"):
        adj = adj.to(device)
    # compute gcn embedding
    with TimeUse("Spmm"):
        embedding = train_X.to(device)
    prior_alpha = torch.Tensor(1, num_classes).fill_(1.0).to(device)

    from utils.earlystopping import EarlyStopping
    save_path = "checkpoints_realworld/{}/{}/".format(args.ds, args.lo)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    early = EarlyStopping(patience=50,
                          path=os.path.join(save_path, "{}_lo={}_seed={}.pt".format(args.ds, args.lo, args.seed)))

    best_epoch, best_val, best_test = -1, -1, -1

    # training
    # opt = torch.optim.SGD(list(net.parameters())+list(enc.parameters())+list(dec.parameters()), lr=args.lr, weight_decay=args.wd, momentum=0.9)
    opt = torch.optim.Adam(list(net.parameters())+list(enc.parameters())+list(dec.parameters()), lr=args.lr, weight_decay=args.wd)
    d_array = deepcopy(o_array)
    features, targets = map(lambda x: x.to(device), (train_X, train_p_Y))
    for epoch in range(0, args.ep):
        outputs = net(features)
        alpha = enc(embedding)
        s_alpha = F.softmax(alpha, dim=1)
        revised_alpha = torch.zeros_like(targets)
        revised_alpha[o_array>0] = 1.0
        s_alpha = s_alpha * revised_alpha
        s_alpha_sum = s_alpha.clone().detach().sum(dim=1, keepdim=True)
        s_alpha = s_alpha / s_alpha_sum + 1e-2
        L_d, new_d = partial_loss(alpha, o_array, None)
        alpha = torch.exp(alpha/4)
        alpha = F.hardtanh(alpha, min_val=1e-2, max_val=30)
        L_alpha = alpha_loss(alpha, prior_alpha)
        dirichlet_sample_machine = torch.distributions.dirichlet.Dirichlet(s_alpha)
        d = dirichlet_sample_machine.rsample()
        x_hat = dec(d)
        x_hat = x_hat.view(features.shape)
        A_hat = F.softmax(dot_product_decode(d), dim=1)
        L_recx = 0.01 * F.mse_loss(x_hat, features)
        L_recy = 0.01 * F.binary_cross_entropy_with_logits(d, targets)
        L_recA = F.mse_loss(A_hat, A.to(device))
        L_rec = L_recx + L_recy + L_recA
        L_o, new_o = partial_loss(outputs, d_array, None)
        L = config.alpha*L_rec + config.beta*L_alpha + config.gamma * L_d + config.theta * L_o
        opt.zero_grad()
        L.backward()
        opt.step()
        new_d = revised_target(d, new_d)
        new_d = config.correct * new_d + (1 - config.correct) * o_array
        d_array = new_d.clone().detach()
        o_array = new_o.clone().detach()
        valid_acc = evaluate_realworld(net, valid_X, valid_Y, device)
        test_acc = evaluate_realworld(net, test_X, test_Y, device)
        print("Epoch {:>3d}, valid acc: {:.2f}, test acc: {:.2f}. ".format(epoch + 1, valid_acc, test_acc))
        if epoch >= 1:
            early(valid_acc, net, epoch)
        if early.early_stop:
            break
        if valid_acc > best_val:
            best_val = valid_acc
            best_epoch = epoch
            best_test = test_acc
    print("Best Epoch {:>3d}, Best valid acc: {:.2f}, test acc: {:.2f}. ".format(best_epoch, best_val, best_test))

    return best_test





# enter
if __name__ == "__main__":
    if args.dt == "realworld":
        if args.ds not in ['spd', 'LYN']:
            train_realworld(args)
        else:
            train_realworld2(args)
