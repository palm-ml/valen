# import
import argparse
from copy import deepcopy
from operator import index
import torch
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet
import os
import pickle
import time
import numpy as np
import random
from utils.args import extract_args
# from utils.data_factory import extract_data, partialize, create_train_loader
from utils.model_factory import create_model
from utils.utils_graph import gen_adj_matrix2
from utils.utils_loss import partial_loss, alpha_loss, kl_loss, revised_target
from utils.utils_algo import dot_product_decode, accuracy_check
from utils.metrics import evaluate_realworld
from utils.utils_log import Monitor, TimeUse
from utils.models import mlp_model, linear_model, save_model, LeNet
from cifar_models import resnet, densenet
from models.VGAE import VAE_Bernulli_Decoder
from utils.cifar10 import load_cifar10
from utils.cifar100 import load_cifar100
from utils.fmnist import load_fmnist
from utils.kmnist import load_kmnist
from utils.mnist import load_mnist


# settings
# run device gpu:x or cpu
args = extract_args()
def set_seed(seed):
    torch.manual_seed(seed) # 为CPU设置随机种子
    torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.	
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_seed(args.seed)

device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else 'cpu')



def warm_up_benchmark(config, model, train_loader, valid_loader, test_loader):
    opt = torch.torch.optim.SGD(list(model.parameters()), lr=config.lr, weight_decay=config.wd, momentum=0.9)
    partial_weight = train_loader.dataset.given_label_matrix.clone().detach().to(device)
    partial_weight = partial_weight / partial_weight.sum(dim=1, keepdim=True)
    print("Begin warm-up, warm up epoch {}".format(config.warm_up))
    for _ in range(0, config.warm_up):
        for features, features_w, features_s, targets, trues, indexes in train_loader:
            features, features_w, features_s, targets, trues = map(lambda x: x.to(device), (features, features_w, features_s, targets, trues))
            _, outputs_w = model(features_w)
            _, outputs_s = model(features_s)
            phi, outputs = model(features)
            L_ce_o, new_labels = partial_loss(outputs, partial_weight[indexes,:].clone().detach(), None)
            L_ce_w, _ = partial_loss(outputs_w, partial_weight[indexes,:].clone().detach(), None)
            L_ce_s, _ = partial_loss(outputs_s, partial_weight[indexes,:].clone().detach(), None)
            L_ce = L_ce_o + L_ce_w + L_ce_s
            partial_weight[indexes,:] = new_labels.clone().detach()
            opt.zero_grad()
            L_ce.backward()
            opt.step()
        valid_acc = accuracy_check(model, valid_loader, device)
        test_acc  = accuracy_check(model, test_loader, device)
        print("After warm up, valid acc: {:.2f}, test acc: {:.2f}".format(valid_acc, test_acc))
    print("Extract feature.")
    feature_extracted = torch.zeros((len(train_loader.dataset), phi.shape[-1])).to(device)
    with torch.no_grad():
        for features, _, _, targets, trues, indexes in train_loader:
            features, targets, trues = map(lambda x: x.to(device), (features, targets, trues))
            feature_extracted[indexes, :] = model(features)[0]
    return model, feature_extracted, partial_weight


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

# train benchmark
def train_benchmark(config):
    if args.ds == "cifar10":
        train_loader, valid_loader, test_loader, dim, K = load_cifar10(args.ds, batch_size=args.bs, device=device)
    if args.ds == "cifar100":
        train_loader, valid_loader, test_loader, dim, K = load_cifar100(args.ds, batch_size=args.bs, device=device)
    if args.ds == "fmnist":
        train_loader, valid_loader, test_loader, dim, K = load_fmnist(args.ds, batch_size=args.bs, device=device)
    if args.ds == "kmnist":
        train_loader, valid_loader, test_loader, dim, K = load_kmnist(args.ds, batch_size=args.bs, device=device)
    if args.ds == "mnist":
        train_loader, valid_loader, test_loader, dim, K = load_mnist(args.ds, batch_size=args.bs, device=device)
    
    if args.mo == 'mlp':
        net = mlp_model(input_dim=dim, output_dim=K)
    elif args.mo == 'linear':
        net = linear_model(input_dim=dim, output_dim=K)
    elif args.mo == 'lenet':
        net = LeNet(out_dim=K)
    elif args.mo == 'densenet':
        net = densenet(num_classes=K)
    elif args.mo == 'resnet':
        net = resnet(depth=32, num_classes=K)
    enc = deepcopy(net)
    dec = VAE_Bernulli_Decoder(K, dim, dim)
        
    net, enc, dec = map(lambda x: x.to(device), (net, enc, dec))

    from utils.earlystopping import EarlyStopping
    save_path = "checkpoints_2/{}/{}/".format(args.ds, args.lo)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    early = EarlyStopping(patience=50, path=os.path.join(save_path, "{}_lo={}_seed={}.pt".format(args.ds, args.lo, args.seed)))

    # warm up
    net, feature_extracted, o_array = warm_up_benchmark(config, net, train_loader, valid_loader, test_loader)
    if config.partial_type == "feature" and config.ds in ["kmnist", "cifar10"]:
        print("Copy Net.")
        enc = deepcopy(net)
    # compute adj matrix
    print("Compute adj maxtrix or Read.")
    with TimeUse("Adj Maxtrix"):
        adj = gen_adj_matrix2(feature_extracted.cpu().numpy(), k=config.knn, path=os.path.abspath("middle/adjmatrix/"+args.dt+"/"+args.ds+".npy"))
    with TimeUse("Adj to Dense"):
        A = adj.to_dense()
    with TimeUse("Adj to Device"):
        adj = adj.to(device)
    # compute gcn embedding
    # with TimeUse("Spmm"):
    #     embedding = train_X.to(device)
    prior_alpha = torch.Tensor(1, K).fill_(1.0).to(device)
    # training
    # if config.ds != "cifar10":
    print("Use SGD with 0.9 momentum")
    opt = torch.optim.SGD(list(net.parameters())+list(enc.parameters())+list(dec.parameters()), lr=args.lr, weight_decay=args.wd, momentum=0.9)
    # else:
    #     print("Use Adam.")
    #     opt = torch.optim.Adam(list(net.parameters())+list(enc.parameters())+list(dec.parameters()), lr=args.lr, weight_decay=args.wd)
    # mit = Monitor(num_samples, num_classes)
    d_array = deepcopy(o_array)

    best_val, best_test, best_epoch = -1, -1, -1
    for epoch in range(0, args.ep):
        net.train()
        enc.train()
        for features, features_w, features_s, targets, trues, indexes in train_loader:
            features, features_w, features_s, targets, trues = map(lambda x: x.to(device), (features, features_w, features_s, targets, trues))
            _, outputs = net(features)
            _, outputs_w = net(features_w)
            _, outputs_s = net(features_s)

            _, alpha = enc(features)
            _, alpha_w = enc(features_w)
            _, alpha_s = enc(features_s)

            s_alpha = F.softmax(alpha, dim=1)

            revised_alpha = torch.zeros_like(targets)
            revised_alpha[o_array[indexes,:]>0] = 1.0
            s_alpha = s_alpha * revised_alpha
            s_alpha = s_alpha / s_alpha.sum(dim=1, keepdim=True) + 1e-6

            L_d_o, new_d = partial_loss(alpha, o_array[indexes,:], None)
            L_d_w, _     = partial_loss(alpha_w, o_array[indexes,:], None)
            L_d_s, _     = partial_loss(alpha_s, o_array[indexes,:], None)
            L_d = L_d_o + L_d_w + L_d_s

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
            L_recA = F.mse_loss(A_hat, A[indexes,:][:,indexes].to(device))
            L_rec = L_recx + L_recy + L_recA
            L_o_o, new_o = partial_loss(outputs,   d_array[indexes,:], None)
            L_o_w, _     = partial_loss(outputs_w, d_array[indexes,:], None)
            L_o_s, _     = partial_loss(outputs_s, d_array[indexes,:], None)
            L_o = L_o_o + L_o_w + L_o_s
            L = config.alpha*L_rec + config.beta*L_alpha + config.gamma * L_d + config.theta * L_o
            opt.zero_grad()
            L.backward()
            opt.step()
            new_d = revised_target(d, new_d)
            new_d = config.correct * new_d + (1 - config.correct) * o_array[indexes,:]
            d_array[indexes,:] = new_d.clone().detach()
            o_array[indexes,:] = new_o.clone().detach()
        net.eval()
        valid_acc = accuracy_check(loader=valid_loader, model=net, device=device)
        test_acc  = accuracy_check(loader=test_loader,  model=net, device=device)
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
    

def train_realworld(config):
    avg_acc = 0.0
    for k in range(0, 5):
        print("fold {}".format(k))
        train_X, train_Y, train_p_Y, test_X, test_Y = next(extract_data(config))
        num_samples = train_X.shape[0]
        train_X_shape = train_X.shape
        train_X = train_X.view((num_samples, -1))
        num_features = train_X.shape[-1]
        train_X = train_X.view(train_X_shape)
        num_classes = train_Y.shape[-1]
        with TimeUse("Create Model"):
            net, enc, dec = create_model(args, num_features=num_features, num_classes=num_classes)
            net, enc, dec = map(lambda x: x.to(device), (net, enc, dec))
        print("Net:\n",net)
        print("Encoder:\n",enc)
        print("Decoder:\n",dec)
        print("The Training Set has {} samples and {} classes".format(num_samples, num_features))
        print("Average Candidate Labels is {:.2f}".format(train_p_Y.sum().item()/num_samples))
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
            test_acc = evaluate_realworld(net, test_X, test_Y, device)
            print("Epoch {}, test acc: {:.2f}".format(epoch, test_acc))
        avg_acc = avg_acc + test_acc
    print("Avg Acc: {:.2f}".format(avg_acc/5))


def train_realworld2(config):
    avg_acc = 0.0
    for k in range(0, 5):
        print("fold {}".format(k))
        train_X, train_Y, train_p_Y, test_X, test_Y = next(extract_data(config))
        num_samples = train_X.shape[0]
        train_X_shape = train_X.shape
        train_X = train_X.view((num_samples, -1))
        num_features = train_X.shape[-1]
        train_X = train_X.view(train_X_shape)
        num_classes = train_Y.shape[-1]
        with TimeUse("Create Model"):
            net, enc, dec = create_model(args, num_features=num_features, num_classes=num_classes)
            net, enc, dec = map(lambda x: x.to(device), (net, enc, dec))
        print("Net:\n",net)
        print("Encoder:\n",enc)
        print("Decoder:\n",dec)
        print("The Training Set has {} samples and {} classes".format(num_samples, num_features))
        print("Average Candidate Labels is {:.2f}".format(train_p_Y.sum().item()/num_samples))
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
            test_acc = evaluate_realworld(net, test_X, test_Y, device)
            print("Epoch {}, test acc: {:.2f}".format(epoch, test_acc))
        avg_acc = avg_acc + test_acc
    print("Avg Acc: {:.2f}".format(avg_acc/5))





# enter
if __name__ == "__main__":
    if args.dt == "benchmark":
        train_benchmark(args)
    if args.dt == "realworld":
        if args.ds not in ['spd', 'LYN']:
            train_realworld(args)
        else:
            train_realworld2(args)
