import copy
import os
import random
import argparse

import torch
import torch.nn.functional as F
import numpy as np
from sensitivity_utils.earlystopping import EarlyStopping

from sensitivity_utils.utils_loss import proden_loss, rc_loss
from sensitivity_utils.utils_algo import accuracy_check
from sensitivity_utils.models import info_mlp_model, mlp_model, linear_model, LeNet
from cifar_models import densenet, resnet
from sensitivity_utils.cifar10 import load_cifar10
from sensitivity_utils.cifar100 import load_cifar100
from sensitivity_utils.fmnist import load_fmnist
from sensitivity_utils.kmnist import load_kmnist
from sensitivity_utils.mnist import load_mnist
from info_nce import InfoNCE


parser = argparse.ArgumentParser()
parser.add_argument('-lr', help='optimizer\'s learning rate', type=float, default=5e-2)
parser.add_argument('-wd', help='weight decay', type=float, default=1e-3)
parser.add_argument('-bs', help='batch size', type=int, default=256)
parser.add_argument('-ep', help='number of epochs', type=int, default=250)
parser.add_argument('-wep', help='number of warm-up epochs', type=int, default=2)
parser.add_argument('-ds', help='specify a dataset', type=str, default='cifar100', required=False)
parser.add_argument('-mo', help='model name', type=str, default='resnet', required=False)
parser.add_argument('-imo', help='information bottleneck model name', type=str, default='linear', required=False)
parser.add_argument('-lo', help='specify a loss function', default='milen', type=str, required=False)
parser.add_argument('-wlo', help='specify a warm_up loss function', default='proden', type=str, required=False)
parser.add_argument('-seed', help='random seed', default=0, type=int)
parser.add_argument('-partial_rate', type=float, default=0.5)
parser.add_argument('-gpu', type=str, default="0")

parser.add_argument('-lambda_', help='loss weight', type=float, default=1)
parser.add_argument('-loss_w_1', help='loss weight', type=float, default=1)
parser.add_argument('-loss_w_2', help='loss weight', type=float, default=1)
parser.add_argument('-beta_1', help='loss weight', type=float, default=1)
parser.add_argument('-beta_2', help='loss weight', type=float, default=1)
parser.add_argument('-T_1', help='hyper param', type=float, default=1)
parser.add_argument('-T_2', help='hyper param', type=float, default=1)
parser.add_argument('-augtype', help='augmentation type', type=str, default="weak+strong")
parser.add_argument('-augk', help='the number of augmentation', type=int, default=6)
parser.add_argument('-correct', help='hyper param', type=float, default=0.5)
args = parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def partial_loss(output, weight, eps=1e-12):
    weight = weight.clone().detach()
    l = weight * torch.log_softmax(output, dim=1)
    loss = (- torch.sum(l)) / l.size(0)
    return loss


def Dir_kl_loss(alpha, prior_alpha):
    KLD = torch.mvlgamma(alpha.sum(1), p=1) - torch.mvlgamma(alpha, p=1).sum(1) - torch.mvlgamma(prior_alpha.sum(1),
                                                                                                 p=1) + \
          torch.mvlgamma(prior_alpha, p=1).sum(1) + ((alpha - prior_alpha) * (
                torch.digamma(alpha) - torch.digamma(alpha.sum(dim=1, keepdim=True).expand_as(alpha)))).sum(1)
    return KLD.mean()

def generate_alpha(X, Y, scale=1, eps=1e-6):
    alpha = torch.exp(X / 4)
    alpha = F.hardtanh(alpha, min_val=1e-2, max_val=30)
    return alpha


def main():

    print(args)
    
    device = torch.device('cuda:' + args.gpu if torch.cuda.is_available() else 'cpu')

    set_seed(args.seed)

    if args.ds == "cifar10":
        partial_train_loader, valid_loader, test_loader, dim, K = load_cifar10(args.ds, batch_size=args.bs, device=device, args=args)
    if args.ds == "cifar100":
        partial_train_loader, valid_loader, test_loader, dim, K = load_cifar100(args.ds, batch_size=args.bs, device=device, args=args)
    if args.ds == "fmnist":
        partial_train_loader, valid_loader, test_loader, dim, K = load_fmnist(args.ds, batch_size=args.bs, device=device, args=args)
    if args.ds == "kmnist":
        partial_train_loader, valid_loader, test_loader, dim, K = load_kmnist(args.ds, batch_size=args.bs, device=device, args=args)
    if args.ds == "mnist":
        partial_train_loader, valid_loader, test_loader, dim, K = load_mnist(args.ds, batch_size=args.bs, device=device, args=args)

    # predictor
    if args.mo == 'mlp':
        model = mlp_model(input_dim=dim, output_dim=K)
    elif args.mo == 'linear':
        model = linear_model(input_dim=dim, output_dim=K)
    elif args.mo == 'lenet':
        model = LeNet(out_dim=K)
    elif args.mo == 'densenet':
        model = densenet(num_classes=K)
    elif args.mo == 'resnet':
        model = resnet(depth=32, num_classes=K)

    # information bottleneck
    if args.imo == 'mlp':
        enc_model2 = info_mlp_model(input_dim=K, output_dim=K)
    elif args.imo == 'linear':
        enc_model2 = linear_model(input_dim=K, output_dim=K)

    if args.wlo == 'proden':
        train_p_Y = torch.Tensor(partial_train_loader.dataset.given_label_matrix)
        warm_loss_fn = proden_loss(train_p_Y, device)
    elif args.wlo == 'rc':
        train_p_Y = torch.Tensor(partial_train_loader.dataset.given_label_matrix)
        warm_loss_fn = rc_loss(train_p_Y, device)

    # try:
    if True:
        model.to(device)

        warm_optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)

        save_path = "checkpoints_baseline/{}/{}/".format(args.ds, args.lo)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        early = EarlyStopping(patience=50,
                              path=os.path.join(save_path, "{}_lo={}_seed={}.pt".format(args.ds, args.lo, args.seed)))

        valid_acc = accuracy_check(loader=valid_loader, model=model, device=device)
        test_acc  = accuracy_check(loader=test_loader, model=model, device=device)
        print("Epoch {:>3d}, valid acc: {:.2f}, test acc: {:.2f}. ".format(0, valid_acc, test_acc))

        # warm_up
        for warm_epoch in range(args.wep):
            model.train()
            for i, (images, images_a_list, labels, true_labels, indexes) in enumerate(partial_train_loader):
                X, Y = images.to(device), labels.to(device)
                X_a_list = list(map(lambda x: x.to(device), images_a_list))
                outputs = model(X)
                outputs_a_list = list(map(lambda x: model(x), X_a_list))
                warm_loss = warm_loss_fn(outputs, indexes)

                if args.wlo in ['proden']:
                    warm_loss_list = list(map(lambda x: warm_loss_fn(x, indexes), outputs_a_list))
                    warm_loss += sum(warm_loss_list)
                    warm_loss_fn.update_conf(outputs, indexes)

                if args.wlo in ['rc', 'cavl']:
                    warm_loss_list = list(map(lambda x: warm_loss_fn(x, indexes), outputs_a_list))
                    warm_loss += sum(warm_loss_list)
                warm_optimizer.zero_grad()
                warm_loss.backward()
                warm_optimizer.step()
            if args.wlo in ['rc', 'cavl']:
                warm_loss_fn.update_conf(model, X, Y, indexes)
            model.eval()
            valid_acc = accuracy_check(loader=valid_loader, model=model, device=device)
            test_acc = accuracy_check(loader=test_loader, model=model, device=device)
            print("After Warm Epoch {:>3d}, valid acc: {:.2f}, test acc: {:.2f}. ".format(warm_epoch, valid_acc, test_acc))

        # d_matrix, d_matrix_w, d_matrix_s = map(lambda x: copy.deepcopy(x), (warm_loss_fn.conf, warm_loss_fn.conf, warm_loss_fn.conf))
        # o_matrix, o_matrix_w, o_matrix_s = map(lambda x: copy.deepcopy(x), (warm_loss_fn.conf, warm_loss_fn.conf, warm_loss_fn.conf))
        d_matrix, o_matrix = map(lambda x: copy.deepcopy(x), (warm_loss_fn.conf, warm_loss_fn.conf))

        # training
        enc_model1 = copy.deepcopy(model)
        enc_model2 = enc_model2.to(device)

        scale_const = 10.0
        prior_alpha = torch.Tensor(1, K).fill_(scale_const).to(device)

        optimizer = torch.optim.SGD(list(model.parameters()) + list(enc_model1.parameters()) + list(enc_model2.parameters()),
                                    lr=args.lr, weight_decay=args.wd, momentum=0.9)
        best_val, best_test, best_epoch = -1, -1, -1

        for epoch in range(args.ep):
            model.train()
            enc_model1.train()
            enc_model2.train()
            for i, (images, images_a_list, labels, true_labels, indexes) in enumerate(partial_train_loader):
                
                X, Y = map(lambda x: x.to(device), (images, labels))
                X_a_list = list(map(lambda x: x.to(device), images_a_list))
                embed = enc_model1(X)
                embed_a_list = list(map(lambda x: enc_model1(x), X_a_list))

                outputs = model(X)
                outputs_a_list = list(map(lambda x: model(x), X_a_list))

                mean_d = F.softmax(embed / args.T_1, dim=1)
                mean_d_a_list = list(map(lambda x: F.softmax(x / args.T_1, dim=1), embed_a_list))

                _xi = F.softmax(outputs / args.T_1, dim=1)
                _xi_a_list = list(map(lambda x: F.softmax(x / args.T_1, dim=1), outputs_a_list))


                # print(torch.max(embed / args.T_1, dim=1)[0].size())
                # alpha = torch.exp(embed / args.T_1 - torch.max(embed / args.T_1, dim=1, keepdim=True)[0])
                # alpha_a_list = list(map(lambda x: torch.exp(x / args.T_1  - torch.max(x / args.T_1, dim=1, keepdim=True)[0]), embed_a_list))
                alpha = generate_alpha(embed, Y, scale_const)
                alpha_a_list = list(map(lambda x: generate_alpha(x, Y, scale_const), embed_a_list))

                d_machine = torch.distributions.Dirichlet(alpha)
                d_machine_a_list = list(map(lambda x: torch.distributions.Dirichlet(x), alpha_a_list))

                d = d_machine.rsample()
                d_a_list = list(map(lambda x: x.rsample(), d_machine_a_list))

                mu = torch.sigmoid(enc_model2(d))
                mu_a_list = list(map(lambda x: torch.sigmoid(enc_model2(x)), d_a_list))

                # loss 1
                loss1_1 = torch.nn.MSELoss()(mu, Y)
                loss1_1_a_list = list(map(lambda x: torch.nn.MSELoss()(x, Y), mu_a_list))
                loss1_1 += sum(loss1_1_a_list)
                loss1_1 = args.loss_w_1 * loss1_1

                loss1_2 = Dir_kl_loss(alpha, prior_alpha)
                loss1_2_a_list = list(map(lambda x: Dir_kl_loss(x, prior_alpha), alpha_a_list))  
                loss1_2 += sum(loss1_2_a_list) / args.augk        
                loss1_2 = args.beta_1 * loss1_2

                loss1 = loss1_1 + loss1_2

                # loss 2
                infoNCE_loss_fn = InfoNCE(negative_mode='unpaired')

                loss2_1_a_list = list(map(lambda x: infoNCE_loss_fn(d, x, d), d_a_list))
                loss2_1 = sum(loss2_1_a_list) / args.augk
                loss2_1 = args.loss_w_2 * loss2_1

                loss2_2 = Dir_kl_loss(alpha, torch.Tensor(1, K).fill_(1.0).to(device))
                loss2_2_a_list = list(map(lambda x: Dir_kl_loss(x, torch.Tensor(1, K).fill_(1.0).to(device)), alpha_a_list))
                loss2_2 += sum(loss2_2_a_list)
                loss2_2 = args.beta_2 * loss2_2
                
                loss2 = loss2_1 + loss2_2

                L_MI = loss1 + loss2

                L_o = partial_loss(embed, o_matrix[indexes,:])
                L_o_list = list(map(lambda x: partial_loss(x, o_matrix[indexes,:]), embed_a_list))
                L_o += sum(L_o_list)

                # classifier loss
                L_f = partial_loss(outputs, d_matrix[indexes,:])
                L_f_list = list(map(lambda x: partial_loss(x, d_matrix[indexes,:]), outputs_a_list))
                L_f += sum(L_f_list)

                # total loss

                loss = args.lambda_ * L_o + L_MI + L_f
                # loss = args.lambda_ * L_o + L_f
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                partial_d = (mean_d.clone().detach() * Y) / (mean_d.clone().detach() * Y).sum(dim=1, keepdim=True)
                xi = (_xi.clone().detach() * Y) / (_xi.clone().detach() * Y).sum(dim=1, keepdim=True)

                d_matrix[indexes, :] = args.correct * partial_d.clone().detach() + (1 - args.correct) * xi.clone().detach()
                o_matrix[indexes, :] = (1 - args.correct) * partial_d.clone().detach() + args.correct * xi.clone().detach()

                # print("Step {}, loss1_1: {}, loss1_2: {}, loss2_1: {}, loss2_2: {}.".format(i, loss1_1, loss1_2, loss2_1, loss2_2))
            model.eval()
            enc_model1.eval()
            enc_model2.eval()
            valid_acc = accuracy_check(loader=valid_loader, model=model, device=device)
            test_acc  = accuracy_check(loader=test_loader, model=model, device=device)
            valid_acc = accuracy_check(loader=valid_loader, model=enc_model1, device=device)
            test_acc  = accuracy_check(loader=test_loader, model=enc_model1, device=device)
            print("Epoch {:>3d}, valid acc: {:.2f}, test acc: {:.2f}. ".format(epoch + 1, valid_acc, test_acc))
            if epoch >= 1:
                early(valid_acc, model, epoch)
            if early.early_stop:
                break
            if valid_acc > best_val:
                best_val = valid_acc
                best_epoch = epoch
                best_test = test_acc

        print(args)
        print("Best Epoch {:>3d}, Best valid acc: {:.2f}, test acc: {:.2f}. ".format(best_epoch, best_val, best_test))
    return best_test


def objective(trial=None):
    if args.ds == "cifar10":
        args.seed     = 46
        args.lr       = 0.01
        args.wd       = 0.001
        args.mo       = "resnet"
        args.imo      = 'mlp'
        args.wlo      = 'proden'
        args.wep      = 10
        args.lambda_  = 1
        args.loss_w_1 = 1
        args.loss_w_2 = 1
        args.beta_1   = 0.001
        args.beta_2   = 0.001
        args.T_1      = 1
        args.T_2      = 1
        args.correct  = 0.2
    if args.ds == "cifar100":
        args.seed     = 46
        args.lr       = 0.01
        args.wd       = 0.001
        args.mo       = "resnet"
        args.imo      = 'mlp'
        args.wlo      = 'proden'
        args.wep      = 2
        args.lambda_  = 1
        args.loss_w_1 = 1
        args.loss_w_2 = 1
        args.beta_1   = 0.001
        args.beta_2   = 0.001
        args.T_1      = 1
        args.T_2      = 1
        args.correct  = 0.2
    if args.ds == "fmnist":
        args.seed     = 27
        args.lr       = 1e-2
        args.wd       = 1e-5
        args.mo       = "lenet"
        args.imo      = 'linear'
        args.wlo      = 'proden'
        args.wep      = 60
        args.lambda_  = 1
        args.loss_w_1 = 1e-2
        args.loss_w_2 = 1e-2
        args.beta_1   = 1e-3
        args.beta_2   = 1
        args.T_1      = 9.883653329972347
        args.T_2      = 1.5758341876406035
        args.correct  = 0.8
    if args.ds == "kmnist":
        args.seed     = 36
        args.lr       = 5e-2
        args.wd       = 1e-5
        args.mo       = "lenet"
        args.imo      = 'linear'
        args.wlo      = 'proden'
        args.wep      = 50
        args.lambda_  = 0.1
        args.loss_w_1 = 0.1
        args.loss_w_2 = 0.01
        args.beta_1   = 0.001
        args.beta_2   = 0.01
        args.T_1      = 2.409795021379017
        args.T_2      = 2.1381551141414
        args.correct  = 0.8
    best_test = main()

    return best_test


if __name__ == "__main__":
    objective(trial=None)