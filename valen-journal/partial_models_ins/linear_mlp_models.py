import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.init as init


class linear_model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(linear_model, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = x.view(-1, self.num_flat_features(x))
        out = self.linear(out)
        return 0, out

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# class mlp_model(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(mlp_model, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_dim, output_dim)

#     def forward(self, x):
#         out = x.view(-1, self.num_flat_features(x))
#         out = self.fc1(out)
#         out = self.relu1(out)
#         out = self.fc2(out)
#         return out

#     def num_flat_features(self, x):
#         size = x.size()[1:]
#         num_features = 1
#         for s in size:
#             num_features *= s
#         return num_features

class mlp_model(nn.Module):
    def __init__(self, input_dim, output_dim, parameter_momentum=0.1):
        super(mlp_model, self).__init__()

        self.L1 = nn.Linear(input_dim, 300, bias=False)
        init.xavier_uniform_(self.L1.weight)
        self.bn1 = nn.BatchNorm1d(300, momentum=parameter_momentum)
        init.ones_(self.bn1.weight)

        self.L2 = nn.Linear(300, 301, bias=False)
        init.xavier_uniform_(self.L2.weight)
        self.bn2 = nn.BatchNorm1d(301, momentum=parameter_momentum)
        init.ones_(self.bn2.weight)

        self.L3 = nn.Linear(301, 302, bias=False)
        init.xavier_uniform_(self.L3.weight)
        self.bn3 = nn.BatchNorm1d(302, momentum=parameter_momentum)
        init.ones_(self.bn3.weight)

        self.L4 = nn.Linear(302, 303, bias=False)
        init.xavier_uniform_(self.L4.weight)
        self.bn4 = nn.BatchNorm1d(303, momentum=parameter_momentum)
        init.ones_(self.bn4.weight)

        self.L5 = nn.Linear(303, output_dim, bias=True)
        init.xavier_uniform_(self.L5.weight)
        init.zeros_(self.L5.bias)

    def forward(self, x):
        x = x.view(-1, self.num_flat_features(x))
        x = self.L1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.L2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.L3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.L4(x)
        x = self.bn4(x)
        x = F.relu(x)

        x = self.L5(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class LeNet(nn.Module):
    def __init__(self, output_dim):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_dim)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, (2, 2))
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, (2, 2))
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


def save_model(model, dir):
    torch.save(model.state_dict(), dir + ".pkl")
    print("Model saved in {}.pkl".format(dir))


def load_pretrain(model, path):
    model.load_state_dict(torch.load(path))
    print("Model loaded in {}".format(path))
    return model
