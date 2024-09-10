import torch 
from torch import nn
import torch.nn.functional as F 
import torch.nn.init as init
import numpy as np


class mlp(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(mlp, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        return out


class mlp_feature(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(mlp_feature, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        feature = self.relu3(out)
        out = self.fc4(out)
        return feature, out


class mlp_phi(nn.Module):
    def __init__(self, n_inputs, n_outputs, parameter_momentum=0.1):
        super(mlp_phi, self).__init__()

        self.L1 = nn.Linear(n_inputs, 300, bias=False)
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

        self.L5 = nn.Linear(303, n_outputs, bias=True)
        init.xavier_uniform_(self.L5.weight)
        init.zeros_(self.L5.bias)
        
    def forward(self, x):
        x = self.L1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.L2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.L3(x)
        x = self.bn3(x)
        x = F.relu(x)

        l = self.L4(x)
        x = self.bn4(l)
        x = F.relu(x)

        x = self.L5(x)
        return l, x


class mlp2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(mlp, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        phi = self.relu2(out)
        out = self.fc3(phi)
        out = self.relu3(out)
        out = self.fc4(out)
        return phi, out


class mlp3(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(mlp3, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        
        return out


class mlp_beta(nn.Module):
    def __init__(self, n_inputs, n_outputs, parameter_momentum=0.1):
        super(mlp_beta, self).__init__()
        self.L1 = nn.Linear(n_inputs, 300, bias=False)
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

        self.L5 = nn.Linear(303, n_outputs, bias=True)
        init.xavier_uniform_(self.L5.weight)
        init.zeros_(self.L5.bias)
        self.L6 = nn.Linear(303, n_outputs, bias=True)
        init.xavier_uniform_(self.L6.weight)
        init.zeros_(self.L5.bias)
        
    def forward(self, x):
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

        alpha = torch.exp(self.L5(x)) 
        beta = torch.exp(self.L6(x)) 
        y = alpha/(alpha + beta)
        return alpha, beta, y