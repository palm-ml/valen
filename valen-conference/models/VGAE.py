from numpy.core.numeric import outer
import torch 
from torch import log, mean, nn
import torch.nn.functional as F 
import numpy as np

class VGAE_Encoder(nn.Module):
    def __init__(self, n_in, n_hid, n_out, adj=None):
        super(VGAE_Encoder, self).__init__()
        self.n_out = n_out
        self.base_gcn = GraphConv2(n_in, n_hid, adj=adj)
        self.gcn1  = GraphConv2(n_hid, n_out, activation=F.elu, adj=adj)
        self.gcn2 =  GraphConv2(n_out, n_out, activation=F.elu, adj=adj)
        self.gcn3 =  GraphConv2(n_out, n_out*2, activation=lambda x:x, adj=adj)
    

    def forward(self, x):
        hidden = self.base_gcn(x)
        out = self.gcn1(hidden)
        out = self.gcn2(out)
        out = self.gcn3(out)
        mean = out[:, :self.n_out]
        std = out[:, self.n_out:]
        return mean, std


    def set_gcn_adj(self, adj):
        self.base_gcn.adj = adj
        self.gcn1.adj = adj
        self.gcn2.adj = adj
        self.gcn3.adj = adj

class VAE_Encoder(nn.Module):
    def __init__(self, n_in, n_hidden, n_out, keep_prob=1.0) -> None:
        super(VAE_Encoder, self).__init__()
        self.n_out = n_out
        self.layer1 = nn.Linear(n_in, n_hidden)
        self.layer2 = nn.Linear(n_hidden, n_out)
        self.layer3 = nn.Linear(n_hidden, n_out)
        self._init_weight()
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.01)
    
    def forward(self, inputs):
        h0 = self.layer1(inputs)
        h0 = F.relu(h0)
        mean = self.layer2(h0)
        logvar = self.layer3(h0)
        # logvar = F.hardtanh(logvar, min_val=0, max_val=30)
        return mean, logvar

class VAE_Bernulli_Decoder(nn.Module):
    def __init__(self, n_in, n_hidden, n_out, keep_prob=1.0) -> None:
        super(VAE_Bernulli_Decoder, self).__init__()
        self.layer1 = nn.Linear(n_in, n_hidden)
        self.layer2 = nn.Linear(n_hidden, n_out)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.01)

    def forward(self, inputs):
        h0 = self.layer1(inputs)
        h0 = F.relu(h0)
        x_hat = self.layer2(h0)
        return x_hat


class Encoder(nn.Module):
    def __init__(self, n_in, n_hid, n_out, adj=None):
        super(Encoder,self).__init__()
        self.n_out = n_out
        self.base_gcn = GraphConv2(n_in, n_hid, adj=adj)
        self.gcn1  = GraphConv2(n_hid, n_out, activation=F.elu, adj=adj)
        self.gcn2 =  GraphConv2(n_out, n_out, activation=F.elu, adj=adj)
        self.gcn3 =  GraphConv2(n_out, n_out, activation=lambda x:x, adj=adj)

    def forward(self, x):
        hidden = self.base_gcn(x)
        out = self.gcn1(hidden)
        out = self.gcn2(out)
        out = self.gcn3(out)
        alpha = torch.exp(out/4)
        alpha = F.hardtanh(alpha, min_val=1e-2, max_val=30)
        return alpha

    def set_gcn_adj(self, adj):
        self.base_gcn.adj = adj
        self.gcn1.adj = adj
        self.gcn2.adj = adj
        self.gcn3.adj = adj


class Encoder2(nn.Module):
    def __init__(self, n_in, n_hidden, n_out, keep_prob=1.0) -> None:
        super(Encoder2, self).__init__()
        self.n_out = n_out
        self.layer1 = nn.Linear(n_in, n_hidden)
        self.layer2 = nn.Linear(n_hidden, n_out)
        self._init_weight()
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.01)
    
    def forward(self, inputs):
        h0 = self.layer1(inputs)
        h0 = F.relu(h0)
        out = self.layer2(h0)
        alpha = torch.exp(out/4)
        alpha = F.hardtanh(alpha, min_val=1e-2, max_val=30)
        return alpha


class Encoder3(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, keep_prob=1.0) -> None:
        super(Encoder3, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        # self._init_weight()
    
    # def _init_weight(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.xavier_normal_(m.weight.data)
    #             m.bias.data.fill_(0.01)
    
    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        # out = torch.exp(out/4)
        # out = F.hardtanh(out, min_val=1e-2, max_val=30)
        return out



class VGAE_Decoder(nn.Module):
    def __init__(self, n_in, n_hid, n_out, n_label, keep_prob=1.0):
        super(VGAE_Decoder,self).__init__()
        self.n_label = n_label
        self.layer1 = nn.Sequential(nn.Linear(n_in,n_hid),
                nn.Tanh(),
                nn.Dropout(1-keep_prob))
        self.layer2 = nn.Sequential(nn.Linear(n_hid,n_hid),
                nn.ELU(),
                nn.Dropout(1-keep_prob))
        self.fc_out = nn.Linear(n_hid,n_out)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.01)

    def forward(self, z):
        h0 = self.layer1(z)
        h1 = self.layer2(h0)
        features_hat = self.fc_out(h1)
        labels_hat = z[:, -self.n_label:]
        adj_hat = dot_product_decode(z)
        return features_hat, labels_hat, adj_hat

class Decoder(nn.Module):
    def __init__(self, n_in, n_hid, n_out, n_label, keep_prob=1.0):
        super(Decoder,self).__init__()
        self.n_label = n_label
        self.layer1 = nn.Sequential(nn.Linear(n_in,n_hid),
                nn.Tanh(),
                nn.Dropout(1-keep_prob))
        self.layer2 = nn.Sequential(nn.Linear(n_hid,n_hid),
                nn.ELU(),
                nn.Dropout(1-keep_prob))
        self.layer3 = nn.Sequential(nn.Linear(n_in,n_hid),
                nn.Tanh(),
                nn.Dropout(1-keep_prob))
        self.layer4 = nn.Sequential(nn.Linear(n_hid,n_label),
                nn.ELU(),
                nn.Dropout(1-keep_prob))
        self.fc_out = nn.Linear(n_hid,n_out)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.01)

    def forward(self, z):
        h0 = self.layer1(z)
        h1 = self.layer2(h0)
        features_hat = self.fc_out(h1)
        h2 = self.layer3(z)
        h3 = self.layer4(h2)
        labels_hat = h3
        adj_hat = dot_product_decode(z)
        return features_hat, labels_hat, adj_hat


class GraphConv(nn.Module):
    def __init__(self, n_in, n_out, adj, activation = F.relu, **kwargs):
        super(GraphConv, self).__init__(**kwargs)
        self.weight = glorot_init(n_in, n_out) 
        self.adj = adj
        self.activation = activation

    def forward(self, inputs):
        x = inputs
        x = torch.mm(x,self.weight)
        x = torch.mm(self.adj, x)
        outputs = self.activation(x)
        return outputs


class GraphConv2(nn.Module):
    def __init__(self, n_in, n_out, activation = F.relu, adj=None, **kwargs):
        super(GraphConv2, self).__init__(**kwargs)
        self.weight = glorot_init(n_in, n_out) 
        self.adj = adj
        self.activation = activation

    def forward(self, inputs):
        x = inputs
        x = torch.mm(x,self.weight)
        x = torch.spmm(self.adj, x)
        outputs = self.activation(x)
        return outputs

def dot_product_decode(Z):
    A_pred = torch.sigmoid(torch.matmul(Z,Z.t()))
    return A_pred


def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0/(input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
    return nn.Parameter(initial)