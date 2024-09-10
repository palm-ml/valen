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
        self.gcn1 = GraphConv2(n_hid, n_out, activation=F.elu, adj=adj)
        self.gcn2 = GraphConv2(n_out, n_out, activation=F.elu, adj=adj)
        self.gcn3 = GraphConv2(n_out, n_out * 2, activation=lambda x: x, adj=adj)

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
        super(Encoder, self).__init__()
        self.n_out = n_out
        self.base_gcn = GraphConv2(n_in, n_hid, adj=adj)
        self.gcn1 = GraphConv2(n_hid, n_out, activation=F.elu, adj=adj)
        self.gcn2 = GraphConv2(n_out, n_out, activation=F.elu, adj=adj)
        self.gcn3 = GraphConv2(n_out, n_out, activation=lambda x: x, adj=adj)

    def forward(self, x):
        hidden = self.base_gcn(x)
        out = self.gcn1(hidden)
        out = self.gcn2(out)
        out = self.gcn3(out)
        alpha = torch.exp(out / 4)
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
        alpha = torch.exp(out / 4)
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
        super(VGAE_Decoder, self).__init__()
        self.n_label = n_label
        self.layer1 = nn.Sequential(nn.Linear(n_in, n_hid),
                                    nn.Tanh(),
                                    nn.Dropout(1 - keep_prob))
        self.layer2 = nn.Sequential(nn.Linear(n_hid, n_hid),
                                    nn.ELU(),
                                    nn.Dropout(1 - keep_prob))
        self.fc_out = nn.Linear(n_hid, n_out)
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


# Decoder for L and A
class Decoder_L(nn.Module):
    def __init__(self, num_classes, hidden_dim, keep_prob=1.0):
        super(Decoder_L, self).__init__()
        # self.n_label = n_label
        # self.layer1 = nn.Sequential(nn.Linear(num_classes, hidden_dim),
        #                             nn.Tanh(),
        #                             nn.Dropout(1-keep_prob))
        # self.layer2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
        #                             nn.ELU(),
        #                             nn.Dropout(1-keep_prob))
        self.layer3 = nn.Sequential(nn.Linear(num_classes, hidden_dim),
                                    nn.Tanh(),
                                    nn.Dropout(1 - keep_prob))
        self.layer4 = nn.Sequential(nn.Linear(hidden_dim, num_classes),
                                    nn.ELU(),
                                    nn.Dropout(1 - keep_prob))
        self.sigmoid = nn.Sigmoid()
        # self.fc_out = nn.Linear(hidden_dim, num_classes)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.01)

    def forward(self, d):
        # h0 = self.layer1(d)
        # h1 = self.layer2(h0)
        # features_hat = self.fc_out(h1)
        h2 = self.layer3(d)
        h3 = self.layer4(h2)
        labels_hat = self.sigmoid(h3)
        # labels_hat = h3
        # adj_hat = dot_product_decode(d)
        # return features_hat, labels_hat, adj_hat
        return labels_hat


class GraphConv(nn.Module):
    def __init__(self, n_in, n_out, adj, activation=F.relu, **kwargs):
        super(GraphConv, self).__init__(**kwargs)
        self.weight = glorot_init(n_in, n_out)
        self.adj = adj
        self.activation = activation

    def forward(self, inputs):
        x = inputs
        x = torch.mm(x, self.weight)
        x = torch.mm(self.adj, x)
        outputs = self.activation(x)
        return outputs


class GraphConv2(nn.Module):
    def __init__(self, n_in, n_out, activation=F.relu, adj=None, **kwargs):
        super(GraphConv2, self).__init__(**kwargs)
        self.weight = glorot_init(n_in, n_out)
        self.adj = adj
        self.activation = activation

    def forward(self, inputs):
        x = inputs
        x = torch.mm(x, self.weight)
        x = torch.spmm(self.adj, x)
        outputs = self.activation(x)
        return outputs


def dot_product_decode(Z):
    A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
    return A_pred


def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return nn.Parameter(initial)


# enc for cifar
class CONV_Encoder(nn.Module):
    """
    Encoder: D and \phi --> z
    """

    def __init__(self, in_channels=3, feature_dim=32, num_classes=2, hidden_dims=[32, 64, 128, 256], z_dim=128):
        super().__init__()
        self.z_dim = z_dim
        self.feature_dim = feature_dim
        self.embed_class = nn.Linear(num_classes, feature_dim * feature_dim)
        self.embed_data = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.in_channels = in_channels
        in_channels += 1
        modules = []

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, z_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1] * 4, z_dim)

    def forward(self, x, partial_label):
        embedded_class = self.embed_class(partial_label)
        x = x.view(x.size(0), self.in_channels, self.feature_dim, self.feature_dim)
        embedded_class = embedded_class.view(-1, self.feature_dim, self.feature_dim).unsqueeze(1)
        embedded_input = self.embed_data(x)
        x = torch.cat([embedded_input, embedded_class], dim=1)
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)

        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)
        return mu, log_var


# dec for cifar
class CONV_Decoder(nn.Module):
    """
    Decoder: z, d --> \phi
    """

    def __init__(self, num_classes=10, hidden_dims=[256, 128, 64, 32], z_dim=128):
        super().__init__()
        self.decoder_input = nn.Linear(z_dim + num_classes, hidden_dims[0] * 4)
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3,
                      kernel_size=3, stride=1, padding=1),
            nn.Sigmoid())

    def forward(self, z, d):
        out = torch.cat((z, d), dim=1)
        out = self.decoder_input(out)
        out = out.view(-1, 256, 2, 2) #cub200 2048
        out = self.decoder(out)
        out = self.final_layer(out)
        return out


# enc for mnist
class CONV_Encoder_MNIST(nn.Module):
    def __init__(self, in_channels=1, feature_dim=28, num_classes=10, hidden_dims=[32, 64, 128, 256], z_dim=2):
        super().__init__()
        self.z_dim = z_dim
        self.feature_dim = feature_dim
        self.embed_class = nn.Linear(num_classes, feature_dim * feature_dim)
        self.embed_data = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        in_channels += 1
        modules = []

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, z_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1] * 4, z_dim)

    def forward(self, x, partial_label):
        embedded_class = self.embed_class(partial_label)
        x = x.view(x.size(0), 1, self.feature_dim, self.feature_dim)
        embedded_class = embedded_class.view(-1, self.feature_dim, self.feature_dim).unsqueeze(1)
        embedded_input = self.embed_data(x)

        x = torch.cat([embedded_input, embedded_class], dim=1)
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)
        return mu, log_var


# dec for mnist
class CONV_Decoder_MNIST(nn.Module):

    def __init__(self, num_classes=2, hidden_dims=[256, 128, 64, 32], z_dim=1):
        super().__init__()
        self.decoder_input = nn.Linear(z_dim + num_classes, hidden_dims[0] * 4)
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               ),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=1,
                      kernel_size=4),
            nn.Sigmoid())

    def forward(self, z, d):
        out = torch.cat((z, d), dim=1)
        out = self.decoder_input(out)
        out = out.view(-1, 256, 2, 2)
        out = self.decoder(out)
        out = self.final_layer(out)
        out = out.view(out.size(0), -1)
        return out


def make_hidden_layers(num_hidden_layers=1, hidden_size=5, prefix="y"):
    block = nn.Sequential()
    for i in range(num_hidden_layers):
        block.add_module(prefix + "_" + str(i),
                         nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU()))
    return block


# class Y_Encoder(nn.Module):
#     def __init__(self, feature_dim = 2, num_classes = 2, num_hidden_layers=1, hidden_size = 5):
#         super().__init__()
#         self.y_fc1 = nn.Linear(feature_dim, hidden_size)
#         self.y_h_layers = make_hidden_layers(num_hidden_layers, hidden_size=hidden_size, prefix="y")
#         self.y_fc2 = nn.Linear(hidden_size, num_classes)
#
#     def forward(self, x):
#         out = F.relu(self.y_fc1(x))
#         out = self.y_h_layers(out)
#         c_logits = self.y_fc2(out)
#         return c_logits


# Encoder/Decoder for realworld dataset
class Z_Encoder(nn.Module):
    def __init__(self, feature_dim=2, num_classes=2, num_hidden_layers=1, hidden_size=5, z_dim=2):
        super().__init__()
        self.z_fc1 = nn.Linear(feature_dim + num_classes, hidden_size)
        self.z_h_layers = make_hidden_layers(num_hidden_layers, hidden_size=hidden_size, prefix="z")
        self.z_fc_mu = nn.Linear(hidden_size, z_dim)  # fc21 for mean of Z
        self.z_fc_logvar = nn.Linear(hidden_size, z_dim)  # fc22 for log variance of Z

    def forward(self, x, y_hat):
        out = torch.cat((x, y_hat), dim=1)
        out = F.relu(self.z_fc1(out))
        out = self.z_h_layers(out)
        mu = F.elu(self.z_fc_mu(out))
        logvar = F.elu(self.z_fc_logvar(out))
        return mu, logvar


class X_Decoder(nn.Module):
    def __init__(self, feature_dim=2, num_classes=2, num_hidden_layers=1, hidden_size=5, z_dim=1):
        super().__init__()
        self.recon_fc1 = nn.Linear(z_dim + num_classes, hidden_size)
        self.recon_h_layers = make_hidden_layers(num_hidden_layers, hidden_size=hidden_size, prefix="recon")
        self.recon_fc2 = nn.Linear(hidden_size, feature_dim)

    def forward(self, z, y_hat):
        out = torch.cat((z, y_hat), dim=1)
        out = F.relu(self.recon_fc1(out))
        out = self.recon_h_layers(out)
        x = self.recon_fc2(out)
        return x
