from copy import deepcopy
from models.linear import linear
from models.mlp import mlp_feature, mlp_phi
from models.VGAE import VAE_Bernulli_Decoder
from models.resnet import resnet

def create_model(config, **args):
    if config.dt == "benchmark":
        if config.ds in ['mnist', 'kmnist', 'fmnist']:
            if config.partial_type == "random":
                net = mlp_feature(args['num_features'], args['num_features'], args['num_classes'])
            if config.partial_type == "feature":
                net = mlp_phi(args['num_features'], args['num_classes'])
        if config.ds in ['cifar10']:
            net = resnet(depth=32, n_outputs = args['num_classes'])
        enc = deepcopy(net)
        dec = VAE_Bernulli_Decoder(args['num_classes'], args['num_features'], args['num_features'])
        return net, enc, dec
    if config.dt == "realworld":
        net = linear(args['num_features'],args['num_classes'])
        enc = deepcopy(net)
        dec = VAE_Bernulli_Decoder(args['num_classes'], args['num_features'], args['num_features'])
        return net, enc, dec
        
    


