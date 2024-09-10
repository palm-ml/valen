import argparse

def extract_args():
    # main setting
    parser = argparse.ArgumentParser(
        prog='VALEN demo file.',
        usage='Demo with partial labels.',
        description='Various algorithms of VALEN.',
        epilog='end',
        add_help=True
    )
    # optional args
    parser.add_argument('--lr', help='optimizer\'s learning rate', type=float, default=1e-2)
    parser.add_argument('--wd', help='weight decay', type=float, default=1e-4)
    parser.add_argument('--bs', help='batch size', type=int, default=256)
    parser.add_argument('--ep', help='number of epochs', type=int, default=500)
    parser.add_argument('--dt', help='type of the dataset', type=str, choices=['benchmark', 'realworld', 'uci'])
    parser.add_argument('--ds', help='specify a dataset', type=str)
    parser.add_argument('--warm_up', help='number of warm-up epochs', type=int, default=10)
    parser.add_argument('--knn', help='number of knn neighbours', type=int, default=3)
    parser.add_argument('--partial_type', help='flipping strategy', type=str, default='feature', choices=['feature', 'random'])
    parser.add_argument('--partial_rate', help='flipping rate', type=float, default=0.4)
    parser.add_argument('--dir', help='result save path', type=str, default='results/', required=False)
    parser.add_argument('--sn', help='result save file name', type=str, default='newete_kmnist3.log', required=False)
    parser.add_argument('--sampling', help='the sampling times of Dirichlet', type=int, default=1, required=False)
    # loss paramters
    parser.add_argument('--alpha','-alpha', type=float, default=1, help = 'balance parameter of the loss function (default=1.0)')
    parser.add_argument('--beta','-beta', type=float, default=1,help = 'balance parameter of the loss function (default=1.0)')
    parser.add_argument('--lambda','-lambda', type=float, default=1, help = 'balance parameter of the loss function (default=1.0)')
    parser.add_argument('--gamma','-gamma', type=float, default=1, help = 'balance parameter of the loss function (default=1.0)')
    parser.add_argument('--theta','-theta', type=float, default=1, help = 'balance parameter of the loss function (default=1.0)')
    parser.add_argument('--sigma','-sigma', type=float, default=1, help = 'balance parameter of the loss function (default=1.0)')
    parser.add_argument('--correct','-correct', type=float, default=1, help = 'balance parameter of the loss function (default=1.0)')
    # model args
    parser.add_argument('--lo', type=str, default='valen', required=False)
    parser.add_argument('--mo', type=str, default='resnet', required=False)
    parser.add_argument('--seed', help='random seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    return args