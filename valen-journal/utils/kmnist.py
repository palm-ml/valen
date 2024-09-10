import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from .randaugment import RandomAugment
import augment
from augment.autoaugment_extra import CIFAR10Policy
from augment.cutout import Cutout
from .utils_algo import generate_instance_dependent_candidate_labels

GENERATE_SEED=42
# GENERATE_SEED=40
def load_kmnist(ds, batch_size, device):
    test_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))])
    
    temp_train = dsets.KMNIST(root='/home/qiaocy/data/REDGE/', train=True, download=True, transform=transforms.ToTensor())
    temp_valid = dsets.KMNIST(root='/home/qiaocy/data/REDGE/', train=True, transform=test_transform)
    data_size = len(temp_train)
    train_dataset, _ = torch.utils.data.random_split(temp_train,
                                                                    [int(data_size * 0.9), data_size - int(data_size * 0.9)],
                                                                    torch.Generator().manual_seed(GENERATE_SEED))
    train_dataset_for_partial_labels, valid_dataset = torch.utils.data.random_split(temp_valid,
                                                                    [int(data_size * 0.9), data_size - int(data_size * 0.9)],
                                                                    torch.Generator().manual_seed(GENERATE_SEED))

    full_train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=False, num_workers=8)
    for data, targets in full_train_loader:
        traindata, trainlabels = data, targets.long()
    # get original data and labels
    # check 
    # full_valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=len(valid_dataset), shuffle=False, num_workers=8)
    # for data, targets in full_valid_loader:
    #     validdata, validlabels = data, targets.long()
    # full_temp_valid_loader = torch.utils.data.DataLoader(dataset=temp_valid_dataset, batch_size=len(temp_valid_dataset), shuffle=False, num_workers=8)
    # for data, targets in full_temp_valid_loader:
    #     tempvaliddata, tempvalidlabels = data, targets.long()
    
    # print(validlabels)
    # print(tempvalidlabels)
    # print((validlabels != tempvalidlabels).sum())
    train_loader_for_partial_labels = torch.utils.data.DataLoader(dataset=train_dataset_for_partial_labels, batch_size=batch_size*4, shuffle=False, num_workers=8)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=len(valid_dataset), shuffle=False, num_workers=8)
    test_dataset = dsets.KMNIST(root='/home/qiaocy/data/REDGE/', train=False, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=8)
    # set test dataloader
    
    partialY, avgC = generate_instance_dependent_candidate_labels(ds, train_loader_for_partial_labels, trainlabels, device)

    print('Average candidate num: ', avgC)
    partial_matrix_dataset = KMNIST_Augmentention(traindata, partialY.float(), trainlabels.float())
    # generate partial label dataset
    partial_matrix_train_loader = torch.utils.data.DataLoader(dataset=partial_matrix_dataset, 
                                                                batch_size=batch_size, 
                                                                shuffle=True, 
                                                                num_workers=8,
                                                                drop_last=True)
    dim = 28 * 28
    K = 10
    return partial_matrix_train_loader, valid_loader, test_loader, dim, K


class KMNIST_Augmentention(Dataset):
    def __init__(self, images, given_label_matrix, true_labels):
        self.images = images
        self.given_label_matrix = given_label_matrix
        # user-defined label (partial labels)
        self.true_labels = true_labels
        # PiCO augmentation
        # self.weak_transform = transforms.Compose(
        #     [
        #     transforms.ToPILImage(),
        #     transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomApply([
        #         transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        #     ], p=0.8),
        #     transforms.RandomGrayscale(p=0.2),
        #     transforms.ToTensor(), 
        #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        # self.strong_transform = transforms.Compose(
        #     [
        #     transforms.ToPILImage(),
        #     transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        #     transforms.RandomHorizontalFlip(),
        #     RandomAugment(3, 5),
        #     transforms.ToTensor(), 
        #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

        # PLCR
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(28, 4, padding_mode='reflect'),
            # transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        self.weak_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(28, 4, padding_mode='reflect'),
            transforms.ToTensor(),
            Cutout(n_holes=1, length=16),
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        self.strong_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(28, 4, padding_mode='reflect'),
            transforms.ToTensor(),
            Cutout(n_holes=1, length=16),
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])


    def __len__(self):
        return len(self.true_labels)
        
    def __getitem__(self, index):
        each_image_o = self.transform(self.images[index])
        each_image_w = self.weak_transform(self.images[index])
        each_image_s = self.strong_transform(self.images[index])
        each_label = self.given_label_matrix[index]
        each_true_label = self.true_labels[index]
        
        return each_image_o, each_image_w, each_image_s, each_label, each_true_label, index

