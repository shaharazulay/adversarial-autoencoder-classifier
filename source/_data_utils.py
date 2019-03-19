import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

def min_max_normalization(tensor, min_value, max_value):
    min_tensor = tensor.min()
    tensor = (tensor - min_tensor)
    max_tensor = tensor.max()
    tensor = tensor / max_tensor
    tensor = tensor * (max_value - min_value) + min_value
    return tensor

default_transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Lambda(lambda tensor:min_max_normalization(tensor, 0, 1))
    #transforms.Normalize((0.1307,), (0.3081,))
    ])


class MNISTSlice(MNIST):

    def __init__(
            self, root, data, labels, train=True, transform=default_transform, target_transform=None):

        super(MNISTSlice, self).__init__(
            root, train, transform, target_transform, download=True)

        data_ = data.clone()
        labels_ = labels.clone()

        if train:
            self.train_data = data_
            self.train_labels = labels_
        else:
            self.test_data = data_
            self.test_labels = labels_

    def dump(self, path):
        pickle.dump(self, open(path, 'wb'))

    @staticmethod
    def load(path):
        return pickle.load(open(path, 'rb'))


def load_data(data_path, batch_size, **kwargs):
    print('loading data started...')

    trainset_labeled = MNISTSlice.load(os.path.join(data_path, 'train_labeled.p'))
    trainset_unlabeled = MNISTSlice.load(os.path.join(data_path, 'train_unlabeled.p'))
    validset = MNISTSlice.load(os.path.join(data_path, 'validation.p'))

    train_labeled_loader = DataLoader(
        trainset_labeled,
        batch_size=batch_size,
        shuffle=True,
        **kwargs)

    train_unlabeled_loader = DataLoader(
        trainset_unlabeled,
        batch_size=batch_size,
        shuffle=True,
        **kwargs)

    valid_loader = DataLoader(validset, batch_size=batch_size, shuffle=True)

    print("dataset size in use: %s [labeled trainset]  %s [un-labeled trainset]  %s [validation]"    % (
        len(trainset_labeled), len(trainset_unlabeled), len(validset)))
    return train_labeled_loader, train_unlabeled_loader, valid_loader


def init_datasets(data_path):
    trainset_full = MNIST(
        data_path, train=True, download=True, transform=default_transform)

    train_label_index = []
    valid_label_index = []
    for i in range(10):
        train_label_list = trainset_full.train_labels.numpy()
        label_index = np.where(train_label_list == i)[0]
        label_subindex = list(label_index[:300])
        valid_subindex = list(label_index[300: 1000 + 300])
        train_label_index += label_subindex
        valid_label_index += valid_subindex

    trainset_np = trainset_full.train_data.numpy()
    trainset_label_np = trainset_full.train_labels.numpy()
    train_data_sub = torch.from_numpy(trainset_np[train_label_index])
    train_labels_sub = torch.from_numpy(trainset_label_np[train_label_index])

    trainset = MNISTSlice(
        root=data_path, data=train_data_sub, labels=train_labels_sub, train=True)
    trainset.dump(os.path.join(data_path, 'train_labeled.p'))

    validset_np = trainset_full.train_data.numpy()
    validset_label_np = trainset_full.train_labels.numpy()
    valid_data_sub = torch.from_numpy(validset_np[valid_label_index])
    valid_labels_sub = torch.from_numpy(validset_label_np[valid_label_index])

    validset = MNISTSlice(
        root=data_path, data=valid_data_sub, labels=valid_labels_sub, train=False)
    validset.dump(os.path.join(data_path, 'validation.p'))

    train_unlabel_index = []
    for i in range(60000):
        if i in train_label_index or i in valid_label_index:
            pass
        else:
            train_unlabel_index.append(i)

    trainset_np = trainset_full.train_data.numpy()
    trainset_label_np = trainset_full.train_labels.numpy()
    train_data_sub_unl = torch.from_numpy(trainset_np[train_unlabel_index])
    train_labels_sub_unl = torch.from_numpy(np.array([-1] * len(train_unlabel_index)))

    trainset_unl = MNISTSlice(
        root=data_path, data=train_data_sub_unl, labels=train_labels_sub_unl, train=True)
    trainset_unl.dump(os.path.join(data_path, 'train_unlabeled.p'))
