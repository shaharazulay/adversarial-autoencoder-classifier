import pickle
from torchvision.datasets import MNIST
from torchvision import transforms


default_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
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
