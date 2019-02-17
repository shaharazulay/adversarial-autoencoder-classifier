import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

from _train_utils import predict_labels


def show_predicted_labels(Q, P, valid_loader):
    batch_size = valid_loader.batch_size
    labels = []

    for _, (X, y) in enumerate(valid_loader):
        X.resize_(batch_size, Q.input_size)

        X, y = Variable(X), Variable(y)
        if cuda:
            X, y = X.cuda(), y.cuda()
        y_pred = predict_labels(Q, X)

        latent_y, latent_z = Q(X)
        #print latent_y, latent_z

        labels.extend(y_pred.numpy())

        # show_reconstruction(Q, P, X)
        # plt.show()

    plt.figure()
    plt.hist(labels, bins=10)
    plt.show()

    plt.figure()
    plt.hist(latent_z.detach().numpy()[:, 0])
    plt.show()

    from collections import Counter
    print Counter(labels)
        # plt.figure()
        # plt.imshow(X[:784].resize_(28 ,28), cmap='gray')
        # plt.title('Orig: %s, Pred: %s' % (target[0], pred[0]))
        # plt.show()


def show_reconstruction(Q, P, X):
    Q.eval()
    P.eval()

    latent_y, latent_z = Q(X)

    latent_vec = torch.cat((latent_y, latent_z), 1)
    X_rec = P(latent_vec)

    img_orig = np.array(X[0].data.tolist()).reshape(28, 28)
    img_rec = np.array(X_rec[0].data.tolist()).reshape(28, 28)
    plt.subplot(1, 2, 1)
    plt.imshow(img_orig, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(img_rec, cmap='gray')


def generate_digits(P, label, n_classes=10, z_dim=2):
    P.eval()

    latent_y = np.eye(n_classes)[label].astype('float32')
    latent_y = Variable(torch.from_numpy(latent_y).resize_(1, n_classes))

    while True:
        latent_z = Variable(torch.randn(1, z_dim))
        latent_vec = torch.cat((latent_y, latent_z), 1)

        X_rec = P(latent_vec)
        plt.imshow(np.array(X_rec[0].data.tolist()).reshape(28, 28), cmap='gray')
        plt.show()

import os
from _model import Q_net, P_net
Q = Q_net().load(os.path.join('../data', 'encoder'))
P = P_net().load(os.path.join('../data', 'decoder'))

import _data_utils
cuda = torch.cuda.is_available()
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
train_labeled_loader, train_unlabeled_loader, valid_loader = _data_utils.load_data(
    data_path='../data', batch_size=100, **kwargs)
#show_predicted_labels(Q, P, valid_loader)
generate_digits(P, label=4)
