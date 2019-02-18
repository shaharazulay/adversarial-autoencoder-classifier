import numpy as np
import torch
from torch.autograd import Variable

from matplotlib import gridspec
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

        #show_reconstruction(Q, P, X)
        #plt.show()

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

    print X[0], X_rec[0]
    img_orig = np.array(X[0].data.tolist()).reshape(28, 28)
    img_rec = np.array(X_rec[0].data.tolist()).reshape(28, 28)
    plt.subplot(1, 2, 1)
    plt.imshow(img_orig, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(img_rec, cmap='gray')
    plt.title('predicted label: %s' % torch.argmax(latent_y, dim=1))


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


def grid_plot(Q, P, data_loader, params):
    Q.eval()
    P.eval()
    X = get_X_batch(data_loader, params, size=10)
    _, z_g = Q(X)

    n_classes = params['n_classes']
    cuda = params['cuda']
    z_dim = params['z_dim']

    z_cat = np.arange(0, n_classes)
    z_cat = np.eye(n_classes)[z_cat].astype('float32')
    z_cat = torch.from_numpy(z_cat)
    z_cat = Variable(z_cat)
    if cuda:
        z_cat = z_cat.cuda()

    nx, ny = 5, n_classes
    plt.subplot()
    gs = gridspec.GridSpec(nx, ny, hspace=0.05, wspace=0.05)

    for i, g in enumerate(gs):
        z_gauss = z_g[i / ny].resize(1, z_dim)
        z_gauss0 = z_g[i / ny].resize(1, z_dim)

        for _ in range(n_classes - 1):
            z_gauss = torch.cat((z_gauss, z_gauss0), 0)

        z = torch.cat((z_cat, z_gauss), 1)
        x = P(z)

        ax = plt.subplot(g)
        img = np.array(x[i % ny].data.tolist()).reshape(28, 28)
        ax.imshow(img, )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('auto')


def grid_plot2d(Q, P, data_loader):
    Q.eval()
    P.eval()

    cuda = False

    z1 = Variable(torch.from_numpy(np.arange(-10, 10, 1.5).astype('float32')))
    z2 = Variable(torch.from_numpy(np.arange(-10, 10, 1.5).astype('float32')))
    if cuda:
        z1, z2 = z1.cuda(), z2.cuda()

    nx, ny = len(z1), len(z2)
    plt.subplot()
    gs = gridspec.GridSpec(nx, ny, hspace=0.05, wspace=0.05)

    for i, g in enumerate(gs):
        z = torch.cat((z1[i / ny], z2[i % nx])).resize(1, 2)
        x = P(z)

        ax = plt.subplot(g)
        img = np.array(x.data.tolist()).reshape(28, 28)
        ax.imshow(img, )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('auto')

import os
from _model import Q_net, P_net
Q = Q_net().load(os.path.join('../data', 'encoder'), z_size=5, n_classes=10)
P = P_net(z_size=5, n_classes=10).load(os.path.join('../data', 'decoder'), z_size=5, n_classes=10)

import _data_utils
cuda = torch.cuda.is_available()
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
train_labeled_loader, train_unlabeled_loader, valid_loader = _data_utils.load_data(
    data_path='../data', batch_size=100, **kwargs)
#show_predicted_labels(Q, P, valid_loader)
grid_plot2d(Q, P, valid_loader)
#generate_digits(P, label=4)
