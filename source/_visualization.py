import numpy as np
import torch
from torch.autograd import Variable

from matplotlib import gridspec
import matplotlib.pyplot as plt

from _train_utils import predict_labels


def show_predicted_labels(Q, P, valid_loader, n_classes=10):
    batch_size = valid_loader.batch_size
    labels = []

    for _, (X, y) in enumerate(valid_loader):

        X.resize_(batch_size, Q.input_size)

        X, y = Variable(X), Variable(y)
        if cuda:
            X, y = X.cuda(), y.cuda()

        show_sample_from_each_class(Q, X, n_classes=n_classes)
        plt.show()

        y_pred = predict_labels(Q, X)

        latent_y, latent_z = Q(X)
        plt.bar(range(n_classes), latent_y.detach().numpy()[0, :])
        plt.show()

        labels.extend(y_pred.numpy())

        show_reconstruction(Q, P, X)
        plt.show()

    plt.figure()
    plt.hist(labels, bins=10)
    plt.show()

    plt.figure()
    plt.hist(latent_z.detach().numpy()[:, 0])
    plt.show()

    from collections import Counter
    print Counter(labels)


def show_reconstruction(Q, P, X):
    Q.eval()
    P.eval()

    latent_y, latent_z = Q(X)

    latent_vec = torch.cat((latent_y, latent_z), 1)
    X_rec = P(latent_vec)

    #print X[0], X_rec[0]
    img_orig = np.array(X[0].data.tolist()).reshape(28, 28)
    img_rec = np.array(X_rec[0].data.tolist()).reshape(28, 28)
    plt.subplot(1, 2, 1)
    plt.imshow(img_orig, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(img_rec, cmap='gray')
    plt.title('predicted label: %s' % torch.argmax(latent_y, dim=1)[0])

def show_sample_from_each_class(Q, X, n_classes):
    Q.eval()

    latent_y, latent_z = Q(X)
    y_class = torch.argmax(latent_y, dim=1).numpy()

    fig, ax = plt.subplots(nrows=3, ncols=n_classes)

    X_samples = {}
    for label in range(n_classes):
        label_indices = np.where(y_class == label)
        try:
            X_samples[label] = X[label_indices][:3, :]  # take first 3 images
        except:
            X_samples[label] = None

    for i, row in zip(range(3), ax):
        for j, col in zip(range(n_classes), row):
            col.set_title("%s" %j)
            col.axis('off')
            try:
                img = X_samples[j][i]
                img = np.array(img.data.tolist()).reshape(28, 28)
                col.imshow(img, cmap='gray')
            except:
                col.imshow(np.zeros((28, 28)), cmap='gray')

    fig.show()


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
Q = Q_net().load(os.path.join('../data', 'encoder_unsupervised'), z_size=2, n_classes=10)
P = P_net().load(os.path.join('../data', 'decoder_unsupervised'), z_size=2, n_classes=10)

import _data_utils
cuda = torch.cuda.is_available()
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
train_labeled_loader, train_unlabeled_loader, valid_loader = _data_utils.load_data(
    data_path='../data', batch_size=100, **kwargs)
show_predicted_labels(Q, P, valid_loader, n_classes=10)
#grid_plot2d(Q, P, valid_loader)
#generate_digits(P, label=4)
