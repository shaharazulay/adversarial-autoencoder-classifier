import argparse
import time
import torch
import pickle
import numpy as np
import itertools
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from _data_utils import MNISTSlice
from _model import Q_net, P_net, D_net_cat, D_net_gauss
from _train_utils import *

# Training settings
parser = argparse.ArgumentParser(description='PyTorch semi-supervised MNIST')

parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--epochs', type=int, default=500, metavar='N',
                    help='number of epochs to train (default: 10)')

args = parser.parse_args()
cuda = torch.cuda.is_available()

seed = 10


kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
n_classes = 10
z_dim = 2
X_dim = 784
# y_dim = 10
train_batch_size = args.batch_size
valid_batch_size = args.batch_size
# N = 1000
epochs = args.epochs

##################################
# Load data and create Data loaders
##################################
def load_data(data_path='../data/'):
    print('loading data!')

    trainset_labeled = MNISTSlice.load(data_path + 'train_labeled.p')
    trainset_unlabeled = MNISTSlice.load(data_path + 'train_unlabeled.p')
    validset = MNISTSlice.load(data_path + 'validation.p')

    train_labeled_loader = torch.utils.data.DataLoader(
        trainset_labeled,
        batch_size=train_batch_size,
        shuffle=True,
        **kwargs)

    train_unlabeled_loader = torch.utils.data.DataLoader(
        trainset_unlabeled,
        batch_size=train_batch_size,
        shuffle=True,
        **kwargs)

    valid_loader = torch.utils.data.DataLoader(
        validset, batch_size=valid_batch_size, shuffle=True)

    print "DATASET SIZES: ", len(trainset_labeled), len(trainset_unlabeled), len(validset)
    return train_labeled_loader, train_unlabeled_loader, valid_loader


####################
# Train procedure
####################
def train(
    P, Q, D_cat, D_gauss,
    P_decoder_optim, Q_encoder_optim,
    Q_classifier_optim,
    Q_regularization_optim, D_cat_optim, D_gauss_optim,
    train_labeled_loader, train_unlabeled_loader):
    '''
    Train procedure for one epoch.
    '''
    TINY = 1e-15
    # Set the networks in train mode (apply dropout when needed)
    Q.train()
    P.train()
    D_cat.train()
    D_gauss.train()

    if train_unlabeled_loader is None:
        train_unlabeled_loader = train_labeled_loader

    # Loop through the labeled and unlabeled dataset getting one batch of samples from each
    # The batch size has to be a divisor of the size of the dataset or it will return
    # invalid samples
    for (X_l, target_l), (X_u, target_u) in itertools.izip(train_labeled_loader, train_unlabeled_loader):

        for X, target in [(X_u, target_u), (X_l, target_l)]:
            if target[0] == -1:
                labeled = False
            else:
                labeled = True

            # Load batch and normalize samples to be between 0 and 1
            #X = X * 0.3081 + 0.1307
            X.resize_(train_batch_size, X_dim)

            X, target = Variable(X), Variable(target)
            if cuda:
                X, target = X.cuda(), target.cuda()

            # Init gradients
            zero_grad_all(P, Q, D_cat, D_gauss)

            #######################
            # Reconstruction phase
            #######################
            if not labeled:
                z_sample = torch.cat(Q(X), 1)
                X_sample = P(z_sample)

                recon_loss = F.binary_cross_entropy(X_sample + TINY, X.resize_(train_batch_size, X_dim) + TINY)
                recon_loss = recon_loss
                recon_loss.backward()
                P_decoder_optim.step()
                Q_encoder_optim.step()

                # Init gradients
                zero_grad_all(P, Q, D_cat, D_gauss)

                recon_loss = recon_loss
                #######################
                # Regularization phase
                #######################
                # Discriminator
                Q.eval()
                z_real_cat = sample_categorical(train_batch_size, n_classes=n_classes)
                z_real_gauss = Variable(torch.randn(train_batch_size, z_dim))
                if cuda:
                    z_real_cat = z_real_cat.cuda()
                    z_real_gauss = z_real_gauss.cuda()

                z_fake_cat, z_fake_gauss = Q(X)

                D_real_cat = D_cat(z_real_cat)
                D_real_gauss = D_gauss(z_real_gauss)
                D_fake_cat = D_cat(z_fake_cat)
                D_fake_gauss = D_gauss(z_fake_gauss)

                D_loss_cat = -torch.mean(torch.log(D_real_cat + TINY) + torch.log(1 - D_fake_cat + TINY))
                D_loss_gauss = -torch.mean(torch.log(D_real_gauss + TINY) + torch.log(1 - D_fake_gauss + TINY))

                D_loss = D_loss_cat + D_loss_gauss
                D_loss = D_loss

                D_loss.backward()
                D_cat_optim.step()
                D_gauss_optim.step()

                # Init gradients
                zero_grad_all(P, Q, D_cat, D_gauss)

                # Generator
                Q.train()
                z_fake_cat, z_fake_gauss = Q(X)

                D_fake_cat = D_cat(z_fake_cat)
                D_fake_gauss = D_gauss(z_fake_gauss)

                G_loss = - torch.mean(torch.log(D_fake_cat + TINY)) - torch.mean(torch.log(D_fake_gauss + TINY))
                G_loss = G_loss
                G_loss.backward()
                Q_regularization_optim.step()

                # Init gradients
                zero_grad_all(P, Q, D_cat, D_gauss)

            #######################
            # Semi-supervised phase
            #######################
            if labeled:
                pred, _ = Q(X)
                class_loss = F.cross_entropy(pred, target)
                class_loss.backward()
                Q_classifier_optim.step()

                # Init gradients
                zero_grad_all(P, Q, D_cat, D_gauss)

    return D_loss_cat, D_loss_gauss, G_loss, recon_loss, class_loss


def generate_model(train_labeled_loader, train_unlabeled_loader, valid_loader):
    torch.manual_seed(10)

    if cuda:
        Q = Q_net().cuda()
        P = P_net().cuda()
        D_cat = D_net_cat().cuda()
        D_gauss = D_net_gauss().cuda()
    else:
        Q = Q_net()
        P = P_net()
        D_gauss = D_net_gauss()
        D_cat = D_net_cat()

    # Set learning rates
    auto_encoder_lr = 0.0006
    regularization_lr = 0.0008
    classifier_lr = 0.001

    # Set optimizators
    P_decoder_optim = optim.Adam(P.parameters(), lr=auto_encoder_lr)
    Q_encoder_optim = optim.Adam(Q.parameters(), lr=auto_encoder_lr)

    Q_regularization_optim = optim.Adam(Q.parameters(), lr=regularization_lr)
    D_gauss_optim = optim.Adam(D_gauss.parameters(), lr=regularization_lr)
    D_cat_optim = optim.Adam(D_cat.parameters(), lr=regularization_lr)

    Q_classifier_optim = optim.Adam(Q.parameters(), lr=classifier_lr)

    start = time.time()
    for epoch in range(epochs):
        D_loss_cat, D_loss_gauss, G_loss, recon_loss, class_loss = train(
            P, Q, D_cat, D_gauss,
            P_decoder_optim, Q_encoder_optim,
            Q_classifier_optim,
            Q_regularization_optim, D_cat_optim, D_gauss_optim,
            train_labeled_loader, train_unlabeled_loader)

        if epoch % 10 == 0:
            train_acc = classification_accuracy(Q, train_labeled_loader)
            val_acc = classification_accuracy(Q, valid_loader)
            report_loss(epoch, D_loss_cat, D_loss_gauss, G_loss, recon_loss)
            print('Classification Loss: {:.3}'.format(class_loss.item()))
            print('Train accuracy: {} %'.format(train_acc))
            print('Validation accuracy: {} %'.format(val_acc))
    end = time.time()
    print('Training time: {} seconds'.format(end - start))

    return Q, P


if __name__ == '__main__':
    train_labeled_loader, train_unlabeled_loader, valid_loader = load_data()
    Q, P = generate_model(train_labeled_loader, train_unlabeled_loader, valid_loader)

    ## show some images
    for batch_idx, (X, target) in enumerate(valid_loader):
        X.resize_(valid_loader.batch_size, X_dim)
        X, target = Variable(X), Variable(target)
        if cuda:
            X, target = X.cuda(), target.cuda()
        pred = predict_labels(Q, X)

        plt.figure()
        plt.imshow(X[:784].resize_(28 ,28), cmap='gray')
        plt.title('Orig: %s, Pred: %s' % (target[0], pred[0]))
        plt.show()
