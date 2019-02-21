import torch
import itertools
import numpy as np

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from _model import Q_net, P_net, D_net_cat, D_net_gauss
from _train_utils import *

cuda = torch.cuda.is_available()
seed = 10
pixelwise_loss = torch.nn.L1Loss()


def _train_epoch(
    models, optimizers, train_unlabeled_loader, n_classes, z_dim):
    '''
    Train procedure for one epoch.
    '''
    epsilon = np.finfo(float).eps

    # load models and optimizers
    P, Q, D_cat, D_gauss, P_mode_decoder = models
    P_decoder_optim, Q_encoder_optim, Q_mode_encoder_optim, P_mode_decoder_optim, Q_regularization_optim, D_cat_optim, D_gauss_optim = optimizers

    # Set the networks in train mode (apply dropout when needed)
    train_all(P, Q, D_cat, D_gauss, P_mode_decoder)

    batch_size = train_unlabeled_loader.batch_size
    n_batches = len(train_unlabeled_loader)

    # Loop through the unlabeled dataset
    for batch_num, (X, target) in enumerate(train_unlabeled_loader):

        X.resize_(batch_size, Q.input_size)

        X, target = Variable(X), Variable(target)
        if cuda:
            X, target = X.cuda(), target.cuda()

        # Init gradients
        zero_grad_all(P, Q, D_cat, D_gauss, P_mode_decoder)

        #######################
        # Reconstruction phase
        #######################
        latent_vec = torch.cat(Q(X), 1)
        X_rec = P(latent_vec)

        recon_loss = F.binary_cross_entropy(X_rec + epsilon, X + epsilon)

        recon_loss.backward()
        P_decoder_optim.step()
        Q_encoder_optim.step()

        # Init gradients
        zero_grad_all(P, Q, D_cat, D_gauss, P_mode_decoder)

        #######################
        # Mode decoding phase
        #######################
        latent_y, latent_z = Q(X)
        X_mode_rec = P_mode_decoder(latent_y)

        mode_recon_loss = F.binary_cross_entropy(X_mode_rec + epsilon, X + epsilon)

        mode_recon_loss.backward()
        P_mode_decoder_optim.step()
        Q_mode_encoder_optim.step()

        # Init gradients
        zero_grad_all(P, Q, D_cat, D_gauss, P_mode_decoder)

        #######################
        # Regularization phase
        #######################
        # Discriminator
        Q.eval()
        z_real_cat = sample_categorical(batch_size, n_classes=n_classes)
        z_real_gauss = Variable(torch.randn(batch_size, z_dim))
        if cuda:
            z_real_cat = z_real_cat.cuda()
            z_real_gauss = z_real_gauss.cuda()

        z_fake_cat, z_fake_gauss = Q(X)

        D_real_cat = D_cat(z_real_cat)
        D_real_gauss = D_gauss(z_real_gauss)
        D_fake_cat = D_cat(z_fake_cat)
        D_fake_gauss = D_gauss(z_fake_gauss)

        D_loss_cat = - torch.mean(torch.log(D_real_cat + epsilon) + torch.log(1 - D_fake_cat + epsilon))
        D_loss_gauss = - torch.mean(torch.log(D_real_gauss + epsilon) + torch.log(1 - D_fake_gauss + epsilon))

        D_loss = D_loss_cat + D_loss_gauss
        D_loss = D_loss

        D_loss.backward()
        D_cat_optim.step()
        D_gauss_optim.step()

        # Init gradients
        zero_grad_all(P, Q, D_cat, D_gauss, P_mode_decoder)

        # Generator
        Q.train()
        z_fake_cat, z_fake_gauss = Q(X)

        D_fake_cat = D_cat(z_fake_cat)
        D_fake_gauss = D_gauss(z_fake_gauss)

        G_loss = - torch.mean(torch.log(D_fake_cat + epsilon)) - torch.mean(torch.log(D_fake_gauss + epsilon))
        G_loss = G_loss
        G_loss.backward()
        Q_regularization_optim.step()

        # Init gradients
        zero_grad_all(P, Q, D_cat, D_gauss, P_mode_decoder)

        # report progress
        ##report_loss(-1, D_loss_cat, D_loss_gauss, G_loss, recon_loss, mode_recon_loss)
        report_progress(float(batch_num) / n_batches)

    return D_loss_cat, D_loss_gauss, G_loss, recon_loss, mode_recon_loss


def train(train_unlabeled_loader, valid_loader, epochs, n_classes, z_dim):
    torch.manual_seed(10)

    Q = Q_net(z_size=z_dim, n_classes=n_classes)
    P = P_net(z_size=z_dim, n_classes=n_classes)
    D_cat = D_net_cat(n_classes=n_classes)
    D_gauss = D_net_gauss(z_size=z_dim)

    # Introducing the new Mode-decoder (it only gets the mode latent y)
    P_mode_decoder = P_net(z_size=0, n_classes=n_classes)

    if cuda:
        Q = Q.cuda()
        P = P.cuda()
        D_gauss = D_gauss.cuda()
        D_cat = D_cat.cuda()
        P_mode_decoder = P_mode_decoder.cuda()

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

    P_mode_decoder_optim = optim.Adam(P_mode_decoder.parameters(), lr=classifier_lr)
    Q_mode_encoder_optim = optim.Adam(Q.parameters(), lr=classifier_lr)

    models = P, Q, D_cat, D_gauss, P_mode_decoder
    optimizers = P_decoder_optim, Q_encoder_optim, Q_mode_encoder_optim, P_mode_decoder_optim, Q_regularization_optim, D_cat_optim, D_gauss_optim

    for epoch in range(epochs):
        D_loss_cat, D_loss_gauss, G_loss, recon_loss, mode_recon_loss = _train_epoch(
            models,
            optimizers,
            train_unlabeled_loader,
            n_classes,
            z_dim)

        if epoch % 2 == 0:
            val_acc = classification_accuracy(Q, valid_loader)
            report_loss(epoch, D_loss_cat, D_loss_gauss, G_loss, recon_loss, mode_recon_loss)
            #print('Classification Loss: {:.3}'.format(class_loss.item()))
            print('Validation accuracy: {} %'.format(val_acc))

    return Q, P
