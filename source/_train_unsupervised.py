import torch
import itertools
import numpy as np

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ._model import Q_net, P_net, D_net_cat, D_net_gauss
from ._train_utils import *

cuda = torch.cuda.is_available()
seed = 10


def _train_epoch(
    models, optimizers, train_unlabeled_loader, n_classes, z_dim, config_dict):
    '''
    Train procedure for one epoch.
    '''
    epsilon = np.finfo(float).eps

    # load models and optimizers
    P, Q, D_cat, D_gauss, P_mode_decoder = models
    auto_encoder_optim, G_optim, D_optim, info_optim, mode_optim = optimizers

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
        auto_encoder_optim.step()

        # Init gradients
        zero_grad_all(P, Q, D_cat, D_gauss, P_mode_decoder)

        #######################
        # Info phase
        #######################
        continuous_loss =  torch.nn.MSELoss()

        latent_y, latent_z = Q(X)
        latent_vec = torch.cat((latent_y, latent_z), 1)
        X_rec = P(latent_vec)

        latent_y_rec, latent_z_rec = Q(X_rec)

        cat_info_loss = F.binary_cross_entropy(latent_y_rec, latent_y.detach())
        gauss_info_loss = continuous_loss(latent_z_rec, latent_z.detach())

        mode_cyclic_loss = 1.0 * cat_info_loss + 0.1 * gauss_info_loss

        mode_cyclic_loss.backward()
        info_optim.step()

        # Init gradients
        zero_grad_all(P, Q, D_cat, D_gauss, P_mode_decoder)

        #######################
        # Mode reconstruction phase
        #######################
        latent_y, latent_z = Q(X)
        X_mode_rec = P_mode_decoder(latent_y)

        mode_recon_loss = F.binary_cross_entropy(X_mode_rec + epsilon, X + epsilon)

        mode_recon_loss.backward()
        mode_optim.step()

        # Init gradients
        zero_grad_all(P, Q, D_cat, D_gauss, P_mode_decoder)

        #######################
        # Mode cyclic phase
        # #######################
        # Q.eval()
        #
        # latent_y, _ = Q(X)
        # X_mode_rec = P_mode_decoder(latent_y)
        #
        # latent_mode_cylic_y, _ = Q(X_mode_rec)
        # mode_cyclic_loss = 10 * F.binary_cross_entropy(latent_y, latent_mode_cylic_y.detach())  # NOTE: *10 is here
        #
        # mode_cyclic_loss.backward()
        # P_mode_decoder_optim.step()
        #
        # # Init gradients
        # zero_grad_all(P, Q, D_cat, D_gauss, P_mode_decoder)

        #######################
        # Mode disentanglement phase
        #######################
        # mode_disentanglement_loss = 0
        #
        # for label_A in range(n_classes):
        #     latent_y_A = get_categorial(label_A)
        #     X_mode_rec_A = P_mode_decoder(latent_y_A)
        #
        #     for label_B in range(label_A + 1, n_classes):
        #         latent_y_B = get_categorial(label_B)
        #         X_mode_rec_B = P_mode_decoder(latent_y_B)
        #
        #         mode_disentanglement_loss += -F.binary_cross_entropy(X_mode_rec_A + epsilon, X_mode_rec_B.detach() + epsilon)
        #
        # mode_disentanglement_loss /= (n_classes * (n_classes - 1) / 2)
        # mode_disentanglement_loss.backward()
        # P_mode_decoder_optim.step()
        #
        # # Init gradients
        # zero_grad_all(P, Q, D_cat, D_gauss, P_mode_decoder)

        #######################
        # Generator phase
        #######################
        z_real_cat = sample_categorical(batch_size, n_classes=n_classes)
        z_real_gauss = Variable(torch.randn(batch_size, z_dim))
        if cuda:
            z_real_cat = z_real_cat.cuda()
            z_real_gauss = z_real_gauss.cuda()

        Q.train()
        z_fake_cat, z_fake_gauss = Q(X)

        D_fake_cat = D_cat(z_fake_cat)
        D_fake_gauss = D_gauss(z_fake_gauss)

        G_loss = - torch.mean(torch.log(D_fake_cat + epsilon)) - torch.mean(torch.log(D_fake_gauss + epsilon))

        G_loss.backward()
        G_optim.step()

        # Init gradients
        zero_grad_all(P, Q, D_cat, D_gauss, P_mode_decoder)

        #######################
        # Discriminator phase
        #######################
        Q.eval()
        z_fake_cat, z_fake_gauss = Q(X)

        D_real_cat = D_cat(z_real_cat)
        D_real_gauss = D_gauss(z_real_gauss)
        D_fake_cat = D_cat(z_fake_cat)
        D_fake_gauss = D_gauss(z_fake_gauss)

        D_loss_cat = - torch.mean(torch.log(D_real_cat + epsilon) + torch.log(1 - D_fake_cat + epsilon))
        D_loss_gauss = - torch.mean(torch.log(D_real_gauss + epsilon) + torch.log(1 - D_fake_gauss + epsilon))

        D_loss = D_loss_cat + D_loss_gauss

        D_loss.backward()
        D_optim.step()

        Q.train()

        # Init gradients
        zero_grad_all(P, Q, D_cat, D_gauss, P_mode_decoder)

        ### REMOVEEEEEEEEEEEEEEEEEEE
        mode_disentanglement_loss = torch.tensor([0.0]) #############
        ###########3

        # report progress
        # report_loss(
        #     epoch=-1,
        #     all_losses=(D_loss_cat, D_loss_gauss, G_loss, recon_loss, mode_recon_loss, mode_cyclic_loss, mode_disentanglement_loss),
        #     descriptions=['D_loss_cat', 'D_loss_gauss', 'G_loss', 'recon_loss', 'mode_recon_loss', 'mode_cyclic_loss', 'mode_disentanglement_loss'])
        report_progress(float(batch_num) / n_batches)


    return D_loss_cat, D_loss_gauss, G_loss, recon_loss, mode_recon_loss, mode_cyclic_loss, mode_disentanglement_loss


def _get_optimizers(models, config_dict):
    '''
    Set and return all relevant optimizers needed for the training process.
    '''
    P, Q, D_cat, D_gauss, P_mode_decoder = models

    # Set learning rates
    learning_rates = config_dict['learning_rates']

    auto_encoder_lr = learning_rates['auto_encoder_lr']
    generator_lr = learning_rates['generator_lr']
    discriminator_lr = learning_rates['discriminator_lr']
    info_lr = learning_rates['info_lr']
    mode_lr = learning_rates['mode_lr']

    # Set optimizators
    auto_encoder_optim = optim.Adam(itertools.chain(Q.parameters(), P.parameters()), lr=auto_encoder_lr)

    G_optim = optim.Adam(Q.parameters(), lr=generator_lr)
    D_optim = optim.Adam(itertools.chain(D_gauss.parameters(), D_cat.parameters()), lr=discriminator_lr)

    info_optim = optim.Adam(itertools.chain(Q.parameters(), P.parameters()), lr=info_lr)
    mode_optim = optim.Adam(itertools.chain(Q.parameters(), P_mode_decoder.parameters()), lr=mode_lr)

    optimizers = auto_encoder_optim, G_optim, D_optim, info_optim, mode_optim

    return optimizers

def _get_models(n_classes, z_dim, config_dict):
    '''
    Set and return all sub-modules that comprise the full model.
    '''
    model_params = config_dict['model']

    Q = Q_net(z_size=z_dim, n_classes=n_classes, hidden_size=model_params['hidden_size'])
    P = P_net(z_size=z_dim, n_classes=n_classes, hidden_size=model_params['hidden_size'])
    D_cat = D_net_cat(n_classes=n_classes, hidden_size=model_params['hidden_size'])
    D_gauss = D_net_gauss(z_size=z_dim, hidden_size=model_params['hidden_size'])

    # Introducing the new Mode-decoder (it only gets the mode latent y)
    P_mode_decoder = P_net(z_size=0, n_classes=n_classes, hidden_size=model_params['hidden_size'])

    if cuda:
        Q = Q.cuda()
        P = P.cuda()
        D_gauss = D_gauss.cuda()
        D_cat = D_cat.cuda()
        P_mode_decoder = P_mode_decoder.cuda()

    models = P, Q, D_cat, D_gauss, P_mode_decoder
    return models


def train(train_unlabeled_loader, valid_loader, epochs, n_classes, z_dim, output_dir, config_dict):
    '''
    Train the full model.
    '''
    #torch.manual_seed(10)
    learning_curve = []

    models = _get_models(n_classes, z_dim, config_dict)
    optimizers = _get_optimizers(models, config_dict)
    P, Q, D_cat, D_gauss, P_mode_decoder = models

    for epoch in range(epochs):
        all_losses = _train_epoch(
            models,
            optimizers,
            train_unlabeled_loader,
            n_classes,
            z_dim,
            config_dict)

        learning_curve.append(all_losses)

        if epoch % 1 == 0:
            val_acc = unsupervised_classification_accuracy(Q, valid_loader, n_classes=n_classes)

            report_loss(
                epoch,
                all_losses,
                descriptions=[
                    'D_loss_cat', 'D_loss_gauss', 'G_loss', 'recon_loss',
                    'mode_recon_loss', 'mode_cyclic_loss', 'mode_disentanglement_loss'],
                output_dir=output_dir)
            print('Unsupervised classification accuracy: {} %'.format(val_acc))

    return Q, P, P_mode_decoder, learning_curve
