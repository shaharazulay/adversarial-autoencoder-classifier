import os
import argparse
import torch

import matplotlib.pyplot as plt

from _data_utils import init_datasets, load_data
from _train_semi_supervised import train as train_semi_supervised
from _train_unsupervised import train as train_unsupervised

cuda = torch.cuda.is_available()


def init_datasets_main(args=None):
    parser = argparse.ArgumentParser(
        description='Initialize all training and validation datasets.')

    _add_dir_path_to_parser(parser)
    args = parser.parse_args()

    init_datasets(args.dir_path)


def train_semi_supervised_model_main(args=None):
    parser = argparse.ArgumentParser(
        description='Train the full model [semi-supervised] and store on disk.')

    _add_dir_path_to_parser(parser)
    _add_batch_size_to_parser(parser)
    _add_epochs_to_parser(parser)
    _add_n_classes_to_parser(parser)
    _add_z_gauss_size_to_parser(parser)
    args = parser.parse_args()

    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    train_labeled_loader, train_unlabeled_loader, valid_loader = load_data(
        data_path=args.dir_path, batch_size=args.batch_size, **kwargs)
    Q, P, learning_curve = train_semi_supervised(
        train_labeled_loader,
        train_unlabeled_loader,
        valid_loader,
        epochs=args.n_epochs,
        n_classes=args.n_classes,
        z_dim=args.z_size)

    Q.save(os.path.join(args.dir_path, 'encoder_semi_supervised'))
    P.save(os.path.join(args.dir_path, 'decoder_semi_supervised'))

    D_loss_cat, D_loss_gauss, G_loss, recon_loss, class_loss =  zip(*learning_curve)

    _save_learning_curve(
        series=[D_loss_cat, D_loss_gauss, G_loss],
        title='Semi-Supervised Adversarial Learning Curve',
        legend=['D_loss_cat', 'D_loss_gauss', 'G_loss'],
        path=os.path.join(args.dir_path, 'semi_supervised_advesarial_learning_curve.png')
    )

    _save_learning_curve(
        series=[recon_loss],
        title='Semi-Supervised Reconstruction Learning Curve',
        legend=['recon_loss'],
        path=os.path.join(args.dir_path, 'semi_supervised_reconstruction_learning_curve.png')
    )

    _save_learning_curve(
        series=[class_loss],
        title='Semi-Supervised Classification Learning Curve',
        legend=['class_loss'],
        path=os.path.join(args.dir_path, 'semi_supervised_classification_learning_curve.png')
    )


def train_unsupervised_model_main(args=None):
    parser = argparse.ArgumentParser(
        description='Train the full model [un-supervised!] and store on disk.')

    _add_dir_path_to_parser(parser)
    _add_batch_size_to_parser(parser)
    _add_epochs_to_parser(parser)
    _add_n_classes_to_parser(parser)
    _add_z_gauss_size_to_parser(parser)
    args = parser.parse_args()

    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    train_labeled_loader, train_unlabeled_loader, valid_loader = load_data(
        data_path=args.dir_path, batch_size=args.batch_size, **kwargs)
    Q, P, P_mode_decoder, learning_curve = train_unsupervised(
        train_unlabeled_loader,
        valid_loader,
        epochs=args.n_epochs,
        n_classes=args.n_classes,
        z_dim=args.z_size)

    Q.save(os.path.join(args.dir_path, 'encoder_unsupervised'))
    P.save(os.path.join(args.dir_path, 'decoder_unsupervised'))
    P_mode_decoder.save(os.path.join(args.dir_path, 'mode_decoder_unsupervised'))

    D_loss_cat, D_loss_gauss, G_loss, recon_loss, mode_recon_loss, mode_cyclic_loss, mode_disentanglement_loss = zip(*learning_curve)

    _save_learning_curve(
        series=[D_loss_cat, D_loss_gauss, G_loss],
        title='Unsupervised Adversarial Learning Curve',
        legend=['D_loss_cat', 'D_loss_gauss', 'G_loss'],
        path=os.path.join(args.dir_path, 'unsupervised_advesarial_learning_curve.png')
    )

    _save_learning_curve(
        series=[recon_loss],
        title='Unsupervised Reconstruction Learning Curve',
        legend=['recon_loss'],
        path=os.path.join(args.dir_path, 'unsupervised_reconstruction_learning_curve.png')
    )

    _save_learning_curve(
        series=[mode_cyclic_loss],
        title='Unsupervised Cyclic Info Learning Curve',
        legend=['mode_cyclic_loss'],
        path=os.path.join(args.dir_path, 'unsupervised_cyclic_info_learning_curve.png')
    )


def _save_learning_curve(series, title, legend, path):
    plt.figure()
    plt.plot(zip(*series))
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(legend)
    plt.savefig(path)


def _add_dir_path_to_parser(parser):
    parser.add_argument(
        '--dir-path',
        dest='dir_path',
        required=True,
        help='Path of the data directory')

def _add_batch_size_to_parser(parser):
    parser.add_argument(
        '--batch-size',
        dest='batch_size',
        type=int,
        default=100,
        help='Batch size for the SGD during training process (default: 100)')

def _add_epochs_to_parser(parser):
    parser.add_argument(
        '--n-epochs',
        dest='n_epochs',
        default=10,
        type=int,
        help='Number of training epochs (default: 10)')

def _add_n_classes_to_parser(parser):
    parser.add_argument(
        '--n-classes',
        dest='n_classes',
        type=int,
        default=10,
        help='Number of available class labels (default: 10)')

def _add_z_gauss_size_to_parser(parser):
    parser.add_argument(
        '--z-size',
        dest='z_size',
        type=int,
        default=5,
        help='Number of nodes used by the latent gauss distributed z encoding (default: 5)')
