import os
import argparse
import torch

import _data_utils
import _train

cuda = torch.cuda.is_available()


def init_datasets_main(args=None):
    parser = argparse.ArgumentParser(
        description='Initialize all training and validation datasets.')

    _add_dir_path_to_parser(parser)
    args = parser.parse_args()

    _data_utils.init_datasets(args.dir_path)


def train_model_main(args=None):
    parser = argparse.ArgumentParser(
        description='Train the full model ans store on disk.')

    _add_dir_path_to_parser(parser)
    _add_batch_size_to_parser(parser)
    _add_epochs_to_parser(parser)
    _add_n_classes_to_parser(parser)
    _add_z_gauss_size_to_parser(parser)
    args = parser.parse_args()

    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    train_labeled_loader, train_unlabeled_loader, valid_loader = _data_utils.load_data(
        data_path=args.dir_path, batch_size=args.batch_size, **kwargs)
    Q, P = _train.train(
        train_labeled_loader,
        train_unlabeled_loader,
        valid_loader,
        epochs=args.n_epochs,
        n_classes=args.n_classes,
        z_dim=args.z_size)
        
    Q.save(os.path.join(args.dir_path, 'encoder'))
    P.save(os.path.join(args.dir_path, 'decoder'))


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
