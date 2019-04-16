import os
import argparse
import torch
import yaml
from datetime import datetime

import matplotlib
matplotlib.use("TKAgg")

from matplotlib import pyplot as plt

from ._data_utils import init_datasets, load_data
from ._train_semi_supervised import train as train_semi_supervised
from ._train_unsupervised import train as train_unsupervised

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
    _add_output_dir_path_to_parser(parser)
    _add_configuration_path_to_parser(parser)
    _add_batch_size_to_parser(parser)
    _add_epochs_to_parser(parser)
    _add_n_classes_to_parser(parser)
    _add_z_gauss_size_to_parser(parser)
    args = parser.parse_args()

    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    train_labeled_loader, train_unlabeled_loader, valid_loader = load_data(
        data_path=args.dir_path, batch_size=args.batch_size, **kwargs)

    _make_dir_if_not_exists(args.output_dir_path)
    config_dict = _load_configuration(args.config_path)['semi_supervised']
    print ('\nusing configuration:\n {}'.format(config_dict))

    Q, P, learning_curve = train_semi_supervised(
        train_labeled_loader,
        train_unlabeled_loader,
        valid_loader,
        epochs=args.n_epochs,
        n_classes=args.n_classes,
        z_dim=args.z_size,
        output_dir=args.output_dir_path,
        config_dict=config_dict)

    Q.save(os.path.join(args.output_dir_path, 'encoder_semi_supervised'))
    P.save(os.path.join(args.output_dir_path, 'decoder_semi_supervised'))

    D_loss_cat, D_loss_gauss, G_loss, recon_loss, class_loss =  zip(*learning_curve)

    _save_learning_curve(
        series=[D_loss_cat, D_loss_gauss, G_loss],
        title='Semi-Supervised Adversarial Learning Curve',
        legend=['D_loss_cat', 'D_loss_gauss', 'G_loss'],
        path=os.path.join(args.output_dir_path, 'semi_supervised_advesarial_learning_curve.png')
    )

    _save_learning_curve(
        series=[recon_loss],
        title='Semi-Supervised Reconstruction Learning Curve',
        legend=['recon_loss'],
        path=os.path.join(args.output_dir_path, 'semi_supervised_reconstruction_learning_curve.png')
    )

    _save_learning_curve(
        series=[class_loss],
        title='Semi-Supervised Classification Learning Curve',
        legend=['class_loss'],
        path=os.path.join(args.output_dir_path, 'semi_supervised_classification_learning_curve.png')
    )

    _save_current_configration(config_dict, args.output_dir_path)


def train_unsupervised_model_main(args=None):
    parser = argparse.ArgumentParser(
        description='Train the full model [un-supervised!] and store on disk.')

    _add_dir_path_to_parser(parser)
    _add_output_dir_path_to_parser(parser)
    _add_configuration_path_to_parser(parser)
    _add_batch_size_to_parser(parser)
    _add_epochs_to_parser(parser)
    _add_n_classes_to_parser(parser)
    _add_z_gauss_size_to_parser(parser)
    args = parser.parse_args()

    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    train_labeled_loader, train_unlabeled_loader, valid_loader = load_data(
        data_path=args.dir_path, batch_size=args.batch_size, **kwargs)

    _make_dir_if_not_exists(args.output_dir_path)
    config_dict = _load_configuration(args.config_path)['unsupervised']
    print ('\nusing configuration:\n {}'.format(config_dict))

    Q, P, P_mode_decoder, learning_curve = train_unsupervised(
        train_unlabeled_loader,
        valid_loader,
        epochs=args.n_epochs,
        n_classes=args.n_classes,
        z_dim=args.z_size,
        output_dir=args.output_dir_path,
        config_dict=config_dict)

    Q.save(os.path.join(args.output_dir_path, 'encoder_unsupervised'))
    P.save(os.path.join(args.output_dir_path, 'decoder_unsupervised'))
    P_mode_decoder.save(os.path.join(args.output_dir_path, 'mode_decoder_unsupervised'))

    D_loss_cat, D_loss_gauss, G_loss, recon_loss, mode_recon_loss, mutual_info_loss = zip(*learning_curve)

    _save_learning_curve(
        series=[D_loss_cat, D_loss_gauss, G_loss],
        title='Unsupervised Adversarial Learning Curve',
        legend=['D_loss_cat', 'D_loss_gauss', 'G_loss'],
        path=os.path.join(args.output_dir_path, 'unsupervised_advesarial_learning_curve.png')
    )

    _save_learning_curve(
        series=[recon_loss],
        title='Unsupervised Reconstruction Learning Curve',
        legend=['recon_loss'],
        path=os.path.join(args.output_dir_path, 'unsupervised_reconstruction_learning_curve.png')
    )

    _save_learning_curve(
        series=[mutual_info_loss],
        title='Unsupervised Mutual Info Learning Curve',
        legend=['mutual_info_loss'],
        path=os.path.join(args.output_dir_path, 'unsupervised_mutual_info_learning_curve.png')
    )

    _save_current_configration(config_dict, args.output_dir_path)

def _save_learning_curve(series, title, legend, path):
    plt.figure()
    plt.plot(list(zip(*series)))
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

def _add_output_dir_path_to_parser(parser):
    parser.add_argument(
        '--output-dir-path',
        dest='output_dir_path',
        default=os.path.join('out', datetime.now().strftime("%Y-%m-%d-%H:%M:%S")),
        help='Path of the output directory')

def _add_configuration_path_to_parser(parser):
    parser.add_argument(
        '--config-path',
        dest='config_path',
        default='source/_config.yml',
        help='Path of the configuration YAML file (default: local default configuration)')

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

def _make_dir_if_not_exists(dir_):
    if not os.path.exists(dir_):
        os.makedirs(dir_)

def _load_configuration(path):
    with open(path, 'r') as f_cfg:
        config = yaml.load(f_cfg)
    return config

def _save_current_configration(config_dict, dir_):
    with open(os.path.join(dir_, 'config.yml'), 'w') as f_cfg:
        dump = yaml.dump(config_dict, default_flow_style=False)
        f_cfg.write(dump)
