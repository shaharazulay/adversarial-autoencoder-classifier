import argparse
import _data_utils

def init_datasets_main(args=None):
    parser = argparse.ArgumentParser(
        description='Initialize all training and validation datasets.')

    _add_dir_path_to_parser(parser)
    args = parser.parse_args()

    _data_utils.init_datasets(args.dir_path)


def _add_dir_path_to_parser(parser):
    parser.add_argument(
        '--dir-path',
        dest='dir_path',
        required=True,
        help='Path of the data directory')


### REMOVE LATER once setup.py is in place
init_datasets_main()
