"""Implement commandline tool for predicting eos with a trained deep_eos model."""

import logging
from pathlib import Path

import click
import toml

from deep_eos.models.deep_eos_lstm_model import DeepEos
from deep_eos.utils import parse_dataset_to_buffer, parse_file_to_buffer

# Info logs can confuse evaluation later, so only display warnings
logging.getLogger('deep_eos').setLevel(logging.WARNING)


def prediction(configuration: dict):
    """Start prediction with a deep_eos model.

    :param configuration: configuration dictionary (from toml)
    """
    task_name = 'ud' if 'ud' in configuration else 'text'
    model_file = configuration['training']['model_file']
    left_ws = configuration['deep_eos']['left_ws']
    right_ws = configuration['deep_eos']['right_ws']
    eos_marker = configuration['prediction']['eos_marker']

    deep_eos: DeepEos = DeepEos.load_from_file(model_file)

    for dataset in ['dev_path', 'test_path']:
        dataset_file = Path(configuration[task_name][dataset])

        output_file = configuration['prediction']['dev_tagged_file'] \
            if dataset == 'dev_path' else configuration['prediction']['test_tagged_file']

        if task_name == 'ud':
            buffer = parse_dataset_to_buffer(dataset_file)
        else:
            buffer = parse_file_to_buffer(dataset_file)

        buffer = buffer.replace("\n", " ")

        f_output = open(output_file, 'wt')

        deep_eos.predict(buffer=buffer, left_ws=left_ws, right_ws=right_ws, eos_marker=eos_marker,
                         io_stream=f_output)

        f_output.close()


@click.command()
@click.option('--config', help='Toml configuration file')
def parse_arguments(config):
    """Parse commandline arguments.

    :param config: toml configuration file
    :return:
    """
    configuration = toml.load(config)

    prediction(configuration)


if __name__ == '__main__':
    parse_arguments()  # pylint: disable=no-value-for-parameter
