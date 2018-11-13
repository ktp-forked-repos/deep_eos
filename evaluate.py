"""Implement commandline tool for evaluating with a trained deep_eos model."""

from pathlib import Path
from typing import Set

import click
import toml

from deep_eos.utils import parse_dataset_to_buffer, parse_file_to_buffer


def retrieve_eos_positions(buffer: str, eos_marker: str):
    """Retrieve recognized eos in text buffer.

    :param buffer: text buffer
    :param eos_marker: eos marker, like </eos>
    :return:
    """
    tok_idx: int = 0

    for token in buffer.split(' '):
        if token == eos_marker:
            yield tok_idx
        else:
            tok_idx += len(token)


def calculate_metrix(gold_positions: Set[int], predicted_positions: Set[int]):
    """Calculate evaluation matrix (precision, recall and f-score).

    :param gold_positions: set of gold eos positions
    :param predicted_positions: set of predicted eos positions
    :return:
    """
    true_positives = len(predicted_positions.intersection(gold_positions))
    false_positives = len(predicted_positions.difference(gold_positions))
    false_negatives = len(gold_positions.difference(predicted_positions))

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    f_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f_score


def evaluation(configuration: dict):
    """Start evaluation on datasets.

    :param configuration: configuration dictionary (from toml)
    :return:
    """
    task_name = 'ud' if 'ud' in configuration else 'text'

    for dataset in ['dev_set', 'test_set']:
        tagged_file = configuration['prediction']['dev_tagged_file'] \
            if dataset == 'dev_set' else configuration['prediction']['test_tagged_file']

        eos_marker = configuration['prediction']['eos_marker']

        with open(tagged_file, 'rt') as f_p:
            predicted_buffer = " ".join(f_p.readlines())

        predicted_buffer = predicted_buffer.replace(eos_marker, f' {eos_marker} ')

        predicted_eos_positions: Set[int] = set(retrieve_eos_positions(buffer=predicted_buffer,
                                                                       eos_marker=eos_marker))

        dataset_type = 'dev_path' if dataset == 'dev_set' else 'test_path'

        dataset_path = Path(configuration[task_name][dataset_type])

        if task_name == 'ud':
            gold_buffer = parse_dataset_to_buffer(dataset_path)
        else:
            gold_buffer = parse_file_to_buffer(dataset_path)

        gold_buffer += "\n"
        gold_buffer = gold_buffer.replace("\n", f' {eos_marker} ')

        gold_eos_positions: Set[int] = set(retrieve_eos_positions(buffer=gold_buffer,
                                                                  eos_marker=eos_marker))

        precision, recall, f_score = calculate_metrix(gold_eos_positions,
                                                      predicted_eos_positions)

        print(f'Evaluation on {dataset.replace("_", " ")}')
        print(f'Precision: {precision}, Recall: {recall}, F-Score: {f_score}')
        print('')


@click.command()
@click.option('--config', help='Toml configuration file')
def parse_arguments(config):
    """Parse commandline arguments.

    :param config: toml configuration file
    :return:
    """
    configuration = toml.load(config)

    evaluation(configuration)


if __name__ == '__main__':
    parse_arguments()  # pylint: disable=no-value-for-parameter
