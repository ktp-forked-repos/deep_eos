"""Implement commandline tool for training a model with deep_eos."""

from pathlib import Path
from typing import List

import click
import toml

from deep_eos.data import Corpus, Context
from deep_eos.data_fetcher import EOSContextFetcher, Task
from deep_eos.models.deep_eos_lstm_model import DeepEos
from deep_eos.trainers.deep_eos_trainer import DeepEosTrainer


def training(configuration: dict):  # pylint: disable=too-many-locals
    """Start training a deep_eos model.

    :param configuration: configuration dictionary (from toml)
    """
    left_ws = configuration['deep_eos']['left_ws']
    right_ws = configuration['deep_eos']['right_ws']

    task_name = 'ud' if 'ud' in configuration else 'text'

    task = Task.UD_UNTOK if task_name == 'ud' else Task.TEXT

    train_contexts: List[Context] = EOSContextFetcher.fetch_contexts(
        task=task,
        base_path=configuration[task_name]['train_path'],
        left_ws=left_ws,
        right_ws=right_ws)

    dev_contexts: List[Context] = EOSContextFetcher.fetch_contexts(
        task=task,
        base_path=configuration[task_name]['dev_path'],
        left_ws=left_ws,
        right_ws=right_ws)

    test_contexts: List[Context] = EOSContextFetcher.fetch_contexts(
        task=task,
        base_path=configuration[task_name]['test_path'],
        left_ws=left_ws,
        right_ws=right_ws)

    batch_size: int = configuration['training']['batch_size']

    corpus: Corpus = Corpus(train_contexts=train_contexts,
                            dev_contexts=dev_contexts,
                            test_contexts=test_contexts,
                            batch_size=batch_size)

    vocab = corpus.context_fields.vocab
    label_vocab = corpus.label_fields.vocab

    embedding_size: int = configuration['lstm']['embedding_size']
    hidden_size: int = configuration['lstm']['hidden_size']
    output_size: int = configuration['lstm']['output_size']
    epochs: int = configuration['training']['epochs']
    learning_rate: float = configuration['training']['learning_rate']
    model_file: Path = Path(configuration['training']['model_file'])

    deep_eos: DeepEos = DeepEos(vocab=vocab, label_vocab=label_vocab, batch_size=batch_size,
                                hidden_size=hidden_size, embedding_length=embedding_size,
                                output_size=output_size)

    trainer: DeepEosTrainer = DeepEosTrainer(model=deep_eos, corpus=corpus)

    trainer.train(model_file=model_file, learning_rate=learning_rate, batch_size=batch_size,
                  epochs=epochs)


@click.command()
@click.option('--config', help='Toml configuration file')
def parse_arguments(config):
    """Parse commandline arguments.

    :param config: toml configuration file
    :return:
    """
    configuration = toml.load(config)

    training(configuration)


if __name__ == '__main__':
    parse_arguments()  # pylint: disable=no-value-for-parameter
