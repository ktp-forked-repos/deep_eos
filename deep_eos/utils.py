"""Implement various helper functions."""

import logging
import os
import shutil
import tempfile
from collections import namedtuple
from pathlib import Path
from typing import Set
from urllib.parse import urlparse

import requests
import tqdm

from deep_eos.data import Context

EOS_CHARS = ['.', ';', '!', '?', ':']
EOS_MARKER = '</eos>'
SEPS = [' ', '\n']
EvaluationResult = namedtuple("EvaluationResult", 'precision, recall, f_score')
CACHE_ROOT = os.path.expanduser(os.path.join('~', '.deep_eos'))

LOG = logging.getLogger(__name__)

def get_char_context(left_window, right_window, buffer):
    """Implement method for fetching left and right context around a potential end-of-sentence.

    :param left_window: left window size
    :param right_window: right window size
    :param buffer: text buffer
    :return: context generator
    """
    for position, char in enumerate(buffer):
        if char in EOS_CHARS and position + 1 < len(buffer) and buffer[position + 1] in SEPS:
            if position - left_window > 0:
                left_context = buffer[position - left_window: position]
            else:
                left_context = buffer[0: position].rjust(left_window)

            right_context = buffer[position + 1: position + right_window + 1]

            if len(right_context) != right_window:
                right_context = right_context.ljust(right_window)

            label = 1 if buffer[position + 1] == '\n' else 0

            context_str = left_context + char + right_context

            context_str = context_str.replace('\n', ' ')

            context_str = context_str.replace(' ', '▁')

            context = Context(context=context_str,
                              label=int(label),
                              position=position)

            yield context


def parse_dataset_to_buffer(filename):
    """Parse dataset to string buffer.

    :param filename: filename to be read
    :return: list of sentences
    """
    sentences = []
    with open(filename, 'rt') as f_p:
        for line in f_p.readlines():
            line = line.rstrip()

            if line.startswith('# text =') and line[-1] in EOS_CHARS:
                sentences.append(line.split('=')[1].lstrip())

    return "\n".join(sentences)


def parse_file_to_buffer(filename):
    """Parse file directly into a text buffer.

    :param filename: filename to be read
    :return:
    """
    with open(filename, 'rt') as f_p:
        return "\n".join(f_p.readlines())


def calculate_evaluation_metrics(gold_positions: Set[int], predicted_positions: Set[int]) \
        -> EvaluationResult:
    """Calculate evaluation metrics (precision, recall and f-score).

    :param gold_positions: set of gold eos positions
    :param predicted_positions: set of predicted eos positions
    :return: precision, recall and f_score as evaluation result (namedtuple)
    """
    true_positives = len(predicted_positions.intersection(gold_positions))
    false_positives = len(predicted_positions.difference(gold_positions))
    false_negatives = len(gold_positions.difference(predicted_positions))

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    f_score = 2 * (precision * recall) / (precision + recall)

    return EvaluationResult(precision=precision, recall=recall, f_score=f_score)


def cached_path(url: str, cache_dir: Path) -> Path:
    """Perform caching of a specific url for a given path.

    If the local path does not exist, download file from url.

    :param url: url of file
    :param cache_dir: local cache dir
    :return: path of local file
    """
    dataset_cache = Path(CACHE_ROOT) / cache_dir
    parsed = urlparse(url)

    if parsed.scheme in ['https']:

        dataset_cache.mkdir(parents=True, exist_ok=True)

        cache_path = dataset_cache / url.split('/')[-1]

        if cache_path.exists():
            return cache_path

        response = requests.head(url)

        if not response:
            LOG.error(f'Could not retrieve model from {url}')

        _, temp_filename = tempfile.mkstemp()

        LOG.info(f'{url} not found in cache, downloading it to {temp_filename}')

        req = requests.get(url, stream=True)
        file_size = int(req.headers.get('Content-Length'))
        chunk_size = 1024
        num_bars = int(file_size / chunk_size)

        with open(temp_filename, 'wb') as temp_file:
            for chunk in tqdm.tqdm(req.iter_content(chunk_size=chunk_size),
                                   total=num_bars,
                                   unit='B',
                                   leave=True):
                temp_file.write(chunk)

        shutil.copyfile(temp_filename, str(cache_path))
        os.remove(temp_filename)

        return cache_path
