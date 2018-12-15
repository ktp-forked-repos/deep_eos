"""Implement various helper functions."""

from collections import namedtuple
from typing import Set

from deep_eos.data import Context

EOS_CHARS = ['.', ';', '!', '?', ':']
EOS_MARKER = '</eos>'
SEPS = [' ', '\n']
EvaluationResult = namedtuple("EvaluationResult", 'precision, recall, f_score')


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
