"""Provide implementations for Context and Corpus class."""

import logging
from typing import List

from torchtext import data

LOG = logging.getLogger(__name__)


class Context:  # pylint: disable=too-few-public-methods
    """Implement class that represents information about end-of-sentence context."""

    def __init__(self, context: str, label: int, position: int):
        """Implement constructor for context class.

        :param context: sequence of characters (context) around a potential end-of-sentence
        :param label: label (eos or non-eos)
        :param position: position in buffer
        """
        self.context = context
        self.label = label
        self.position = position


class Corpus:  # pylint: disable=too-few-public-methods,too-many-instance-attributes
    """Implement corpus class that includes dataset contexts and torchtext iterators."""

    def __init__(self, train_contexts: List[Context],
                 dev_contexts: List[Context],
                 test_contexts: List[Context],
                 batch_size: int = 32):
        """Implement constructor for corpus class.

        :param train_contexts: list of contexts in training set
        :param dev_contexts: list of contexts in development set
        :param test_contexts: list of contexts in test set
        :param batch_size: size for one batch
        """
        self.train_contexts = train_contexts
        self.dev_contexts = dev_contexts
        self.test_contexts = test_contexts

        self._write_csv(train_contexts, 'train.tsv')
        self._write_csv(dev_contexts, 'dev.tsv')
        self._write_csv(test_contexts, 'test.tsv')

        fix_length: int = len(train_contexts[0].context)

        self.context_fields = data.Field(batch_first=True,
                                         include_lengths=True,
                                         fix_length=fix_length,
                                         tokenize=list,
                                         sequential=True,
                                         pad_token='▁')
        self.label_fields = data.LabelField(sequential=False)

        self.train_, self.dev_, self.test_ = data.TabularDataset.splits(
            path='./',
            train='train.tsv',
            validation='dev.tsv',
            test='test.tsv',
            format='tsv',
            fields=[('label', self.label_fields),
                    ('text', self.context_fields)])

        self.context_fields.build_vocab(self.train_)
        self.label_fields.build_vocab(self.train_)

        params = {
            'datasets': (self.train_, self.dev_, self.test_),
            'batch_size': batch_size,
            'sort_key': lambda x: len(x.text),
            'repeat': False,
            'shuffle': True,
            'sort': False
        }

        self.train_iter, self.dev_iter, self.test_iter = data.BucketIterator.splits(**params)  # pylint: disable=unbalanced-tuple-unpacking # noqa: E501

        LOG.info('Corpus successfully created with %s train contexts, %s dev contexts and '
                 '%s test contexts', len(train_contexts), len(dev_contexts), len(test_contexts))

    @staticmethod
    def _write_csv(contexts: List[Context], filename: str):
        """Write contexts to csv file.

        :param contexts: list of contexts
        :param filename: filename for csv file
        """
        with open(filename, 'wt') as f_p:
            for context in contexts:
                context_str = context.context
                label = context.label

                f_p.write(f'{label}\t{context_str}\n')
