"""Implement various methods for deep eos model using LSTM."""

import logging
import sys
from pathlib import Path
from typing import Set, Union

import numpy as np
import torch
from torch.autograd import Variable
from torchtext.vocab import Vocab

from deep_eos.dropout import CharacterDropout, LockedDropout
from deep_eos.file_utils import get_cached_path
from deep_eos.utils import EOS_CHARS
from deep_eos.utils import get_char_context

LOG = logging.getLogger(__name__)


class DeepEos(torch.nn.Module):  # pylint: disable=too-many-instance-attributes
    """Implement deep eos model using LSTM."""

    def __init__(self,  # pylint: disable=too-many-arguments
                 vocab: Vocab,
                 label_vocab: Vocab,
                 batch_size: int = 32,
                 hidden_size: int = 256,
                 embedding_length: int = 256,
                 output_size: int = 2,
                 use_dropout: float = 0.0,
                 use_character_dropout: float = 0.05,
                 use_locked_dropout: float = 0.5):
        """Define constructor for DeepEos class.

        :param vocab: vocabulary
        :param label_vocab: vocabulary for labels
        :param batch_size: batch size
        :param hidden_size: hidden size of LSTM
        :param embedding_length: embedding size of LSTM
        :param output_size: output size (normally 2: EOS and NON-EOS)
        :param use_dropout: define dropout
        :param use_character_dropout: define character dropout
        :param use_locked_dropout: define locked dropout
        """
        super(DeepEos, self).__init__()

        self.vocab = vocab
        self.label_vocab = label_vocab
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.embedding_length = embedding_length
        self.output_size = output_size

        self.embeddings = torch.nn.Embedding(len(self.vocab), self.embedding_length)
        self.lstm = torch.nn.LSTM(self.embedding_length, self.hidden_size)
        self.label = torch.nn.Linear(self.hidden_size, self.output_size)

        self.use_dropout: float = use_dropout
        self.use_character_dropout: float = use_character_dropout
        self.use_locked_dropout: float = use_locked_dropout

        if use_dropout > 0.0:
            self.dropout = torch.nn.Dropout(use_dropout)

        if use_character_dropout > 0.0:
            self.character_dropout = CharacterDropout(use_character_dropout)

        if use_locked_dropout > 0.0:
            self.locked_dropout = LockedDropout(use_locked_dropout)

        if torch.cuda.is_available():
            self.cuda()

    def save(self, model_file: Path):
        """Save PyTorch model to file.

        :param model_file: model filename
        """
        model_state = {
            'state_dict': self.state_dict(),
            'vocab': self.vocab,
            'label_vocab': self.label_vocab,
            'batch_size': self.batch_size,
            'embeddings': self.embeddings,
            'hidden_size': self.hidden_size,
            'embedding_length': self.embedding_length,
            'output_size': self.output_size
        }
        torch.save(model_state, str(model_file), pickle_protocol=4)
        LOG.info('Model successfully saved to %s', str(model_file))

    def save_checkpoint(self, model_file: Path, optimizer_state: dict, scheduler_state: dict,  # pylint: disable=too-many-arguments # noqa: E501
                        epoch: int, loss: float):
        """Save PyTorch model to checkpoint file.

        :param model_file: model filename
        :param optimizer_state: optimizer state
        :param scheduler_state: lr scheduler state
        :param epoch: current epoch
        :param loss: current loss
        """
        model_state = {
            'state_dict': self.state_dict(),
            'vocab': self.vocab,
            'label_vocab': self.label_vocab,
            'batch_size': self.batch_size,
            'embeddings': self.embeddings,
            'hidden_size': self.hidden_size,
            'embedding_length': self.embedding_length,
            'output_size': self.output_size,
            'optimizer_state_dict': optimizer_state,
            'scheduler_state_dict': scheduler_state,
            'epoch': epoch,
            'loss': loss
        }
        model_file_name = f'{model_file}-{epoch}.pt'
        torch.save(model_state, str(model_file_name), pickle_protocol=4)
        LOG.info('Checkpoint successfully saved to %s', str(model_file_name))

    @classmethod
    def load_from_file(cls, model_file: Path) -> "DeepEos":
        """Load model from file.

        :param model_file: model filename to be loaded
        :return: model
        """
        state = torch.load(model_file, map_location={'cuda:0': 'cpu'})

        use_dropout = 0.0 if 'use_dropout' not in state.keys() \
            else state['use_dropout']

        model = DeepEos(vocab=state['vocab'],
                        label_vocab=state['label_vocab'],
                        batch_size=state['batch_size'],
                        hidden_size=state['hidden_size'],
                        embedding_length=state['embedding_length'],
                        output_size=state['output_size'],
                        use_dropout=use_dropout)

        model.load_state_dict(state['state_dict'])
        model.eval()

        if torch.cuda.is_available():
            model = model.cuda()

        LOG.info('Model successfully loaded from %s', str(model_file))

        return model

    @classmethod
    def load_from_cloud(cls, language: str) -> Union[None, "DeepEos"]:
        """Download model from cloud and returns it.

        :param language: language code for model
        :return: model
        """
        model_path = get_cached_path(language=language)

        model = None

        if model_path:
            model = cls.load_from_file(model_path)

        return model

    def forward(self, input_context, batch_size=None):  # pylint: disable=arguments-differ
        """Define computation performed at every call.

        :param input_context: context for potential end-of-sentence
        :param batch_size: batch size
        :return: final output for forward pass
        """
        input_ = self.embeddings(input_context)
        input_ = input_.permute(1, 0, 2)

        if self.use_dropout > 0.0:
            input_ = self.dropout(input_)

        if self.use_character_dropout > 0.0:
            input_ = self.character_dropout(input_)

        if self.use_locked_dropout > 0.0:
            input_ = self.locked_dropout(input_)

        batch_size = self.batch_size if batch_size is None else batch_size

        if torch.cuda.is_available():
            h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())  # pylint: disable=no-member # noqa: E501
            c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())  # pylint: disable=no-member # noqa: E501
        else:
            h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size))  # pylint: disable=no-member # noqa: E501
            c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size))  # pylint: disable=no-member # noqa: E501

        _, (final_hidden_state, _) = self.lstm(input_, (h_0, c_0))
        final_output = self.label(final_hidden_state[-1])

        return final_output

    def predict(self, buffer: str, left_ws: int, right_ws: int, eos_marker='</eos>', io_stream=sys.stdout):  # pylint: disable=too-many-locals # noqa: E501
        """Predict and eos tag input buffer.

        :param buffer: text buffer
        :param left_ws: left window size (context before eos marker)
        :param right_ws: right window size (context after eos marker)
        :param eos_marker: eos marker (like </eos>) used for highlighting an end-of-sentence
        """
        contexts = get_char_context(left_window=left_ws,
                                    right_window=right_ws,
                                    buffer=buffer)

        found_eos_positions: Set[int] = set()

        with torch.no_grad():
            for context in contexts:
                context_str = context.context
                position = context.position

                context_str = list(context_str)
                context_str = [[self.vocab.stoi[s] for s in context_str]]

                context_array = np.asarray(context_str)
                context_array = torch.LongTensor(context_array)  # pylint: disable=no-member # noqa: E501

                context_var = Variable(context_array)

                if torch.cuda.is_available():
                    context_var = context_var.cuda()  # pylint: disable=no-member # noqa: E501

                self.eval()

                output = self(context_var, 1)
                _, predicted = torch.max(output, 1)  # pylint: disable=no-member # noqa: E501
                label = self.label_vocab.itos[predicted.data.item()]

                if int(label) == 1:
                    found_eos_positions.add(position)

        for pos, char in enumerate(buffer):
            io_stream.write(char)

            if pos in found_eos_positions:
                found_eos_positions.remove(pos)
                io_stream.write(eos_marker)

        if buffer.rstrip()[-1] in EOS_CHARS:
            io_stream.write(eos_marker)
