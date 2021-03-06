"""Define trainer methods for deep_eos."""

import logging
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F  # pylint: disable=import-error,no-name-in-module
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

from deep_eos.data import Corpus, Context
from deep_eos.models.deep_eos_lstm_model import DeepEos

LOG = logging.getLogger(__name__)


class DeepEosTrainer:
    """Implement deep_eos trainer class."""

    def __init__(self, model: DeepEos, corpus: Corpus):
        """Define eos trainer class constructor.

        :param model: deep_eos model
        :param corpus: corpus incl. various datasets
        """
        self.model = model
        self.corpus = corpus
        self.loss_fn = F.cross_entropy

    def clip_gradient(self, clip_value):
        """Implement gradien clipping method.

        :param clip_value: clip value
        """
        params = list(filter(lambda p: p.grad is not None, self.model.parameters()))
        for param in params:
            param.grad.data.clamp_(-clip_value, clip_value)

    def evaluate_dev_set(self, contexts: List[Context]):
        """Evaluate model on development set.

        :param contexts: development contexts
        :return: accuracy on development set
        """
        counter = 0
        correct = 0

        with torch.no_grad():
            for context in contexts:
                context_str = self.corpus.context_fields.preprocess(context.context)
                context_str = [[self.corpus.context_fields.vocab.stoi[s]
                                for s in context_str]]

                context_array = np.asarray(context_str)
                context_array = torch.LongTensor(context_array)  # pylint: disable=no-member

                context_var = Variable(context_array)

                if torch.cuda.is_available():
                    context_var = context_var.cuda()  # pylint: disable=no-member

                self.model.eval()

                output = self.model(context_var, 1)
                _, predicted = torch.max(output, 1)  # pylint: disable=no-member
                current_label = self.corpus.label_fields.vocab.itos[predicted.data.item()]

                if int(current_label) == int(context.label):
                    correct += 1

                counter += 1

        return correct / counter

    def train(self, model_file: Path,  # pylint: disable=too-many-arguments,too-many-locals
              learning_rate: float = 0.001,
              batch_size: int = 32,
              epochs: int = 10,
              anneal_factor: float = 0.5,
              patience: int = 3,
              checkpoint: bool = False
              ):
        """Train model for full epochs.

        :param model_file: filename for saving best trained model
        :param learning_rate: learning rate for adam optimizer
        :param batch_size: batch size
        :param epochs: number of epochs to be trained
        :param anneal_factor: factor by which the learning rate will be reduced
        :param patience: number of epochs with no improvement after which lr will be reduced
        :param checkpoint: defines if checkpoint file should be written after every epoch
        """
        best_acc = 0.0

        LOG.info('Start training for %s epochs', epochs)

        if torch.cuda.is_available():
            self.model.cuda()

        optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                 lr=learning_rate)

        anneal_mode = 'max'
        scheduler = ReduceLROnPlateau(optim, factor=anneal_factor, patience=patience,
                                      mode=anneal_mode, verbose=True)

        for epoch in range(epochs):
            total_epoch_loss = 0
            total_epoch_acc = 0

            for group in optim.param_groups:
                learning_rate = group['lr']

            steps = 0

            if learning_rate < 0.0001:
                logging.info("Learning rate too small - exiting training!")
                break

            self.model.train()

            for _, batch in enumerate(self.corpus.train_iter):
                text = batch.text[0]
                target = batch.label
                target = torch.autograd.Variable(target).long()  # pylint: disable=no-member
                if torch.cuda.is_available():
                    text = text.cuda()
                    target = target.cuda()

                if text.size()[0] is not batch_size:
                    continue

                optim.zero_grad()

                prediction = self.model(text)

                loss = self.loss_fn(prediction, target)

                num_corrects = (torch.max(prediction, 1)[1].view(  # pylint: disable=no-member # noqa: E501
                    target.size()).data == target.data).float().sum()
                acc = num_corrects / len(batch)
                loss.backward()
                self.clip_gradient(1e-1)
                optim.step()
                steps += 1
                total_epoch_loss += loss.item()
                total_epoch_acc += acc.item()

            train_loss = total_epoch_loss / len(self.corpus.train_iter)
            train_acc = total_epoch_acc / len(self.corpus.train_iter)

            val_acc = self.evaluate_dev_set(self.corpus.dev_contexts)

            scheduler.step(val_acc)

            LOG.info('Epoch: %s, Train Loss: %s, Train Acc: %s, Val. Acc: %s, Learning Rate: %s',
                     epoch + 1, round(train_loss, 4), round(train_acc, 4), round(val_acc, 4),
                     learning_rate)

            if val_acc > best_acc:
                LOG.info('Save new best model...')
                self.model.save(model_file)
                best_acc = val_acc

            if checkpoint:
                self.model.save_checkpoint(Path('./checkpoint'),
                                           optim.state_dict(), scheduler.state_dict(),
                                           epoch + 1, train_loss)

            LOG.info('-' * 100)
