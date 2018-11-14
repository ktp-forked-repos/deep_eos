"""Implement various methods of dropouts.

This class is heavily taken from Zalando's flair library:
<https://github.com/zalandoresearch/flair/blob/master/flair/nn.py>
"""

import torch.nn


class CharacterDropout(torch.nn.Module):
    """Implement character dropout.

    This dropout method randomly drops out entire characters in embedding space.
    """

    def __init__(self, dropout_rate=0.05):
        """Define constructor of character dropout class.

        :param dropout_rate: dropout rate
        """
        super(CharacterDropout, self).__init__()
        self.dropout_rate = dropout_rate

    def forward(self, x):  # pylint: disable=arguments-differ
        """Implement forward pass of character dropout class.

        :param x: context embedding
        :return: masked context embedding
        """
        if not self.training or not self.dropout_rate:
            return x

        masked_data = x.data.new(x.size(0), 1, 1).bernoulli_(1 - self.dropout_rate)
        mask = torch.autograd.Variable(masked_data, requires_grad=False)
        mask = mask.expand_as(x)  # pylint: disable=no-member # noqa: E501

        return mask * x


class LockedDropout(torch.nn.Module):
    """Implement locked or variational dropout.

    This dropout method randomly drops out entire parameters in embedding space.
    """

    def __init__(self, dropout_rate=0.5):
        """Define constructor of locked dropout class.

        :param dropout_rate: dropout rate
        """
        super(LockedDropout, self).__init__()
        self.dropout_rate = dropout_rate

    def forward(self, x):  # pylint: disable=arguments-differ
        """Implement forward pass of locked dropout class.

        :param x: context embedding
        :return: masked input context embedding
        """
        if not self.training or not self.dropout_rate:
            return x

        masked_data = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - self.dropout_rate)
        mask = torch.autograd.Variable(masked_data, requires_grad=False) / (1 - self.dropout_rate)
        mask = mask.expand_as(x)  # pylint: disable=no-member # noqa: E501
        return mask * x
