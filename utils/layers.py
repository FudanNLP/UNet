#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : layers.py
# Author            : Sun Fu <cstsunfu@gmail.com>
# Date              : 23.06.2018
# Last Modified Date: 05.11.2018
# Last Modified By  : Sun Fu <cstsunfu@gmail.com>
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, Linear
from typing import Callable

from torch.autograd import Variable

class StackedBRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 dropout=0, concat=True, use_cuda = True, dropout_output=False, rnn_type=nn.LSTM,
                 padding=True):
        super(StackedBRNN, self).__init__()
        self.padding = padding
        self.dropout_output = dropout_output
        self.dropout = dropout
        self.num_layers = num_layers
        self.concat = concat
        self.rnns = nn.ModuleList()
        for i in range(num_layers):
            input_size = input_size if i == 0 else 2 * hidden_size
            self.rnns.append(rnn_type(input_size, hidden_size,
                                      num_layers=1,
                                      bidirectional=True))

    def forward(self, x, x_mask):
        """Can choose to either handle or ignore variable length sequences.
        Always handle padding in eval.
        """
        # No padding necessary.
        if x_mask.data.sum() == 0:
            return self._forward_unpadded(x, x_mask)
        # Pad if we care or if its during eval.
        if self.padding or not self.training:
            return self._forward_padded(x, x_mask)
        # We don't care.
        return self._forward_unpadded(x, x_mask)

    def _forward_unpadded(self, x, x_mask):
        """Faster encoding that ignores any padding."""
        # Transpose batch and sequence dims
        x = x.transpose(0, 1)

        # Encode all layers
        outputs = [x]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            # Apply dropout to hidden input
            if self.dropout > 0:
                rnn_input = F.dropout(rnn_input,
                                      p=self.dropout,
                                      training=self.training)
            # Forward
            rnn_output = self.rnns[i](rnn_input)[0]
            outputs.append(rnn_output)

        # Concat hidden layers
        if self.concat:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

        # Transpose back
        output = output.transpose(0, 1)

        # Dropout on output layer
        if self.dropout_output and self.dropout > 0:
            output = F.dropout(output,
                               p=self.dropout,
                               training=self.training)
        return output

    def _forward_padded(self, x, x_mask):
        """Slower (significantly), but more precise,
        encoding that handles padding."""
        # Compute sorted sequence lengths
        lengths = x_mask.data.eq(0).long().sum(1).squeeze()
        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        lengths = list(lengths[idx_sort])
        idx_sort = Variable(idx_sort)
        idx_unsort = Variable(idx_unsort)

        # Sort x
        x = x.index_select(0, idx_sort)

        # Transpose batch and sequence dims
        x = x.transpose(0, 1)

        # Pack it up
        rnn_input = nn.utils.rnn.pack_padded_sequence(x, lengths)

        # Encode all layers
        outputs = [rnn_input]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            # Apply dropout to input
            if self.dropout > 0:
                dropout_input = F.dropout(rnn_input.data,
                                          p=self.dropout,
                                          training=self.training)
                rnn_input = nn.utils.rnn.PackedSequence(dropout_input,
                                                        rnn_input.batch_sizes)
            outputs.append(self.rnns[i](rnn_input)[0])

        # Unpack everything
        for i, o in enumerate(outputs[1:], 1):
            outputs[i] = nn.utils.rnn.pad_packed_sequence(o)[0]

        # Concat hidden layers or take final
        if self.concat:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

        # Transpose and unsort
        output = output.transpose(0, 1)
        output = output.index_select(0, idx_unsort)

        # Pad up to original batch sequence length
        if output.size(1) != x_mask.size(1):
            padding = torch.zeros(output.size(0),
                                  x_mask.size(1) - output.size(1),
                                  output.size(2)).type(output.data.type())
            output = torch.cat([output, Variable(padding)], 1)

        # Dropout on output layer
        if self.dropout_output and self.dropout > 0:
            output = F.dropout(output,
                               p=self.dropout,
                               training=self.training)
        return output


class CharEmbeddingLayer(nn.Module):
    def __init__(self, char_single_embedding_dim, char_embedding_dim, char_embedding):
        super(CharEmbeddingLayer, self).__init__()
        self.char_single_embedding_dim = char_single_embedding_dim
        self.char_embedding_dim = char_embedding_dim

        self.embedding_lookup = nn.Embedding(char_embedding.size(0), char_embedding.size(1), padding_idx=0)
        self.cnn = CNN(embedding_dim=char_single_embedding_dim, num_filters=char_embedding_dim)

    def forward(self, text_char, text_char_mask):
        batch_size, max_length, max_word_length = text_char.size()
        # embedding look up
        text_char = text_char.contiguous().view(batch_size * max_length, max_word_length)
        text_char_mask = text_char_mask.view(batch_size * max_length, max_word_length)

        text_char = self.embedding_lookup(text_char)
        text_char = self.cnn(text_char, text_char_mask)
        text_char = text_char.contiguous().view(batch_size * max_length * self.char_embedding_dim, -1)
        text_char = nn.functional.relu(text_char)

        text_char = torch.max(text_char, 1)[0]
        text_char = text_char.contiguous().view(batch_size, max_length, self.char_embedding_dim)
        return text_char


class CNN(nn.Module):

    """Docstring for CNN"""

    def __init__(self, embedding_dim, num_filters=100, ngram_filter_sizes=(2, 3, 4, 5), conv_layer_activation='relu', output_dim=None):
        """TODO: to be defined1. """
        nn.Module.__init__(self)

        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.ngram_filter_sizes = ngram_filter_sizes
        self.conv_layer_activation = torch.nn.functional.relu
        self.output_dim = output_dim

        self.convolution_layers = nn.ModuleList()
        for ngram_size in self.ngram_filter_sizes:
            self.convolution_layers.append(Conv1d(in_channels=self.embedding_dim,
                                                  out_channels=self.num_filters,
                                                  kernel_size=ngram_size))

        maxpool_output_dim = self.num_filters * len(self.ngram_filter_sizes)
        if self.output_dim:
            self.projection_layer = Linear(maxpool_output_dim, self.output_dim)
        else:
            self.projection_layer = None
            self.output_dim = maxpool_output_dim

    def forward(self, tokens, mask):
        """TODO: Docstring for forward.

        :tokens: TODO
        :mask: TODO
        :returns: TODO

        """

        tokens = torch.transpose(tokens, 1, 2)

        filter_outputs = []
        for convolution_layer in self.convolution_layers:
            filter_outputs.append(
                self.conv_layer_activation(convolution_layer(tokens)).max(dim=2)[0]
            )

        maxpool_output = torch.cat(filter_outputs, dim=1) if len(filter_outputs) > 1 else filter_outputs[0]

        if self.projection_layer:
            result = self.projection_layer(maxpool_output)
        else:
            result = maxpool_output
        return result


class Highway(torch.nn.Module):
    """
    A `Highway layer <https://arxiv.org/abs/1505.00387>`_ does a gated combination of a linear
    transformation and a non-linear transformation of its input.  :math:`y = g * x + (1 - g) *
    f(A(x))`, where :math:`A` is a linear transformation, :math:`f` is an element-wise
    non-linearity, and :math:`g` is an element-wise gate, computed as :math:`sigmoid(B(x))`.

    This module will apply a fixed number of highway layers to its input, returning the final
    result.

    Parameters
    ----------
    input_dim : ``int``
        The dimensionality of :math:`x`.  We assume the input has shape ``(batch_size,
        input_dim)``.
    num_layers : ``int``, optional (default=``1``)
        The number of highway layers to apply to the input.
    activation : ``Callable[[torch.Tensor], torch.Tensor]``, optional (default=``torch.nn.functional.relu``)
        The non-linearity to use in the highway layers.
    """
    def __init__(self,
                 input_dim: int,
                 num_layers: int = 2,
                 activation: Callable[[torch.Tensor], torch.Tensor] = torch.nn.functional.relu) -> None:
        super(Highway, self).__init__()
        self._input_dim = input_dim
        self._layers = torch.nn.ModuleList([torch.nn.Linear(input_dim, input_dim * 2)
                                            for _ in range(num_layers)])
        self._activation = activation
        for layer in self._layers:
            # We should bias the highway layer to just carry its input forward.  We do that by
            # setting the bias on `B(x)` to be positive, because that means `g` will be biased to
            # be high, to we will carry the input forward.  The bias on `B(x)` is the second half
            # of the bias vector in each Linear layer.
            layer.bias[input_dim:].data.fill_(1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # pylint: disable=arguments-differ
        current_input = inputs
        for layer in self._layers:
            projected_input = layer(current_input)
            linear_part = current_input
            # NOTE: if you modify this, think about whether you should modify the initialization
            # above, too.
            nonlinear_part = projected_input[:, (0 * self._input_dim):(1 * self._input_dim)]
            gate = projected_input[:, (1 * self._input_dim):(2 * self._input_dim)]
            nonlinear_part = self._activation(nonlinear_part)
            gate = torch.nn.functional.sigmoid(gate)
            if self.training:
                epsilon = (torch.rand(gate.size()) - 0.5) / 10.0
                epsilon = Variable(epsilon.cuda(gate.get_device()))
                current_input = gate * linear_part + (1 - gate + epsilon) * nonlinear_part
            else:
                current_input = gate * linear_part + (1 - gate) * nonlinear_part
        return current_input


class TimeDistributed(torch.nn.Module):
    """
    Given an input shaped like ``(batch_size, time_steps, [rest])`` and a ``Module`` that takes
    inputs like ``(batch_size, [rest])``, ``TimeDistributed`` reshapes the input to be
    ``(batch_size * time_steps, [rest])``, applies the contained ``Module``, then reshapes it back.

    Note that while the above gives shapes with ``batch_size`` first, this ``Module`` also works if
    ``batch_size`` is second - we always just combine the first two dimensions, then split them.
    """
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self._module = module

    def forward(self, *inputs):  # pylint: disable=arguments-differ
        reshaped_inputs = []
        for input_tensor in inputs:
            input_size = input_tensor.size()
            if len(input_size) <= 2:
                raise RuntimeError("No dimension to distribute: " + str(input_size))

            # Squash batch_size and time_steps into a single axis; result has shape
            # (batch_size * time_steps, input_size).
            squashed_shape = [-1] + [x for x in input_size[2:]]
            reshaped_inputs.append(input_tensor.contiguous().view(*squashed_shape))

        reshaped_outputs = self._module(*reshaped_inputs)

        # Now get the output back into the right shape.
        # (batch_size, time_steps, [hidden_size])
        new_shape = [input_size[0], input_size[1]] + [x for x in reshaped_outputs.size()[1:]]
        outputs = reshaped_outputs.contiguous().view(*new_shape)

        return outputs


class FullAttention(nn.Module) :

    def __init__(self, input_size, hidden_size, dropout, use_cuda = True):
        super(FullAttention, self).__init__()
        self.use_cuda = use_cuda
        self.dropout = dropout
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.U = nn.Linear(input_size, hidden_size, bias=False)
        self.D = nn.Parameter(torch.ones(1, hidden_size), requires_grad=True)

        self.init_weights()

    def init_weights(self) :
        nn.init.xavier_uniform_(self.U.weight.data)

    def forward(self, passage, p_mask, question, q_mask, rep, rep_p, is_training):

        if is_training :
            keep_prob = 1.0 - self.dropout
            drop_mask = Dropout(passage, self.dropout, is_training, return_mask=True, use_cuda = self.use_cuda)
            d_passage = torch.div(passage, keep_prob) * drop_mask
            d_ques = torch.div(question, keep_prob) * drop_mask
        else :
            d_passage = passage
            d_ques = question

        Up = F.relu(self.U(d_passage))
        Uq = F.relu(self.U(d_ques))
        D = self.D.expand_as(Uq)

        Uq = D * Uq

        scores = Up.bmm(Uq.transpose(2, 1))

        output_q = None
        if rep_p is not None:
            scores_T = scores.clone().transpose(1, 2)
            mask_p = p_mask.unsqueeze(1).repeat(1, question.size(1), 1)
            scores_T.data.masked_fill_(mask_p.data, -float('inf'))
            alpha_p = F.softmax(scores_T, 2)
            output_q = torch.bmm(alpha_p, rep_p)

        mask = q_mask.unsqueeze(1).repeat(1, passage.size(1), 1)
        scores.data.masked_fill_(mask.data, -float('inf'))
        alpha = F.softmax(scores, 2)
        output = torch.bmm(alpha, rep)

        return output, output_q


class WordAttention(nn.Module) :

    def __init__(self, input_size, hidden_size, dropout, use_cuda = True) :
        super(WordAttention, self).__init__()
        self.use_cuda = use_cuda
        self.dropout = dropout
        self.hidden_size = hidden_size

        self.W = nn.Linear(input_size, hidden_size)
        self.init_weights()

    def init_weights(self) :
        nn.init.xavier_uniform_(self.W.weight.data)
        self.W.bias.data.fill_(0.1)

    def forward(self, passage, p_mask, question, q_mask, is_training):

        if is_training :
            keep_prob = 1.0 - self.dropout
            drop_mask = Dropout(passage, self.dropout, is_training, return_mask = True, use_cuda = self.use_cuda)
            d_passage = torch.div(passage, keep_prob) * drop_mask
            d_ques = torch.div(question, keep_prob) * drop_mask
        else :
            d_passage = passage
            d_ques = question

        Wp = F.relu(self.W(d_passage))
        Wq = F.relu(self.W(d_ques))

        scores = torch.bmm(Wp, Wq.transpose(2, 1))

        mask = q_mask.unsqueeze(1).repeat(1, passage.size(1), 1)
        scores.data.masked_fill_(mask.data, -float('inf'))
        alpha = F.softmax(scores, dim=2)
        output = torch.bmm(alpha, question)

        return output


class Summ(nn.Module) :

    def __init__(self, input_size, dropout, use_cuda = True) :
        super(Summ, self).__init__()
        self.use_cuda = use_cuda
        self.dropout = dropout
        self.w = nn.Linear(input_size, 1)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.w.weight.data)
        self.w.bias.data.fill_(0.1)

    def forward(self, x, mask, is_training) :

        d_x = Dropout(x, self.dropout, is_training, use_cuda = self.use_cuda)
        beta = self.w(d_x).squeeze(2)
        beta.data.masked_fill_(mask.data, -float('inf'))
        beta = F.softmax(beta, 1)
        output = torch.bmm(beta.unsqueeze(1), x).squeeze(1)
        return output


class Answer(nn.Module):
    """A bilinear attention layer over a sequence X w.r.t y:
    * o_i = softmax(x_i'Wy) for x_i in X.

    Optionally don't normalize output weights.
    """
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}

    def __init__(self, x_size, y_size, opt, first=False, check=False):
        super(Answer, self).__init__()
        self.linear = nn.Linear(y_size, x_size)
        # self.linearx = nn.Linear(x_size, x_size)
        self.first = first
        self.rnn = nn.GRUCell(x_size, y_size)
        self.opt = opt
        self.check = check
        if self.first:
            self.rnn = nn.GRUCell(x_size, y_size)
        if self.check is True:
            self.doc_rnn = StackedBRNN(input_size = 2 * x_size,
                                       hidden_size = opt['hidden_size'],
                                       num_layers = 1,
                                       dropout = opt['dropout'])
        self.init_weights()

    def init_weights(self) :
        nn.init.xavier_uniform_(self.linear.weight.data)
        self.linear.bias.data.fill_(0.1)

    def forward(self, x, y, x_mask):
        """
        x = batch * len * h1
        y = batch * h2
        x_mask = batch * len
        """
        d_y = F.dropout(y, p=self.opt['dropout'], training=self.training)
        yW = self.linear(d_y)
        # xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)
        xWy = yW.unsqueeze(1).bmm(x.transpose(2, 1)).squeeze(1)
        xWy.data.masked_fill_(x_mask.data, -float('inf'))

        if self.first or self.check:
            alpha = F.softmax(xWy, dim=1)
        # x_wave = alpha.unsqueeze(1).bmm(x).squeeze(1)
        # y_new = self.rnn(x_wave, y)
        y_new = None
        x_new = None
        x_check = None
        if self.first:
            rnn_input = torch.bmm(alpha.unsqueeze(1), x).squeeze(1)
            rnn_input = F.dropout(rnn_input, p=self.opt['dropout'], training=self.training)
            y_new = self.rnn(rnn_input, y)

        if self.check is True:
            x_check = alpha.unsqueeze(2).expand(x.size(0), x.size(1), x.size(2)) * x
            x_new = torch.cat((x_check, x), 2)
            x_new = self.doc_rnn(x_new, x_mask)
            xWy = None

        return xWy, x_new, y_new, x_check


class PointerNet(nn.Module) :

    def __init__(self, input_size, opt, use_cuda = True) :
        super(PointerNet, self).__init__()
        self.opt = opt
        self.input_size = input_size
        if opt['check_answer']:
            self.check = Answer(input_size, input_size, opt, first=False, check=True)
        self.start = Answer(input_size, input_size, opt, first=True, check=False)
        self.end = Answer(input_size, input_size, opt, first=False, check=False)

    def forward(self, self_states, p_mask, init_states, q_summ, is_training) :

        x_check = None
        if self.opt['check_answer']:
            _, self_states, _, x_check = self.check(self_states, init_states, p_mask)

        logits1, _, init_states, _ = self.start(self_states, q_summ, p_mask)
        logits2, _, _, _ = self.end(self_states, q_summ, p_mask)

        return logits1, logits2, x_check


def Dropout(x, dropout, is_train, return_mask = False, var=True, use_cuda=True) :

    if not var :
        return F.dropout(x, dropout, is_train)

    if dropout > 0.0 and is_train :
        shape = x.size()
        keep_prob = 1.0 - dropout
        random_tensor = keep_prob
        tmp = Variable(torch.FloatTensor(shape[0], 1, shape[2]))
        if use_cuda :
            tmp = tmp.cuda()
        nn.init.uniform_(tmp)
        random_tensor += tmp
        binary_tensor = torch.floor(random_tensor)
        x = torch.div(x, keep_prob) * binary_tensor

    if return_mask :
        return binary_tensor

    return x
