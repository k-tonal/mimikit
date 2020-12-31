import torch
import torch.nn as nn
import numpy as np
import math

from .modules import GatedConv, LearnablePad1d


def accum(x, y, shift=1):
    if shift == 0:
        if x.size(-1) == y.size(-1):
            return x + y
        else:
            raise ValueError("zero shift with size: %i and %i" % (x.size(-1), y.size(-1)))
    side = np.sign(shift)
    aligned_x = slice(*((shift, None)[::side]))
    n_aligned = x.size(-1) - abs(shift)
    aligned_y = slice(*((None, n_aligned * side)[::side]))
    compl_y = slice(*((n_aligned * side, None)[::side]))
    aligned = x[:, :, aligned_x] + y[:, :, aligned_y]
    rest = y[:, :, compl_y]
    return torch.cat((aligned, rest)[::side], dim=-1)


def concat(x, y, shift=1):
    if shift == 0:
        if x.size(-1) != y.size(-1):
            return concat(x, y, - x.size(-1) - y.size(-1))
        return y
    side = np.sign(shift)
    compl_x = slice(*((None, shift)[::side]))
    return torch.cat((x[:, :, compl_x], y)[::side], dim=-1)


class FreqLayer(nn.Module):
    kernel_size = 2
    stride = 1
    bias = True
    sides = {1: 1, -1: -1, 0: 0, "left": 1, "right": -1, None: 0}

    def __init__(self,
                 layer_index,
                 input_dim=512,
                 layer_dim=512,
                 groups=1,
                 strict=False,
                 accum_outputs=None,
                 concat_outputs=None,
                 pad_input=None,
                 learn_padding=False,
                 with_skip_conv=False,
                 with_residual_conv=False,
                 ):
        super(FreqLayer, self).__init__()
        self.layer_index = layer_index
        self.input_dim = input_dim
        self.layer_dim = layer_dim
        self.groups = groups
        self.strict = strict
        self.accum_outputs = self.sides.get(accum_outputs, 0)
        self.concat_outputs = self.sides.get(concat_outputs, 0)
        self.pad_input = self.sides.get(pad_input, 0)
        self.learn_padding = learn_padding
        self.with_skip_conv = with_skip_conv
        self.with_residual_conv = with_residual_conv

        self.pad = LearnablePad1d(self.input_dim, self.padding, self.learn_padding)

        convs_kwargs = dict(dilation=self.dilation, stride=self.stride,
                            groups=self.groups, bias=self.bias)
        self.gate = GatedConv(self.input_dim, self.layer_dim, self.kernel_size, **convs_kwargs)

        self.skips = nn.Conv1d(self.layer_dim, self.layer_dim, kernel_size=1, **convs_kwargs) \
            if self.with_skip_conv else None
        self.residuals = nn.Conv1d(self.layer_dim, self.input_dim, kernel_size=1, **convs_kwargs) \
            if self.with_residual_conv else None

    def forward(self, x, skip=None):
        x = self.pad(x)

        y = self.gate(x)
        h = self.skips(y) if self.with_skip_conv else None
        y = self.residuals(y) if self.with_residual_conv else y

        accum_outputs, concat_outputs = self.accum_outputs, self.concat_outputs
        shift = self.rel_shift()

        if accum_outputs:
            y = accum(x, y, shift * accum_outputs)
            if skip is None and h is not None:
                skip = torch.zeros_like(h).to(h)
                skip = accum(skip, h, shift * accum_outputs)

        if concat_outputs:
            y = concat(x, y, shift * concat_outputs)
            if skip is None and h is not None:
                skip = torch.zeros_like(h).to(h)
                skip = concat(skip, h, shift * concat_outputs)

        if self.with_skip_conv:
            if skip is None or (not concat_outputs and not accum_outputs):
                skip = h
        else:
            skip = y

        return y, skip

    @property
    def dilation(self):
        return 2 ** self.layer_index

    @property
    def padding(self):
        """
        signed amount of padding necessary to output as many time-steps as there were in the inputs.
        The sign of padding corresponds to the desired side (1: left, -1: right)
        """
        return self.pad_input * (self.kernel_size - 1) * self.dilation

    def receptive_field(self):
        """
        amount of inputs necessary for 1 output (this is independent of the shift!)
        """
        return 2 ** (self.layer_index + 1)

    def shift(self):
        """total shift at this layer wrt. to the begining of its block"""
        return self.receptive_field() + self.layer_index * int(self.strict)

    def rel_shift(self):
        """relative shift at this layer wrt. to the previous layer"""
        return int(self.strict) + self.receptive_field() // 2

    def output_length(self, input_length):
        if abs(self.concat_outputs):
            # concats appends past steps which grow by 1 at each layer when strict=True
            return input_length + int(self.strict)
        if abs(self.pad_input):
            # no matter what, padding input gives the same output shape
            return input_length
        # output is gonna be less than input
        numerator = input_length + 2 * abs(self.padding) - self.dilation * (self.kernel_size - 1) - 1
        denominator = self.stride
        return math.floor(1 + numerator / denominator)