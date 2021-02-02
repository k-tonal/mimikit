import torch.nn as nn
import numpy as np

from .modules import GatedLinearInput, AbsLinearOutput, mean_L1_prop
from .freq_layer import FreqLayer
from .base import FreqNetModel


class FreqNet(FreqNetModel):
    """
    auto-regressive model for learning and generating audio in the frequency domain.


    Parameters
    ----------
    loss_fn : function, optional
        the loss function to use for training.
        Must accept 2 arguments ``outputs`` and ``targets`` as returned by ``forward()`` and the batch respectively.
        Default is ``mimikit.freqnet.modules.mean_L1_prop``
    model_dim : int, optional
        the number of channels to use in the layers' parameters.
        Default is 512
    groups : int, optional
        the number of groups to use in the layers' convolutions.
        Default is 1.
        See `the torch documentation <>`__ for more information.
    n_layers : tuple of ints,
        each int ``n`` in ``n_layers`` represents a block of ``n`` layers.
        See the `Freqnet Guide` for more information.
    kernel_size : int, optional
        the size of the kernel in the convolutions.
        Default is 2
    strict : bool, optional
        whether to enforce layer-wise time-shifting.
        Default is ``False``.
        See the `Freqnet Guide` for more information.
    accum_outputs : int or str, optional
        commonly known as applying residuals : whether to sum the outputs of each layer with their respective inputs.
        See the note "On sides" for accepted values.
        Default is 0.
    concat_outputs : int or str, optional
        when turned on, this technique pads outputs of each layer with time-steps of the inputs from the left or right side.
        See the note "On sides" for accepted values.
        Default is 0.
    pad_input : int or str, optional
        whether and where to pad the inputs to each layer.
        See the note "On sides" for accepted values.
        Default is 0.
    learn_padding : bool, optional
        whether to use learnable padding parameters when ``pad_input`` is neither ``0`` nor ``None``
        Default is ``False``
    with_skip_conv : bool, optional
        whether to add skips convolutions and corresponding mechanism to the network.
        Default is ``False``
    with_residual_conv : bool, optional
        whether to add an additional 1x1 convolution after the gate of each layer.
        Default is False.
    data_optim_kwargs
        optional parameters for data and optimization submodules.
        See ``FreqData`` and ``FreqOptim`` for more information.

    Notes
    -----
    .. note::
        On sides.
        Because convolution layers as they are used in ``FreqNet`` outputs less time-steps than they receive as input,
        applying techniques such as residuals or padding is only possible when one specifies how inputs and outputs are
        supposed to be aligned.
        ``FreqNet`` has 3 arguments that turn such techniques on and off : ``pad_input``, ``accum_outputs`` and ``concat_outputs``.
        They all accept the same values :
            - ``0`` or ``None`` mean the technique is not applied.
            - ``1`` or ``"left"`` aligns outputs and inputs on the left and apply the technique.
            - ``-1`` or ``"right"`` aligns outputs and inputs on the right and apply the technique.

    """

    LAYER_KWARGS = ["groups", "strict", "accum_outputs", "concat_outputs", "kernel_size",
                    "pad_input", "learn_padding", "with_skip_conv", "with_residual_conv"]

    def __init__(self,
                 loss_fn=mean_L1_prop,
                 model_dim=512,
                 groups=1,
                 n_layers=(2,),
                 kernel_size=2,
                 strict=False,
                 accum_outputs=0,
                 concat_outputs=0,
                 pad_input=0,
                 learn_padding=False,
                 with_skip_conv=True,
                 with_residual_conv=True,
                 **data_optim_kwargs):
        super(FreqNet, self).__init__(**data_optim_kwargs)
        self._loss_fn = loss_fn
        self.model_dim = model_dim
        self.groups = groups
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.strict = strict
        self.accum_outputs = accum_outputs
        self.concat_outputs = concat_outputs
        self.pad_input = pad_input
        self.learn_padding = learn_padding
        self.with_skip_conv = with_skip_conv
        self.with_residual_conv = with_residual_conv

        # Input Encoder
        self.inpt = GatedLinearInput(self.input_dim, self.model_dim)

        # Auto-regressive Part
        layer_kwargs = {attr: getattr(self, attr) for attr in self.LAYER_KWARGS}
        # for simplicity we keep all the layers in a flat list
        self.layers = nn.ModuleList([
            FreqLayer(layer_index=i, input_dim=model_dim, layer_dim=model_dim, **layer_kwargs)
            for n_layers in self.n_layers for i in range(n_layers)
        ])

        # Output Decoder
        self.outpt = AbsLinearOutput(self.model_dim, self.input_dim)

        self.save_hyperparameters()

    def forward(self, x):
        """
        """
        x = self.inpt(x)
        skips = None
        for layer in self.layers:
            x, skips = layer(x, skips)
        x = self.outpt(skips)
        return x

    def loss_fn(self, predictions, targets):
        return self._loss_fn(predictions, targets)

    def all_rel_shifts(self):
        """sequence of shifts from one layer to the next"""
        return tuple(layer.rel_shift() for layer in self.layers)

    def shift(self):
        """total shift of the network"""
        if not self.strict and (self.pad_input == 1 or self.concat_outputs == 1):
            return 1
        elif self.pad_input == -1 or self.concat_outputs == -1:
            return self.receptive_field()
        elif self.strict and self.concat_outputs == 1:
            return 0
        elif self.strict and self.pad_input == 1:
            return len(self.layers)
        else:
            return sum(self.all_rel_shifts()) + int(not self.strict)

    def all_shifts(self):
        """the accumulated shift at each layer"""
        return tuple(np.cumsum(self.all_rel_shifts()) + int(not self.strict))

    def receptive_field(self):
        block_rf = []
        for i, layer in enumerate(self.layers[:-1]):
            if self.layers[i + 1].layer_index == 0:
                block_rf += [layer.receptive_field() - 1]
        block_rf += [self.layers[-1].receptive_field()]
        return sum(block_rf)

    def output_length(self, input_length):
        return self.all_output_lengths(input_length)[-1]

    def all_output_lengths(self, input_length):
        out_length = input_length
        lengths = []
        for layer in self.layers:
            out_length = layer.output_length(out_length)
            lengths += [out_length]
        return tuple(lengths)

    def targets_shifts_and_lengths(self, input_length):
        return [(self.shift(), self.output_length(input_length))]

    def generation_slices(self):
        # input is always the last receptive field
        input_slice = slice(-self.receptive_field(), None)
        if not self.strict and (self.pad_input == 1 or self.concat_outputs == 1):
            # then there's only one future time-step : the last output
            output_slice = slice(-1, None)
        elif self.strict and (self.pad_input == 1 or self.concat_outputs == 1):
            # then there are as many future-steps as they are layers and they all are
            # at the end of the outputs
            output_slice = slice(-len(self.layers), None)
        else:
            # for all other cases, the first output has the shift of the whole network
            output_slice = slice(None, 1)
        return input_slice, output_slice
