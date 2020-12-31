import torch
import torch.nn as nn

from .freqnet import FreqNet
from .modules import mean_L1_prop


class HKFreqNet(FreqNet):
    """
    Add a hierarchical quality to the conditioning sequence by combining FreqLayers in a
    final Conv2d making depth in the network proportional to the distance to the predicted time-step.

    N.B : Could be fun to try the other way around : the closer to now -> the deeper in the net.
    """

    def __init__(self,
                 loss_fn=mean_L1_prop,
                 model_dim=512,
                 groups=1,
                 n_layers=2,
                 strict=False,
                 accum_outputs=None,
                 concat_outputs=None,
                 pad_input=None,
                 learn_padding=False,
                 with_skip_conv=True,
                 with_residual_conv=True,
                 **data_optim_kwargs):
        super(HKFreqNet, self).__init__(loss_fn, model_dim, groups, (n_layers,), strict,
                                        accum_outputs, concat_outputs, pad_input, learn_padding,
                                        with_skip_conv, with_residual_conv,
                                        **data_optim_kwargs)
        # the 4th dimension in the final conv2d is 2 encoder outputs + n_conv_layers
        self.conv_f = nn.Conv2d(model_dim, model_dim, (1, n_layers + 2), groups=groups)
        self.conv_g = nn.Conv2d(model_dim, model_dim, (1, n_layers + 2), groups=groups)

    def forward(self, x):
        """

        """
        x = self.inpt(x)

        # the conv2d layer adds a power of 2 to the n standard layers before it
        n_outs = 1 + x.size(-1) - self.receptive_field() * 2

        # stack the 2 encoder outputs
        hk_input = torch.stack((x[:, :, -n_outs - 1:-1], x[:, :, -n_outs:]), dim=-1)
        skip = None

        for layer in self.layers:
            x, skip = layer(x, skip)

            # we only keep n_outs outputs finishing at the -`layer.shift()` step
            to_stack = slice(-n_outs - layer.shift(), -layer.shift())
            hk_input = torch.cat((hk_input, skip[:, :, to_stack].unsqueeze(-1)), dim=-1)

        # Do the Gated2dConv on the gathered pieces of layers
        f = nn.Tanh()(self.conv_f(hk_input))
        g = nn.Sigmoid()(self.conv_g(hk_input))

        y = self.outpt((f * g).squeeze(-1))
        return y