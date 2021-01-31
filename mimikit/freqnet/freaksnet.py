import torch.nn as nn

from .freq_layer import FreqLayer
from .freqnet import FreqNet
from .modules import mean_L1_prop, GatedLinearInput, AbsLinearOutput


class FreaksNet(FreqNet):
    """
    Simple adaptation of FreqNet to have varying parameters for each layer.
    to each block of layer corresponds a tuple of values for the params listed in FeqNet.LAYER_PARAMS
    """

    def __init__(self,
                 loss_fn=mean_L1_prop,
                 input_dim=1025,
                 model_dim=512,
                 groups=((1, 1, 1),),
                 n_layers=(3,),
                 kernel_size=((2, 2, 2),),
                 strict=((False, True, False),),
                 accum_outputs=((-1, 1, -1),),
                 concat_outputs=((1, -1, 1),),
                 pad_input=((0, 0, 0),),
                 learn_padding=((False, False, False),),
                 with_skip_conv=((False, True, False,), ),
                 with_residual_conv=((True, True, False), ),
                 **data_optim_kwargs):
        super(FreaksNet, self).__init__(**data_optim_kwargs)
        self._loss_fn = loss_fn
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.groups = groups
        self.n_layers = n_layers
        self.strict = strict
        self.kernel_size = kernel_size
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
            FreqLayer(layer_index=i, input_dim=model_dim, layer_dim=model_dim,
                      **{k: v[j][i] for k, v in layer_kwargs.items()})
            for j, n_layers in enumerate(self.n_layers) for i in range(n_layers)
        ])

        # Output Decoder
        self.outpt = AbsLinearOutput(self.model_dim, self.input_dim)

        self.save_hyperparameters()
