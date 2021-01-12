import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from multiprocessing import cpu_count
import os

from . import FreqNet
from .modules.io import PermuteTF
from ..data import Database, make_root_db
from ..kit import NeptuneConnector
from ..data.transforms import file_to_qx


def wavenet_db(target,
               roots=None,
               files=None,
               mu=255,
               sample_rate=22050,
               neptune_path=None):
    transform = partial(file_to_qx,
                        mu=mu,
                        sr=sample_rate)
    if roots is None and files is None:
        roots = "./"
    make_root_db(target, roots=roots, files=files, extract_func=transform, n_cores=cpu_count() // 2)

    if neptune_path is not None:
        path = neptune_path.split("/")
        user, rest = path[0], "/".join(path[1:]) if path[-1] is not None else path[1]
        connector = NeptuneConnector(user=user, setup={"db": rest})
        db = Database(target)
        print("uploading database to neptune...")
        connector.upload_database("db", db)
    print("done!")
    return Database(target)


def wavenet_loss_fn(output, target):
    return F.nll_loss(output.view(-1, output.size(-1)), target.view(-1))


class WaveNet(FreqNet):

    def __init__(self,
                 input_dim=256,
                 model_dim=256,
                 groups=1,
                 n_layers=(1,),
                 with_skip_conv=True,
                 with_residual_conv=True,
                 **data_optim_kwargs
                 ):
        super(WaveNet, self).__init__(
            loss_fn=wavenet_loss_fn,
            model_dim=model_dim,
            groups=groups,
            n_layers=n_layers,
            strict=False,
            accum_outputs="right",
            concat_outputs=None,
            pad_input="left",
            learn_padding=False,
            with_skip_conv=with_skip_conv,
            with_residual_conv=with_residual_conv,
            **data_optim_kwargs)

        self.input_dim = input_dim

        self.inpt = nn.Sequential(
            nn.Embedding(self.input_dim, self.model_dim), PermuteTF())

        self.outpt = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv1d(self.model_dim, self.model_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.model_dim, self.input_dim, 1),
            PermuteTF(),
            nn.LogSoftmax(dim=1)
        )

        self.save_hyperparameters()

    def forward(self, x):
        x = self.inpt(x.squeeze())
        skips = None
        for layer in self.layers:
            x, skips = layer(x, skips)
        x = self.outpt(skips)
        return x
