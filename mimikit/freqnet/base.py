import torch
from pytorch_lightning import LightningModule, LightningDataModule
from torch.utils.data import DataLoader
import librosa
from abc import ABC

from ..data import DataObject, FeatureProxy, HOP_LENGTH
from ..kit import ShiftedSeqsPair, MMKHooks, LoggingHooks, tqdm


class ManyOneCycleLR(torch.optim.lr_scheduler.OneCycleLR):
    def step(self, epoch=None):
        try:
            super(ManyOneCycleLR, self).step(epoch=epoch)
        except ValueError:
            self.last_epoch = 0
            super(ManyOneCycleLR, self).step(epoch=epoch)


class FreqOptim:

    def __init__(self,
                 model,
                 max_lr=1e-3,
                 betas=(.9, .9),
                 div_factor=3.,
                 final_div_factor=1.,
                 pct_start=.25,
                 cycle_momentum=False):
        self.model = model
        self.max_lr = max_lr
        self.betas = betas
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.pct_start = pct_start
        self.cycle_momentum = cycle_momentum
        # those are set in setup when we know what the trainer and datamodule got
        self.steps_per_epoch = None
        self.max_epochs = None

    def configure_optimizers(self):
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.max_lr, betas=self.betas)
        self.sched = ManyOneCycleLR(self.opt,
                                    steps_per_epoch=self.steps_per_epoch,
                                    epochs=self.max_epochs,
                                    max_lr=self.max_lr, div_factor=self.div_factor,
                                    final_div_factor=self.final_div_factor,
                                    pct_start=self.pct_start,
                                    cycle_momentum=self.cycle_momentum
                                    )
        return [self.opt], [{"scheduler": self.sched, "interval": "step", "frequency": 1}]


class FreqData(LightningDataModule):
    def __init__(self,
                 model,
                 data_object=None,
                 input_seq_length=64,
                 batch_size=64,
                 to_gpu=True,
                 splits=None,
                 **loader_kwargs,
                 ):
        super(FreqData, self).__init__()
        self.model = model
        self.ds = DataObject(data_object)
        self.input_seq_length = input_seq_length
        self.batch_size = batch_size
        self.to_gpu = to_gpu
        self.splits = splits
        loader_kwargs.setdefault("drop_last", False)
        loader_kwargs.setdefault("shuffle", True)
        self.loader_kwargs = loader_kwargs
        self.train_ds, self.val_ds, self.test_ds = None, None, None

    def prepare_data(self, *args, **kwargs):
        if not (getattr(self.model, "targets_shifts_and_lengths", False)):
            raise TypeError("Expected `model` to implement `targets_shifts_and_lengths(input_length)`"
                            " in order to compute the right slices for the batches")
        targets_def = self.model.targets_shifts_and_lengths(self.input_seq_length)
        wrapper = ShiftedSeqsPair(self.input_seq_length, targets_def)
        self.ds = wrapper(self.ds)

    def setup(self, stage=None):
        if stage == "fit":
            if self.to_gpu and torch.cuda.is_available():
                self.ds.to_tensor()
                self.ds.to("cuda")
            if self.splits is None:
                self.splits = (1.,)
            sets = self.ds.split(self.splits)
            for ds, attr in zip(sets, ["train_ds", "val_ds", "test_ds"]):
                setattr(self, attr, ds)

    def train_dataloader(self):
        if not self.has_prepared_data:
            self.prepare_data()
        if not self.has_setup_fit:
            self.setup("fit")
        return DataLoader(self.train_ds, batch_size=self.batch_size, **self.loader_kwargs)

    def val_dataloader(self, shuffle=False):
        has_val = len(self.splits) >= 2 and self.splits[1] is not None
        if not has_val:
            return None
        if not self.has_prepared_data:
            self.prepare_data()
        if not self.has_setup_fit:
            self.setup("fit")
        kwargs = self.loader_kwargs.copy()
        kwargs["shuffle"] = shuffle
        return DataLoader(self.val_ds, batch_size=self.batch_size, **kwargs)

    def test_dataloader(self, shuffle=False):
        has_test = len(self.splits) >= 3 and self.splits[2] is not None
        if not has_test:
            return None
        if not self.has_prepared_data:
            self.prepare_data()
        if not self.has_setup_fit:
            self.setup("test")
        kwargs = self.loader_kwargs.copy()
        kwargs["shuffle"] = shuffle
        return DataLoader(self.test_ds, batch_size=self.batch_size, **kwargs)


class FreqNetModel(MMKHooks,
                   LoggingHooks,
                   LightningModule,
                   ABC):

    @property
    def data(self):
        """short-hand to quickly access the data object passed to the constructor"""
        return self.datamodule.ds.data

    def __init__(self,
                 data_object=None,
                 input_seq_length=64,
                 batch_size=64,
                 to_gpu=True,
                 splits=[.8, .2],
                 max_lr=1e-3,
                 betas=(.9, .9),
                 div_factor=3.,
                 final_div_factor=1.,
                 pct_start=.25,
                 cycle_momentum=False,
                 **loaders_kwargs):
        super(FreqNetModel, self).__init__()
        # dimensionality of inputs is automatically available
        if data_object is not None and not isinstance(data_object, str):
            self.input_dim = data_object.shape[-1]
            self.datamodule = FreqData(self, data_object, input_seq_length, batch_size,
                                       to_gpu, splits, **loaders_kwargs)
        else:
            raise ValueError("Please pass a valid data_object to instantiate a FreqNetModel."
                             " If you are loading from a checkpoint, you can do so by modifying your method call to :\n"
                             "FreqNetModel.load_from_checkpoint(path_to_ckpt, data_object=my_data_object)")
        self.optim = FreqOptim(self, max_lr, betas, div_factor, final_div_factor, pct_start,
                               cycle_momentum)
        # calling this updates self.hparams from any subclass : call it when subclassing!
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        batch, target = batch
        output = self.forward(batch)
        recon = self.loss_fn(output, target)
        return {"loss": recon}

    def validation_step(self, batch, batch_idx):
        batch, target = batch
        output = self.forward(batch)
        recon = self.loss_fn(output, target)
        return {"val_loss": recon}

    def setup(self, stage: str):
        if stage == "fit":
            self.logger.log_hyperparams(self.hparams)
            self.optim.max_epochs = self.trainer.max_epochs
            self.optim.steps_per_epoch = len(self.datamodule.train_dataloader())

    def configure_optimizers(self):
        return self.optim.configure_optimizers()

    def _set_hparams(self, hp):
        """
        Small handy hook to
            - copy params of a db in hparams
            - make sure we don't store arrays or tensors in hparams by replacing them
            by their string
        """
        # replace inputs passed to the hparams with their string
        if "data_object" in hp:
            if isinstance(hp["data_object"], FeatureProxy):
                # copy the attrs of the db to hparams
                for k, v in hp["data_object"].attrs.items():
                    hp[k] = v
            string = repr(hp["data_object"])
            hp["data_object"] = string[:88] + ("..." if len(string) > 88 else "")
        super(FreqNetModel, self)._set_hparams(hp)

    # Convenience Methods for generative audio models (see also, `LoggingHooks.log_audio`):

    def random_train_example(self):
        return next(iter(self.datamodule.train_dataloader()))[0][0:1]

    def random_val_example(self):
        return next(iter(self.datamodule.val_dataloader(shuffle=True)))[0][0:1]

    def generation_slices(self):
        raise NotImplementedError("subclasses of `FreqNetModel` have to implement `generation_slices`")

    def targets_shifts_and_lengths(self, input_length):
        raise NotImplementedError("subclasses of `FreqNetModel` have to implement `targets_shifts_and_lengths`")

    def generate(self, prompt, n_steps, hop_length=HOP_LENGTH, time_domain=True):
        was_training = self.training
        self.eval()
        initial_device = self.device
        self.to("cuda" if torch.cuda.is_available() else "cpu")
        if not isinstance(prompt, torch.Tensor):
            prompt = torch.from_numpy(prompt)
        if len(prompt.shape) < 3:
            prompt = prompt.unsqueeze(0)
        input_slice, output_slice = self.generation_slices()
        generated = prompt.to(self.device)
        for _ in tqdm(range(n_steps), desc="Generate", dynamic_ncols=True, leave=False, unit="step"):
            with torch.no_grad():
                out = self(generated[:, input_slice])
                generated = torch.cat((generated, out[:, output_slice]), dim=1)
        if time_domain:
            generated = generated.transpose(1, 2).squeeze()
            generated = librosa.griffinlim(generated.cpu().numpy(), hop_length=hop_length, n_iter=64)
            generated = torch.from_numpy(generated)
        else:  # for consistency with time_domain :
            generated = generated.cpu()
        self.to(initial_device)
        self.train() if was_training else None
        return generated.unsqueeze(0)
