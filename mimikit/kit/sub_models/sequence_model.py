import torch
import pytorch_lightning as pl
from abc import ABC
import matplotlib.pyplot as plt


from .utils import MMKHooks, LoggingHooks, tqdm
from ...utils import audio


class SequenceModel(MMKHooks,
                    LoggingHooks,
                    pl.LightningModule,
                    ABC):

    loss_fn = None

    def __init__(self):
        super(pl.LightningModule, self).__init__()
        MMKHooks.__init__(self)
        LoggingHooks.__init__(self)

    def training_step(self, batch, batch_idx):
        batch, target = batch
        output = self.forward(batch)
        L = self.loss_fn(output, target)
        return {"loss": L}

    def validation_step(self, batch, batch_idx):
        batch, target = batch
        output = self.forward(batch)
        L = self.loss_fn(output, target)
        return {"val_loss": L}

    def setup(self, stage: str):
        if stage == "fit" and getattr(self, "logger", None) is not None:
            self.logger.log_hyperparams(self.hparams)

    def batch_info(self, *args, **kwargs):
        raise NotImplementedError("subclasses of `SequenceModel` have to implement `batch_info`")

    def before_generate(self, *args, **kwargs):
        # prepare model
        self._was_training = self.training
        self._initial_device = self.device
        self.eval()
        self.to("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_grad_enabled(False)

    def after_generate(self, *args, **kwargs):
        # reset model
        self.to(self._initial_device)
        self.train() if self._was_training else None
        torch.set_grad_enabled(True)

    def prepare_prompt(self, prompt, n_steps, at_least_nd=2):
        if not isinstance(prompt, torch.Tensor):
            prompt = torch.from_numpy(prompt)
        while len(prompt.shape) < at_least_nd:
            prompt = prompt.unsqueeze(0)
        prompt = prompt.to(self.device)
        dims = prompt.size(0), n_steps, *prompt.size()[2:]
        return torch.cat((prompt, torch.zeros(*dims).to(prompt)), dim=1)

    @staticmethod
    def generate_tqdm(rng):
        return tqdm(rng, desc="Generate", dynamic_ncols=True, leave=False, unit="step", mininterval=0.25)

    def generate(self, prompt, n_steps, decode_outputs=False, **kwargs):
        raise NotImplementedError

    def encode_inputs(self, inputs: torch.Tensor):
        raise NotImplementedError

    def decode_outputs(self, outputs: torch.Tensor):
        raise NotImplementedError


class GenerateCallBack(pl.callbacks.Callback):

    def __init__(self, every_n_epochs=10, n_prompts=3, n_steps=1000,
                 plot_audios=True, play_audios=True, log_audios=False, **gen_kwargs):
        self.every_n_epochs = every_n_epochs
        self.n_prompts = n_prompts
        self.n_steps = n_steps
        self.kwargs = gen_kwargs
        self.log_audios = log_audios
        self.plot_audios = plot_audios
        self.play_audios = play_audios

    def on_epoch_end(self, trainer: pl.Trainer, model: SequenceModel):
        if trainer.current_epoch % self.every_n_epochs != 0:
            return
        prompt = next(iter(trainer.train_dataloader))[:self.n_prompts]
        output = model.generate(prompt, self.n_steps, decode_outputs=True, **self.kwargs)
        for i in range(output.size(0)):
            y = output[i].detach().cpu().numpy()
            if self.plot_audios:
                plt.figure(figsize=(20, 2))
                plt.plot(y)
                plt.show()
            if self.play_audios:
                audio(y, sr=model.hparams.get("sr", 22050), hop_length=model.hparams.get("hop_length", 512))

        if self.log_audios:
            for i in range(output.size(0)):
                filename = "epoch=%i - prompt=%i" % (trainer.current_epoch, i)
                model.log_audio(filename, output[i].unsqueeze(0), sample_rate=model.hparams.get("sr", 22050))
