from pytorch_lightning.utilities.cloud_io import load as pl_load
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from time import time, gmtime
from typing import Optional, Union, Callable, Dict, IO
import torch
import os
import warnings
import soundfile as sf

try:
    from neptune.experiments import Experiment as NeptuneExperiment
    from neptunecontrib.api.audio import log_audio
except ImportError:
    NeptuneExperiment = type(None)
    warnings.warn("It seems neptune and/or neptunecontrib are not installed. "
                  "You won't be able to log checkpoints and data to neptune.")

from .. import __version__ as version
from ..data.transforms import SR


class LoggingHooks:

    @property
    def neptune_experiment(self):
        """ short hand to access own NeptuneExperiment"""
        exps = [exp for exp in self.logger.experiment
                if isinstance(exp, NeptuneExperiment)]
        if any(exps):
            return exps[0]
        return None

    def on_epoch_start(self):
        # convenience to have printing of the loss at the end of the epoch
        self._ep_metrics = {}
        self._batch_count = {}

    def log_output(self, out):
        for metric, val in out.items():
            if metric not in self._ep_metrics:
                self._ep_metrics.setdefault(metric, val.detach())
                self._batch_count[metric] = 0
            else:
                self._ep_metrics[metric] += val.detach()
            self._batch_count[metric] += 1
        return out

    def training_step_end(self, out):
        return self.log_output(out)

    def validation_step_end(self, out):
        return self.log_output(out)

    def on_train_epoch_end(self, *args):
        to_print = "Epoch %i " % self.current_epoch
        to_log = {}
        for k, v in self._ep_metrics.items():
            to_print += "- %s : %.4f " % (k, v / self._batch_count[k])
            to_log[k] = (v / self._batch_count[k]).item()
        self.print(to_print)
        if getattr(self, "logger", None) is not None:
            self.logger.log_metrics(to_log)

    def on_fit_start(self):
        self._ep_time = time()

    def on_fit_end(self):
        duration = time() - self._ep_time
        total_time = gmtime(duration)
        self.print("Training finished after "
                   "{0} days {1} hours {2} mins {3} seconds".format(total_time[2] - 1, total_time[3],
                                                                    total_time[4], total_time[5]))
        if getattr(self, "logger", None) is not None:
            # accumulate if we resumed
            duration = self.hparams.get("training_time_sec", 0) + duration
            self.hparams.update({"training_time_sec": duration})
            self.logger.log_hyperparams(self.hparams)

    def log_audio(self, filename, audio_tensor, sample_rate=SR, experiments=None):
        # TensorBoards write their own "Event" file which is quite cumbersome to extract afterwards...
        # NeptuneLoggers need an audio file written on disk and store it afterwards as html on the server...
        # Just to be sure and even if it is then in 3 places : we add the audio in root_dir/audios of the model!

        # figure out where we'll save the file :
        if self.trainer is not None:
            root = os.path.join(self.trainer.default_root_dir, "audios")
        elif self._loaded_checkpoint is not None:
            root_dir = os.path.split(os.path.dirname(self._loaded_checkpoint))[0]
            root = os.path.join(root_dir, "audios")
        else:
            if os.path.dirname(filename):
                root = os.path.dirname(filename)
            else:
                root = "./"
        if not os.path.exists(root):
            os.makedirs(root, exist_ok=True)
        path = os.path.join(root, filename)
        if ".wav" != os.path.splitext(path)[-1]:
            path = path + ".wav"
        if isinstance(audio_tensor, torch.Tensor):
            audio_tensor = audio_tensor.squeeze().detach().cpu().numpy()
        sf.write(path, audio_tensor, sample_rate, 'PCM_24')
        exps = [] if experiments is None else experiments
        exps += self.logger.experiment if getattr(self, "logger", False) else []
        for exp in exps:
            # Neptune and TestTube experiments have different APIs...
            if getattr(exp, "add_audio", False):
                exp.add_audio(filename + ".wav", audio_tensor, sample_rate=sample_rate)
                exp.save()
                print("Updated TensorBoard with", filename)
            elif isinstance(exp, NeptuneExperiment):
                # add .wav.html to neptune UI
                log_audio(path, os.path.split(path)[-1], exp)
                # add .wav to artifacts
                exp.log_artifact(path, "audios/" + os.path.split(path)[-1])
                print("Updated neptune experiment", exp.id, "with", filename)
        return 1


def _check_version(other_v):
    if other_v.split(".")[0] != version.split(".")[0]:
        v = str(other_v)
        warnings.warn(("You are loading a checkpoint made by a different version of mimikit (%s) as the one" % v) +
                      (" imported in this runtime (%s). If you encounter errors " % version) +
                      (", try to install the right version with `pip install mimikit==%s" % v))


class MMKHooks:
    _loaded_checkpoint = None

    @classmethod
    def load_from_checkpoint(
            cls,
            checkpoint_path: Union[str, IO],
            map_location: Optional[Union[Dict[str, str], str, torch.device, int, Callable]] = None,
            strict: bool = True,
            **kwargs):
        """
        This is NOT used when restoring a model from the trainer, but still practical in other cases!
        """
        if map_location is not None:
            checkpoint = pl_load(checkpoint_path, map_location=map_location)
        else:
            checkpoint = pl_load(checkpoint_path, map_location=lambda storage, loc: storage)

        if checkpoint.get("version", None) is not None:
            _check_version(checkpoint["version"])
        model = cls._load_model_state(checkpoint, strict=strict, **kwargs)
        model._loaded_checkpoint = checkpoint_path
        return model

    def on_save_checkpoint(self, checkpoint):
        ckpt_callbacks = [cb for cb in self.trainer.callbacks
                          if issubclass(type(cb), ModelCheckpoint)]
        if any(ckpt_callbacks):
            path = ckpt_callbacks[0].dirpath
        else:
            path = self.trainer.default_root_dir

        # overwrite previous optim state
        torch.save({"optimizer_states": checkpoint["optimizer_states"],
                    "lr_schedulers": checkpoint["lr_schedulers"]
                    }, os.path.join(path, "last_optim_state.ckpt"))

        # pop optim items from checkpoint
        checkpoint.pop("optimizer_states")
        checkpoint.pop("lr_schedulers")
        # this seem to fix ValueError caused by lr_scheduler when resuming:
        checkpoint["global_step"] -= 2
        checkpoint["mmk_version"] = version
        return checkpoint

    def on_load_checkpoint(self, checkpoint):
        if checkpoint.get("version", None) is not None:
            _check_version(checkpoint["version"])
        # we restore the training state only if the model has a trainer...
        if getattr(self, "trainer", None) is not None:
            # ... and the trainer got a ckpt_path to resume from
            if self.trainer.resume_from_checkpoint is not None:
                ckpt_path = self.trainer.resume_from_checkpoint
                version_dir = os.path.abspath(os.path.dirname(ckpt_path))

                # make sure we'll find the optim-state or let the user know
                if "last_optim_state.ckpt" not in os.listdir(version_dir):
                    raise FileNotFoundError("Expected to find 'last_optim_state.ckpt' in the parent directory "
                                            "of the model's checkpoint's directory but no such file were found.")
                # update the checkpoint
                optim_state = torch.load(os.path.join(version_dir, "last_optim_state.ckpt"))
                checkpoint["optimizer_states"] = optim_state["optimizer_states"]
                checkpoint["lr_schedulers"] = optim_state["lr_schedulers"]

        return checkpoint
