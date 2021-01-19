import os
import torch
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from .loggers import MMKDefaultLogger
from .checkpoint import MMKCheckpoint
from .callbacks import EpochProgressBarCallback
from pytorch_lightning.trainer import Trainer
import warnings


def get_trainer(root_dir=None,
                version=None,
                resume_from_checkpoint=None,
                epochs=None,  # add MMKCheckpoint if not None
                model=None,
                neptune_connector=None,
                **kwargs):
    """
    pre-configure a `pytorch_lightning.Trainer` and returns it.

    Parameters
    ----------
    root_dir : str, optional
        the directory where all the files created during training will be saved. If `None` it defaults to `'./'`.
    version : int, optional
        an optional version number. This creates a sub-directory structure of the form root_dir/version_i/
        - the value `-1` creates a new version dynamically by finding the greatest
          version number in `root_dir` and adding `1`.
        - any other specific `int` creates/overwrites the version for this `int`.
        - `None` bypasses versioning
    resume_from_checkpoint : str, optional
        path to a checkpoint you want to resume training from.
    epochs : int or list of ints, optional
        - if int : checkpoints will be saved every int epochs
        - if list of ints : checkpoints will be saved at those specific ints
        > Note : a final checkpoint will always be saved at the end of the training and if you interrupt the training
                manually with a `KeyboardInterrupt`.
    model : pytorch_lightning.LightningModule, optional
        the model you will train. Only required when `neptune_connector` is not None
    neptune_connector : NeptuneConnector, optional
        if this argument is set, ``neptune_connector`` is expected to have a "model" key in its ``setup``
        in order to bind the model with a neptune experiment.
        If the value associated to the "model" key contains no experiment-id, a new experiment will be created.
        If the value does contain an experiment-id, this exact experiment will be accessed and updated.
    kwargs
        additional keywords arguments are passed directly to the `Trainer`.
        see https://pytorch-lightning.readthedocs.io/en/latest/trainer.html#trainer-class-api for full references.

    Returns
    -------
    trainer : pytorch_lightning.Trainer

    Notes
    -----
    1. Differences to the default Trainer in lightning are :
        - we add an `EpochProgressBarCallback`
        - the `progress_bar_refresh_rate` is set to `20` which avoids spurious crashes in colab
        - `num_sanity_val_steps` is set to 0
        - the number of `gpus` is set to automatically be the number of devices `torch` discovered.
        - a default `MMKDefaultLogger` will be added to the loggers if you don't pass `loggers=False` in the `kwargs.
          It is a subclass of a `TestTubeLogger` from `lightning` and will save its files in `root_dir/logs`

    Raises
    ------
    TypeError if `version` is neither `None` nor an `int`.

    ValueError if `neptune_connector` is not `None` while `model` is.
    """
    # Figure out the root_dir
    if root_dir is None:
        if resume_from_checkpoint is not None:
            # keep the same root_dir as the checkpoint
            ckpt_dir = os.path.abspath(os.path.dirname(resume_from_checkpoint))
            root_dir = os.path.dirname(ckpt_dir)
            kwargs["resume_from_checkpoint"] = resume_from_checkpoint
        else:
            root_dir = "./"
    else:
        if resume_from_checkpoint is not None:
            kwargs.setdefault("resume_from_checkpoint", resume_from_checkpoint)

    # Figure out the version
    if version is not None:
        if not isinstance(version, int):
            raise TypeError("Expected `version` to be of type int. Got %s" % str(type(version)))
        if version == -1:
            versions = [int(v.split("_")[1]) for v in os.listdir(root_dir) if "version" in v]
            next_version = 1 + (max(versions) if versions else -1)
        else:
            next_version = version  # this overwrites any existing files!
        default_root_dir = os.path.join(root_dir, "version_" + str(next_version))
    else:
        next_version = 0
        default_root_dir = root_dir

    user_logger = kwargs.get("logger", None)
    loggers = []

    if neptune_connector is not None:
        if model is None:
            raise ValueError("Expected `model` not to be None in order to create a neptune.Experiment")
        api_token = neptune_connector.api_token
        path = neptune_connector.path('model', split=True)
        project = "/".join(path[:2])
        exp_id = path[-1]
        loggers.append(NeptuneLogger(api_token, project, params=model.hparams,
                                     experiment_name=default_root_dir, experiment_id=exp_id))
    if user_logger is None:
        loggers.append(MMKDefaultLogger(default_root_dir, next_version))
    elif user_logger:
        loggers.append(user_logger)
    else:  # it is falsy and the user DOESN'T want logging
        loggers = False
        if neptune_connector is not None:
            warnings.warn("You provided arguments to instantiate a NeptuneLogger but set `logger=False` in the kwargs. "
                          "The latter resulting in turning any logging off, your experiment won't be logged to neptune.")

    kwargs["logger"] = loggers

    # Figure out checkpoints
    # if the user specifies one, we don't add ours
    if not any(issubclass(type(cb), ModelCheckpoint) for cb in kwargs.get("callbacks", [])):
        if epochs is not None:
            kwargs.setdefault("callbacks", []).append(MMKCheckpoint(dirpath=default_root_dir, epochs=epochs))
        else:
            warnings.warn("You neither specified `epochs` to enable default checkpoint_callback "
                          "nor passed a `checkpoint_callback` argument as kwarg. Saving of your model won't be managed"
                          " by mimikit nor by pytorch_lightning's Trainer.")
    # add epoch progress bar
    kwargs.setdefault("callbacks", []).append(EpochProgressBarCallback())
    kwargs.setdefault("progress_bar_refresh_rate", 20)
    kwargs.setdefault("process_position", 1)
    kwargs.setdefault("num_sanity_val_steps", 0)  # this is 2 by default and messes up the steps count...
    kwargs.setdefault("gpus", torch.cuda.device_count() if torch.cuda.is_available() else 0)
    print("Checkpoints and logs will be saved in", os.path.abspath(default_root_dir))
    return Trainer(default_root_dir=default_root_dir, **kwargs)
