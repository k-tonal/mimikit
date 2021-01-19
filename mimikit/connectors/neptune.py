from neptune import Session
from neptune.experiments import Experiment
from zipfile import ZipFile
import os
from typing import Tuple
from getpass import getpass
import shutil

try:
    from pytorch_lightning import LightningModule
except ModuleNotFoundError:
    # we only need lightning for annotation purposes
    # and setup.py allows for a "no-torch" install that contains this module
    # so we simulate the import if the module wasn't found
    class pytorch_lightning:
        LightningModule = object()

from mimikit.data import Database


class NeptuneConnector:
    NEPTUNE_TOKEN_KEY = "NEPTUNE_API_TOKEN"

    @property
    def api_token(self):
        """
        either finds neptune's API token in os.environ[self.NEPTUNE_TOKEN_KEY]
        or prompt the user to paste it and stores it in the environment under self.NEPTUNE_TOKEN_KEY

        Returns
        -------
        api_token : str

        """
        api_token = os.environ.get(self.NEPTUNE_TOKEN_KEY, None)
        if api_token is None:
            api_token = getpass("Couldn't find your api token in this system's environment. "
                                "Please paste your API token from neptune here: ")
            os.environ[self.NEPTUNE_TOKEN_KEY] = api_token
        return api_token

    @property
    def session(self):
        """
        neptune.Session initialized with a found/prompted token

        Returns
        -------
        session : neptune.Session

        """
        if self._session is None:
            self._session = Session.with_default_backend(api_token=self.api_token)
        return self._session

    def __init__(self,
                 user: str = None,
                 setup: dict = None):
        """
        manages the integration of ``neptune`` into ``mimikit``.

        In order to connect to neptune, the ``NeptuneConnector`` probes the system's environment for an API token, and
        if none were found, it prompts the user for it and subsequently stores it in the environment.
        You can configure the name of the environment variable through the static attribute
        ``NeptuneConnector.NEPTUNE_TOKEN_KEY`` which is, by default, set to "NEPTUNE_API_TOKEN".

        Each instance of the ``NeptuneConnector`` type holds a dictionary in it's ``setup`` attribute where the keys
        are user-defined and the values are expected to be paths on a neptune-user's graph.
        Specifically, ``"<user>"`` being expected in the ``user`` argument of the ``NeptuneConnector`` constructor,
        the values in ``setup`` can be of two forms :
            1.  ``"<project>"``
            2. ``"<project>/<experiment-id>"``
        thus allowing the ``NeptuneConnector`` to
            1. create new experiments in ``"<user>/<project>"``
            2. retrieve data from specific experiments @ ``"<user>/<project>/<experiment-id>"``

        All methods of the ``NeptuneConnector`` take a ``setup_key`` argument to specify "where" the method should be
        executed.

        Parameters
        ----------
        user : str
            the neptune username
        setup : dict
            a dictionary where the values are paths in the neptune user's graph.
            see description above for details.

        """
        self.user = user
        self.setup = setup
        self._session = None
        # initialize the token right-away
        token = self.api_token

    def path(self, setup_key: str, split: bool = False):
        """
        returns the full neptune path corresponding to ``setup_key``.

        Parameters
        ----------
        setup_key : str
            the key to look up in self.setup
        split : bool, optional
            whether to split the path. If there's no exp-id at the end of the setup_value, None is appended

        Returns
        -------
        path : str or list of str depending whether or not `split=True`

        """
        if setup_key not in self.setup:
            raise KeyError("%s not in found in setup" % setup_key)
        p = os.path.join(self.user, self.setup[setup_key])
        if split:
            p = p.split("/")
            if len(p) == 2:  # i.e. no experiment ID
                p += [None]
        return p

    def get_experiment(self, setup_key: str):
        """
        get the experiment object for setup_key

        Parameters
        ----------
        setup_key : str
            the key to look up in self.setup

        Returns
        -------
        experiment : neptune.Experiment
            the experiment object

        Raises
        ------
        ValueError if the value for the setup_key doesn't contain an experiment-id at its end
        """
        namespace, project, exp_id = self.path(setup_key, split=True)
        if exp_id is None:
            raise ValueError("'%s' has no experiment-id at the end" % self.path(setup_key))
        project = self.session.get_project(namespace + "/" + project)
        exps = project.get_experiments(id=exp_id)
        return None if not any(exps) else exps[0]

    def create_experiment(self, setup_key: str, params: dict, **kwargs):
        """
        creates a new experiment in the project of the setup_key

        Parameters
        ----------
        setup_key : str
            the key to look up in self.setup where the experiment will be created
        params : dict
            the params (hyper-parameters) of the experiment
        kwargs
            optional arguments for the native `create_experiment()` method from neptune.
            see https://docs.neptune.ai/api-reference/neptune/index.html#neptune.create_experiment

        Returns
        -------
        experiment : neptune.Experiment
            the created experiment

        """
        user, project, exp_id = self.path(setup_key, split=True)
        project_name = os.path.join(self.user, project)
        project = self.session.get_project(project_name)
        return project.create_experiment(params=params, **kwargs)

    def download_experiment(self, setup_key: str, destination: str = "./", artifacts: str = "/"):
        """
        downloads all artifacts lying at the root of an experiment into a folder named after the experiment-id

        Parameters
        ----------
        setup_key : str
            the key to look up in self.setup
        destination : str, optional
            a directory where to create the folder `"<exp-id>/"` in which the data will be unzipped.
        artifacts : str, optional
            an optional path in the artifacts to download, default is "/" i.e. all artifacts. Has to be a directory.

        Returns
        -------
        experiment : neptune.Experiment
            the neptune object for the downloaded experiment

        """
        namespace, project, exp_id = self.path(setup_key, split=True)
        project_name = namespace + "/" + project
        model_project = self.session.get_project(project_name)
        exp = model_project.get_experiments(id=exp_id)[0]
        destination = os.path.join(destination, exp_id)
        exp.download_artifacts(artifacts, destination)
        artifacts_name = "output" if artifacts == "/" else os.path.split(artifacts.strip("/"))[-1]
        with ZipFile(os.path.join(destination, artifacts_name + ".zip")) as f:
            f.extractall(destination)
        if artifacts == "/":
            for subdir in os.listdir(os.path.join(destination, artifacts_name)):
                shutil.move(src=os.path.join(destination, artifacts_name, subdir),
                            dst=os.path.join(destination, subdir))
        os.remove(os.path.join(destination, artifacts_name + ".zip"))
        if artifacts == "/":
            shutil.rmtree(os.path.join(destination, artifacts_name))
        return exp

    def get_project(self, setup_key: str):
        """
        get the project object for setup_key

        Parameters
        ----------
        setup_key : str
            the key to look up in self.setup

        Returns
        -------
        project : neptune.Project
            the project object

        """
        user, project, _ = self.path(setup_key, split=True)
        project = self.session.get_project(os.path.join(self.user, project))
        return project

    def upload_database(self, setup_key: str, db: Database):
        """
        upload .h5 file to a neptune Experiment.

        If the value of the setup_key doesn't contain an experiment-ID, one is created.
        If it does, the .h5 file is added to the top-level of the artifacts of this experiment.

        Parameters
        ----------
        setup_key : str
            the key in the setup you want to upload to
        db : Database
            the object pointing to the file you want to upload

        Returns
        -------
        `1` if the upload was successful
        """
        _, _, exp_id = self.path(setup_key, split=True)
        if exp_id is None:
            feature = [name for name in db.features if "label" not in name][0]
            feat_prox = getattr(db, feature)
            params = {"name": db.h5_file,
                      "feature_name": feature,
                      "shape": feat_prox.shape,
                      "files": len(db.metadata)}
            params.update(feat_prox.attrs)
            exp = self.create_experiment(setup_key, params=params)
        else:
            exp = self.get_experiment(setup_key)
        exp.log_artifact(db.h5_file)
        exp.stop() if exp_id is None else None
        return 1

    def download_database(self, setup_key: str, path_to_artifact: str, destination="./"):
        """
        downloads a database (.h5 artifact) from the experiment corresponding to a setup_key

        Parameters
        ----------
        setup_key : str
            the key where to find the experiment in self.setup
        path_to_artifact :
            the path to desired artifact within the neptune's experiment
        destination : str, optional
            the path on your file-system where to put the artifact. default is "./"

        Returns
        -------
        db : Database
            an Database instance of the downloaded .h5 file
        """
        namespace, project, exp_id = self.path(setup_key, split=True)
        project_name = namespace + "/" + project
        data_project = self.session.get_project(project_name)
        exp = data_project.get_experiments(id=exp_id)[0]
        exp.download_artifact(path_to_artifact, destination)
        return Database(os.path.join(destination, os.path.split(path_to_artifact)[-1]))

    def upload_model(self,
                     setup_key: str,
                     model: [str, LightningModule],
                     artifacts: Tuple[str] = ("states", "logs", "audios")):
        """
        upload the sub-directories of a model to neptune at the specified setup_key

        Parameters
        ----------
        setup_key : str
            the key in self.setup where the data should be uploaded.
        model : str or LightningModule
            the model (``LightningModule``) to be uploaded or a root directory (``str``) for the ``artifacts``.
        artifacts : tuple of str, optional
            tuple containing the names of the sub-directories to be uploaded.
            by default `upload_model` uploads the sub-directories "states", "logs" and "audios".
            each uploaded artifact will sit at the top-level of the experiment

        Returns
        -------
        `1` if the upload was successful

        Raises
        ------
        ValueError if `model` is not a string and has neither a `trainer` nor a `_loaded_from_checkpoint` attribute
        """
        if isinstance(model, str):
            root_dir = model
            experiment = None
        else:
            if getattr(model, "trainer", None) is not None:
                root_dir = model.trainer.default_root_dir
            elif getattr(model, "_loaded_checkpoint", False):
                root_dir = os.path.split(os.path.dirname(model._loaded_checkpoint))[0]
            else:
                raise ValueError("Expected to either find a 'trainer' or a '_loaded_from_checkpoint' attribute in the"
                                 " model. Found neither.")
            experiment = [exp for exp in model.logger.experiment if isinstance(exp, Experiment)]
            experiment = None if not any(experiment) else experiment[0]
        if experiment is None:
            _, _, exp_id = self.path(setup_key, split=True)
            if exp_id is None:
                experiment = self.create_experiment(setup_key, {})
            else:
                experiment = self.get_experiment(setup_key)
        for directory in os.listdir(root_dir):
            if directory in artifacts:
                artifact = os.path.join(root_dir, directory)
                experiment.log_artifact(artifact, directory)
                print("successfully uploaded", artifact, "to", experiment.id)
        return 1
