import h5py
import numpy as np
import pandas as pd
import os
from datetime import datetime

from neptune import Session
from torch.utils.data.dataset import Subset

from .metadata import Metadata
from ..data.data_object import DataObject


class FeatureProxy(object):
    """
    Interface to the numerical data (array) stored in a .h5 file created with mimikit.

    Parameters
    ----------
    h5_file : str
        name of the file
    ds_name : str
        name of the ``h5py.Dataset`` containing the data

    Attributes
    ----------
    N : int
        the length of the first dimension of the underlying array
    shape : tuple of int
        the shape of the underlying array
    attrs : dict
        dictionary of additional information about the data as returned
        by the ``extract_func`` passed to ``make_root_db``.
        Typically, this is where you want to store the parameters of your extracting function, e.g.
        sample rate, hop length etc.
    """

    def __init__(self, h5_file: str, ds_name: str):

        self.h5_file = h5_file
        self.name = ds_name
        with h5py.File(h5_file, "r") as f:
            ds = f[self.name]
            self.N = ds.shape[0]
            self.shape = ds.shape
            self.attrs = {k: v for k, v in ds.attrs.items()}

    def __len__(self):
        return self.N

    def __getitem__(self, item):
        with h5py.File(self.h5_file, "r") as f:
            rv = f[self.name][item]
        return rv

    def get(self, metadata):
        t_axis = self.attrs.get("time_axis", 0)
        slices = metadata.slices(t_axis)
        return np.concatenate(tuple(self[slice_i] for slice_i in slices), axis=t_axis)

    def add(self, array, filename=None):
        new = add_data(self.h5_file, self.name, array, filename)
        return new

    def subset(self, indices):
        if isinstance(indices, Metadata):
            indices = indices.all_indices
        return Subset(DataObject(self), indices)

    def __repr__(self):
        return "<FeatureProxy: '%s/%s'>" % (self.h5_file, self.name)


class Database(object):
    """
    interface to .h5 databases created by mimikit

    Parameters
    ----------
    h5_file : str
        path to the .h5 file containing the data

    Attributes
    ----------
    metadata : Metadata
        pandas DataFrame where each row contains information about one stored file.
        see ``Metadata`` for more information

    <feature_proxy> : FeatureProxy
        each feature created by the extracting function passed to ``make_root_db``
        is automatically added as attribute. If the extracting function returned a feature
        by the name ``"fft"``, the attribute ``fft`` of type ``FeatureProxy`` will be automatically
        added when the file is loaded and you will be able to access it through ``db.fft``.
    """
    def __init__(self, h5_file: str):
        self.h5_file = h5_file
        self.info = self._get_dataframe("/info")
        self.metadata = Metadata(self._get_dataframe("/metadata"))
        with h5py.File(h5_file, "r") as f:
            # add found features as self.feature_name = FeatureProxy(self.h5_file, feature_name)
            self._register_features(self.features)
            self._register_dataframes(self.dataframes)

    @property
    def features(self):
        names = self.info.iloc[:, 2:].T.index.get_level_values(0)
        return list(set(names))

    @property
    def dataframes(self):
        keys = set()

        def func(k, v):
            if "pandas_type" in v.attrs.keys() and k.split("/")[0] not in ("layouts", "info", "metadata"):
                keys.add(k)
            return None

        self.visit(func)
        return list(keys)

    def visit(self, func=print):
        with h5py.File(self.h5_file, "r") as f:
            f.visititems(func)

    def _get_dataframe(self, key):
        try:
            return pd.read_hdf(self.h5_file, key=key)
        except KeyError:
            return pd.DataFrame()

    def save_dataframe(self, key, df):
        with h5py.File(self.h5_file, "r+") as f:
            if key in f:
                f.pop(key)
        df.to_hdf(self.h5_file, key=key, mode="r+")
        return self._get_dataframe(key)

    def layout_for(self, feature):
        with h5py.File(self.h5_file, "r") as f:
            if "layouts" not in f.keys():
                return pd.DataFrame()
        return self._get_dataframe("layouts/" + feature)

    def _register_features(self, names):
        for name in names:
            setattr(self, name, FeatureProxy(self.h5_file, name))
        return None

    def _register_dataframes(self, names):
        for name in names:
            setattr(self, name, self._get_dataframe(name))
        return None

    def __repr__(self):
        return "<Database: '%s'>" % os.path.split(self.h5_file)[-1]


def add_feature(h5_file, feature_name, array):
    # TODO !
    pass


def add_metadata(h5_file, start, stop, duration, ds_name, filename=None):
    meta = pd.read_hdf(h5_file, "metadata")
    layout = pd.read_hdf(h5_file, "layouts/" + ds_name)
    info = pd.read_hdf(h5_file, "info")
    new = Metadata.from_start_stop([start], [stop], [duration])
    filename = datetime.now() if filename is None else filename
    new["name"] = filename
    new_meta = pd.concat((meta, new), axis=0, ignore_index=True)
    new_layout = pd.concat((layout, new), axis=0, ignore_index=True)
    new_info = info.iloc[info.index.max()]
    new_info.loc[:] = ("added", filename,
                       new_info[(ds_name, "dtype")],
                       (new.duration.item(), *new_info[(ds_name, "shape")][1:]),
                       "xMb")
    new_info = info.append(new_info, ignore_index=True)
    new_meta.to_hdf(h5_file, "metadata")
    new_layout.to_hdf(h5_file, "layouts/" + ds_name)
    new_info.to_hdf(h5_file, "info")
    return new_info


def add_data(h5_file, ds_name, array, filename):
    N = array.shape[0]
    with h5py.File(h5_file, "r+") as f:
        if f[ds_name].shape[1:] != array.shape[1:]:
            raise ValueError(
                ("expected all but the first dimension of `array` to match %s. " % str(f[ds_name].shape[1:])) +
                ("Got %s" % str(array.shape[1:])))
        M = f[ds_name].shape[0]
        f[ds_name].resize(M + N, axis=0)
        f[ds_name][-N:] = array
        rv = f[ds_name][-N:]
    add_metadata(h5_file, M, N + M, N, ds_name, filename)
    return rv


def upload_database(db: Database,
                    api_token: str,
                    project_name: str,
                    experiment_name: str):
    """
    upload a Database object to neptune.ai by adding it to the artifacts of a new experiment

    Parameters
    ----------
    db : Database
        a Database object bound to the .h5 file you want to upload
    api_token : str
        your neptune API token
    project_name : str
        the name of the neptune project where you want to upload the db.
        This is always of the form ``"account/project"``
    experiment_name : str
        the name you want to give to the experiment that will be created

    Returns
    -------

    """
    session = Session.with_default_backend(api_token=api_token)
    data_project = session.get_project(project_name)
    feature = [name for name in db.features if "label" not in name][0]
    feat_prox = getattr(db, feature)
    params = {"name": experiment_name,
              "feature_name": feature,
              "shape": feat_prox.shape,
              "files": len(db.metadata)}
    params.update(feat_prox.attrs)
    exp = data_project.create_experiment(name=experiment_name,
                                         params=params)
    exp.log_artifact(db.h5_file)
    return exp.stop()


def download_database(api_token: str,
                      full_exp_path: str,
                      database_name: str,
                      destination: str = "./"):
    """
    download a .h5 database file from a neptune.ai Experiment

    Parameters
    ----------
    api_token : str
        your neptune API token
    full_exp_path : str
        the full path to the experiment, i.e. ``"account/project/experiment-id"``
    database_name : str
        name of the .h5 file (artifact) to be downloaded
    destination : str, optional
        where you want to store the downloaded file on your system

    Returns
    -------
    Database

    """
    session = Session.with_default_backend(api_token=api_token)
    namespace, project, exp_id = full_exp_path.split("/")
    project_name = namespace + "/" + project
    data_project = session.get_project(project_name)
    exp = data_project.get_experiments(id=exp_id)[0]
    exp.download_artifact(database_name, destination)
    return Database(os.path.join(destination, database_name))
