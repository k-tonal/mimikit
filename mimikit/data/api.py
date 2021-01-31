import h5py
import numpy as np
import pandas as pd
import os
from datetime import datetime

from torch.utils.data.dataset import Subset

from .regions import Regions
from ..data.data_object import DataObject


class FeatureProxy(object):
    """
    Interface to the numerical data (array) stored in a .h5 file created with ``mimikit.data.make_root_db``

    ``FeatureProxy`` objects are like ``numpy`` arrays (on disk) and can be used as map-style ``Dataset``
    because they implement ``__len__`` and ``__getitem__``.

    Parameters
    ----------
    h5_file : ``str``
        name of the file
    ds_name : ``str``
        name of the ``h5py.Dataset`` containing the data to be served

    Attributes
    ----------
    N : ``int``
        the length of the first dimension of the underlying array
    shape : ``tuple`` of ``int``
        the shape of the underlying array
    attrs : ``dict``
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

    def get(self, regions):
        """
        get the data (numpy array) corresponding to the rows of ``regions``

        Parameters
        ----------
        regions : ``Regions``
            the files you want to get as array

        Returns
        -------
        data : ``np.ndarray``
            the concatenated data for all the requested files

        Example
        --------
        .. code-block::

            from mimikit import Database
            db = Database("my-db.h5")
            # only get files 0, 3 & 7
            db.fft.get(db.regions.iloc[[0, 3, 7]])

        """
        t_axis = self.attrs.get("time_axis", 0)
        slices = regions.slices(t_axis)
        return np.concatenate(tuple(self[slice_i] for slice_i in slices), axis=t_axis)

    def add(self, array, filename=None):
        """
        EXPERIMENTAL! append ``array`` to the feature and fill the ``"name"`` column of ``db.regions`` with 'filename'
        Parameters
        ----------
        array : ``np.ndarray``
            the array to append at the end of the feature
        filename : ``str``, optional
            a name for the array being added

        Returns
        -------
        new : ``FeatureProxy``
            the updated object
        """
        new = _add_data(self.h5_file, self.name, array, filename)
        return new

    def subset(self, indices):
        """
        transform self into a torch ``Subset`` (``Dataset``) containing only ``indices`` from the original data

        Parameters
        ----------
        indices : ``Regions`` or any object that ``Subset`` accepts as indices
            if ``indices`` is of type ``Regions``, the returned ``Subset`` will only contain the files (rows) present
            in ``indices``.

        Returns
        -------
        subset : ``torch.utils.data.Subset``
            the obtained subset
        """
        if isinstance(indices, Regions):
            indices = indices.all_indices
        return Subset(DataObject(self), indices)

    def regions(self, indices):
        """
        transform self into a torch ``Subset`` containing the files whose indices have been passed
        as ``ints`` in ``indices``

        Parameters
        ----------
        indices : ``list`` of ``ints``
            correspond to the rows of ``db.regions`` you want to keep

        Returns
        -------
        subset : ``torch.utils.data.Subset``
            the obtained subset

        Example
        --------
        .. code-block::

            from mimikit import Database
            db = Database("test.h5")
            sub = db.fft.regions([1, 2, 3, 9])
        """
        regions = Database(self.h5_file).regions.iloc[indices]
        return self.subset(regions)

    def __repr__(self):
        return "<FeatureProxy: '%s/%s'>" % (self.h5_file, self.name)


class Database(object):
    """
    interface to .h5 databases created by ``mimiki.data.make_root_db``

    Parameters
    ----------
    h5_file : ``str``
        path to the .h5 file containing the data

    Attributes
    ----------
    regions : ``Regions``
        pandas DataFrame where each row contains information about one stored file.
        see ``Regions`` for more information

    <feature_proxy> : ``FeatureProxy``
        each feature created by the extracting function passed to ``make_root_db``
        is automatically added as attribute to ``Database`` instances loading the .h5 file.
        For instance, if the extracting function returned a feature by the name ``"fft"``,
        the attribute ``fft`` of type ``FeatureProxy`` will be automatically
        added as attribute when the file is loaded and you will be able to access it through ``db.fft``.
    """
    def __init__(self, h5_file: str):
        self.h5_file = h5_file
        self.regions = Regions(self._get_dataframe("/regions"))
        with h5py.File(h5_file, "r") as f:
            self.attrs = {k: v for k, v in f.attrs.items()}
            self.features = f.attrs["features"]
            # add found features as self.feature_name = FeatureProxy(self.h5_file, feature_name)
            self._register_features(self.features)

    def visit(self, func=print):
        """
        wrapper for ``h5py.File.visititems()``

        Parameters
        ----------
        func : function
            a function to be applied recursively. The default prints names and values of the h5 items.

        Returns
        -------
        None
        """
        with h5py.File(self.h5_file, "r") as f:
            f.visititems(func)

    def _get_dataframe(self, key):
        try:
            return pd.read_hdf(self.h5_file, key=key)
        except KeyError:
            return pd.DataFrame()

    def save_dataframe(self, key, df):
        """
        stores a ``pd.DataFrame`` object under ``key``

        Parameters
        ----------
        key : ``str``
            the key under which ``df`` will be stored
        df : pd.DataFrame
            the ``DataFrame`` to be stored

        Returns
        -------
        df : ``pd.DataFrame``
            the ``DataFrame`` as it has been stored
        """
        with h5py.File(self.h5_file, "r+") as f:
            if key in f:
                f.pop(key)
        df.to_hdf(self.h5_file, key=key, mode="r+")
        return self._get_dataframe(key)

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


def _add_regions(h5_file, start, stop, duration, filename=None):
    """
    adds a row of regions to ``db.regions``
    """
    meta = pd.read_hdf(h5_file, "regions")
    new = Regions.from_start_stop([start], [stop], [duration])
    filename = datetime.now() if filename is None else filename
    new["name"] = filename
    new_meta = pd.concat((meta, new), axis=0, ignore_index=True)
    new_meta.to_hdf(h5_file, "regions")
    return new_meta


def _add_data(h5_file, ds_name, array, filename):
    """
    EXPERIMENTAL!! adds data at the end of a ``Dataset`` in a .h5 file.

    Parameters
    ----------
    h5_file : ``str``
        path to the file
    ds_name : ``str``
        name of the ``h5py.Dataset`` to which ``array`` will be added
    array : np.ndarray
        the array to add
    filename : ``str``
        a name for the array in ``db.regions``

    Returns
    -------
    arr : np.ndarray
        the array as it has been stored
    """
    N = array.shape[0]
    with h5py.File(h5_file, "r+") as f:
        if f[ds_name].shape[1:] != array.shape[1:]:
            raise ValueError(
                ("expected all but the first dimension of ``array`` to match %s. " % str(f[ds_name].shape[1:])) +
                ("Got %s" % str(array.shape[1:])))
        M = f[ds_name].shape[0]
        f[ds_name].resize(M + N, axis=0)
        f[ds_name][-N:] = array
        rv = f[ds_name][-N:]
    _add_regions(h5_file, M, N + M, N, filename)
    return rv
