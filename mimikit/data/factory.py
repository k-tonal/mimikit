import h5py
import numpy as np
import pandas as pd
from multiprocessing import cpu_count, Pool
from typing import Iterable
import os
import warnings

from .api import Database
from .regions import Regions
from .transforms import default_extract_func

warnings.filterwarnings("ignore", message="PySoundFile failed.")


class AudioFileWalker:

    AUDIO_EXTENSIONS = ("wav", "aif", "aiff", "mp3", "m4a", "mp4")

    def __init__(self, roots=None, files=None):
        """
        recursively find audio files from `roots` and/or collect audio files passed in `files`

        Parameters
        ----------
        roots : str or list of str
            a single path (string, os.Path) or an Iterable of paths from which to collect audio files recursively
        files : str or list of str
            a single path (string, os.Path) or an Iterable of paths

        Examples
        --------
        >>> files = list(AudioFileWalker(roots="./my-directory", files=["sound.mp3"]))

        Notes
        ------
        any file whose extension isn't in AudioFileWalker.AUDIO_EXTENSIONS will be ignored,
        regardless whether it was found recursively or passed through the `files` argument.
        """
        generators = []

        if roots is not None and isinstance(roots, Iterable):
            if isinstance(roots, str):
                if not os.path.exists(roots):
                    raise FileNotFoundError("%s does not exist." % roots)
                generators += [AudioFileWalker.walk_root(roots)]
            else:
                for r in roots:
                    if not os.path.exists(r):
                        raise FileNotFoundError("%s does not exist." % r)
                generators += [AudioFileWalker.walk_root(root) for root in roots]

        if files is not None and isinstance(files, Iterable):
            if isinstance(files, str):
                if not os.path.exists(files):
                    raise FileNotFoundError("%s does not exist." % files)
                generators += [(f for f in [files] if AudioFileWalker.is_audio_file(files))]
            else:
                for f in files:
                    if not os.path.exists(f):
                        raise FileNotFoundError("%s does not exist." % f)
                generators += [(f for f in files if AudioFileWalker.is_audio_file(f))]

        self.generators = generators

    def __iter__(self):
        for generator in self.generators:
            for file in generator:
                yield file

    @staticmethod
    def walk_root(root):
        for directory, _, files in os.walk(root):
            for audio_file in filter(AudioFileWalker.is_audio_file, files):
                yield os.path.join(directory, audio_file)

    @staticmethod
    def is_audio_file(filename):
        # filter out hidden files
        if os.path.split(filename.strip("/"))[-1].startswith("."):
            return False
        return os.path.splitext(filename)[-1].strip(".") in AudioFileWalker.AUDIO_EXTENSIONS


def _sizeof_fmt(num, suffix='b'):
    for unit in ['', 'k', 'M', 'G', 'T', 'P', 'E', 'Z']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


# Core function

def file_to_db(abs_path, extract_func=default_extract_func, output_path=None, mode="w"):
    """
    apply ``extract_func`` to ``abs_path`` and write the result in a .h5 file.

    Parameters
    ----------
    abs_path : str
        path to the file to be extracted
    extract_func : function
        the function to use for the extraction - should take exactly one argument
    output_path : str or None
        name of the created .h5 file if not None, else the name of the created .h5 file
        will be of the form : "<abs_path_without_extension>.h5"
    mode : str
        the mode to use when opening the .h5 file. default is "w".

    Returns
    -------
    created_file, infos : str, dict
        the name of the created .h5 file and a ``dict`` with the keys ``"dtype"`` and ``"shape"``
    """
    print("making .h5 for %s" % abs_path)
    if output_path is None:
        output_path = os.path.splitext(abs_path)[0] + ".h5"
    else:
        if output_path[-3:] != ".h5":
            output_path += ".h5"
    rv = extract_func(abs_path)
    if "regions" not in rv:
        raise ValueError("Expected `extract_func` to return a ('regions', Regions) item. Found none")
    info = {}
    f = h5py.File(output_path, mode)
    for name, (attrs, data) in rv.items():
        if issubclass(type(data), np.ndarray):
            ds = f.create_dataset(name=name, shape=data.shape, data=data)
            ds.attrs.update(attrs)
            info[name] = {"dtype": ds.dtype, "shape": ds.shape}
        elif issubclass(type(data), pd.DataFrame):
            f.close()
            pd.DataFrame(data).to_hdf(output_path, name, "r+")
            f = h5py.File(output_path, "r+")
    f.flush()
    f.close()
    return output_path, info


# Multiprocessing routine

def _make_db_for_each_file(file_walker,
                           extract_func=default_extract_func,
                           n_cores=cpu_count()):
    """
    apply ``extract_func`` to the files found by ``file_walker``

    If file_walker found more than ``n_cores`` files, a multiprocessing ``Pool`` is used to speed up the process.

    Parameters
    ----------
    file_walker : iterable of str
        collection of files to be processed
    extract_func : function
        the function to apply to each file. Must take only one argument (the path to the file)
    n_cores : int, optional
        the number of cores used in the ``Pool``, default is the number of available cores on the system.

    Returns
    -------
    temp_dbs : list of tuples
        each tuple in the list is of the form ``("<created_file>.h5", dict(feature_name=dict(dtype=..., shape=...), ...))``
    """
    # add ".tmp_" prefix to the output_paths
    args = [(file, extract_func, os.path.join(os.path.split(file)[0], ".tmp_" + os.path.split(file)[1]))
            for file in file_walker]
    if len(args) > n_cores:
        with Pool(n_cores) as p:
            tmp_dbs_infos = p.starmap(file_to_db, args)
    else:
        tmp_dbs_infos = [file_to_db(*arg) for arg in args]
    return tmp_dbs_infos


def _aggregate_db_infos(infos):
    """
    helper to reduce shapes and dtypes of several files and to concatenate their regions

    Parameters
    ----------
    infos : list of tuples
        see the returned value of ``_make_db_for_each_file`` for the expected type

    Returns
    -------
    ds_definitions : dict
        the keys are the name of the H5Datasets (features)
        the values are dictionaries with keys ("shape", "dtype", "regions") and corresponding values
    """
    paths = [x[0] for x in infos]
    features = set([feature for db in infos for feature in db[1].keys()])
    ds_definitions = {}
    for f in features:
        dtype = set([db[1][f]["dtype"] for db in infos])
        assert len(dtype) == 1, "aggregated features must be of a unique dtype"
        dtype = dtype.pop()
        shapes = [db[1][f]["shape"] for db in infos]
        dims = shapes[0][1:]
        assert all(shp[1:] == dims for shp in
                   shapes[1:]), "all features should have the same dimensions but for the first axis"
        # collect the regions for the files
        regions = Regions.from_duration([s[0] for s in shapes])
        ds_shape = (regions.last_stop, *dims)
        regions.index = paths
        ds_definitions[f] = {"shape": ds_shape, "dtype": dtype, "meta_regions": regions}
    return ds_definitions


# Aggregating function

def _aggregate_dbs(target, tmp_dbs_infos, mode="w"):
    """
    copy a list of .h5 files as defined in ``tmp_dbs_infos`` into ``target``

    Parameters
    ----------
    target : str
        name for the target file
    tmp_dbs_infos : list
        see the returned value of ``_make_db_for_each_file`` for the expected type
    mode : str
        the mode with which ``target`` is opened the first time. Must be 'w' if ``target`` doesn't exist.
    """
    features_infos = _aggregate_db_infos(tmp_dbs_infos)

    # open the file & create the master datasets to host the features
    with h5py.File(target, mode) as f:
        # add the list of features
        f.attrs["features"] = list(features_infos.keys())
        for name, params in features_infos.items():
            f.create_dataset(name, shape=params["shape"], dtype=params["dtype"],
                             chunks=True, maxshape=(None, *params["shape"][1:]))

    # prepare args for copying the tmp_files into the appropriate regions
    args = []
    for feature, info in features_infos.items():
        # collect the arguments for copying each file into its regions in the master datasets/features
        args += [(source, feature, indices) for source, indices in
                 zip(info["meta_regions"].index, info["meta_regions"].slices(time_axis=0))]

    # copy the data
    intra_regions = None
    for source, key, indices in args:
        with h5py.File(source, "r") as src:
            data = src[key][()]
            attrs = {k: v for k, v in src[key].attrs.items()}
        # concat the regions :
        regions = pd.read_hdf(source, key="regions", mode="r")
        regions.loc[:, ("start", "stop")] += indices[0].start
        # remove ".tmp_" from the source's name
        regions.loc[:, "name"] = "".join(source.split(".tmp_"))
        intra_regions = pd.concat(([intra_regions] if intra_regions is not None else []) + [regions])
        with h5py.File(target, "r+") as trgt:
            trgt[key][indices] = data
            trgt[key].attrs.update(attrs)
    pd.DataFrame(intra_regions).to_hdf(target, "regions", mode="r+")

    # remove the temp_dbs
    for src, _ in tmp_dbs_infos:
        if src != target:
            os.remove(src)


def make_root_db(db_name, roots='./', files=None, extract_func=default_extract_func,
                 n_cores=cpu_count()):
    """
    extract and aggregate several files into a single .h5 Database

    Parameters
    ----------
    db_name : str
        the name of the db to be created
    roots : str or list of str, optional
        directories from which to search recursively for audio files.
        default is "./"
    files : str or list of str, optional
        single file(s) to include in the db.
        default is `None`
    extract_func : function, optional
        the function to use for the extraction - should take exactly one argument.
        the default transforms the file to the stft with n_fft=2048 and hop_length=512.
    n_cores : int, optional
        the number of cores to use to parallelize the extraction process.
        default is the number of available cores on the system.

    Returns
    -------
    db : Database
        the created db
    """
    walker = AudioFileWalker(roots, files)
    tmp_dbs_infos = _make_db_for_each_file(walker, extract_func, n_cores)
    _aggregate_dbs(db_name, tmp_dbs_infos, "w")
    return Database(db_name)
