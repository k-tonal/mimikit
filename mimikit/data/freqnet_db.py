"""
FreqNet Database
================
"""

import argparse
from functools import partial
from multiprocessing import cpu_count

from . import Database, make_root_db, file_to_fft, AudioFileWalker
from ..connectors.neptune import NeptuneConnector


parser = argparse.ArgumentParser(prog="freqnet-db",
                                 description="transform found audio files to FFTs with specified parameters and put "
                                             "them in a .h5 Database that freqnets can consume")

parser.add_argument("target", type=str,
                    help="the name (path) of the .h5 db you want to create")
parser.add_argument("--roots", "-r",
                    type=str, nargs="*",
                    default=None,
                    help="list of paths from which to search for audio files to include in the db")
parser.add_argument("--files", "-f",
                    type=str, nargs="*",
                    default=None,
                    help="list of audio files to include in the db")
parser.add_argument("--n-fft", '-d',
                    type=int, default=2048,
                    help="the fft size used to transform the files (default=2048)")
parser.add_argument("--hop-length", "-o",
                    type=int, default=512,
                    help="the hop length used to transform the files (default=512)")
parser.add_argument("--sample-rate", "-s",
                    type=int, default=22050,
                    help="the sample rate used to transform the files (default=22050)")
parser.add_argument("--neptune-path", '-p',
                    type=str, default=None,
                    help="path to the neptune.ai project or experiment you wish to upload the db to"
                         " (requires that you stored your neptune api token in the environment of this script)")


def freqnet_db(target,
               roots=None,
               files=None,
               n_fft=2048,
               hop_length=512,
               sample_rate=22050,
               neptune_path=None):
    """
    transform found audio files to STFTs with specified parameters and put
    them in a .h5 ``Database``

    Parameters
    ----------
    target : str
        the name (or path) of the .h5 db you want to create
    roots : str or list of str, optional
        list of paths from which to search for audio files to include in the db
    files : str or list of str, optional
        list of audio files to include in the db

        .. note::
            if no ``roots`` and no ``files`` are provided, the fucntion will search
            for audio files in the current working directory.

    n_fft : int, optional
        the fft size used to transform the files (default is 2048)
    hop_length : int, optional
        the hop length used to transform the files (default is 512)
    sample_rate : int, optional
        the sample rate used to transform the files (default is 22050)
    neptune_path : str or None, optional
        name of the neptune.ai project you wish to upload the db to.
        `neptune_path` is expected to be of the form `"username/project"`.
        Requires that you stored your neptune api token in the environment.

    Returns
    -------
    db : Database
        the created db
    """
    namespace = argparse.Namespace(target=target, roots=roots, files=files, n_fft=n_fft,
                                   hop_length=hop_length, sample_rate=sample_rate,
                                   neptune_path=neptune_path)
    main(namespace)
    return Database(target)


def main(namespace=None):
    if namespace is None:
        args = parser.parse_args()
    else:
        args = namespace
    transform = partial(file_to_fft,
                        n_fft=args.n_fft,
                        hop_length=args.hop_length,
                        sr=args.sample_rate)
    if args.roots is None and args.files is None:
        args.roots = "./"
    walker = AudioFileWalker(roots=args.roots, files=args.files)
    print("Found following audio files :", "\n", *["\t" + file + "\n" for file in list(walker)])
    print("Making the database...")
    make_root_db(args.target, roots=args.roots, files=args.files, extract_func=transform, n_cores=cpu_count()//2)

    if args.neptune_path is not None:
        path = args.neptune_path.split("/")
        user, rest = path[0], "/".join(path[1:]) if path[-1] is not None else path[1]
        connector = NeptuneConnector(user=user, setup={"db": rest})
        db = Database(args.target)
        print("uploading database to neptune...")
        connector.upload_database("db", db)
    print("done!")
