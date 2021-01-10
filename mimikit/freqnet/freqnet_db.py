"""
FreqNet Database
================
"""

import argparse
import os
from functools import partial
from itertools import tee
from multiprocessing import cpu_count

from mimikit.data import Database, make_root_db, file_to_fft, AudioFileWalker, upload_database

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
parser.add_argument("--neptune-project", '-p',
                    type=str, default=None,
                    help="name of the neptune.ai project you wish to upload the db to (requires that you stored your"
                         " neptune api token in the environment of this script)")


def freqnet_db(target,
               roots=None,
               files=None,
               n_fft=2048,
               hop_length=512,
               sample_rate=22050,
               neptune_project=None):
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
    neptune_project : str or None, optional
        name of the neptune.ai project you wish to upload the db to.
        Requires that you stored your neptune api token in the environment.

    Returns
    -------
    Database

    """
    namespace = argparse.Namespace(target=target, roots=roots, files=files, n_fft=n_fft,
                                   hop_length=hop_length, sample_rate=sample_rate,
                                   neptune_project=neptune_project)
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
    walker = AudioFileWalker()
    walker, backup = tee(walker)
    print("Found following audio files :", "\n", *["\t" + file + "\n" for file in list(backup)])
    print("Making the database...")
    make_root_db(args.target, roots=args.roots, files=args.files, extract_func=transform, n_cores=cpu_count()//2)

    if args.neptune_project is not None:
        token = os.environ["NEPTUNE_API_TOKEN"]
        db = Database(args.target)
        print("uploading database to neptune...")
        upload_database(db, token, args.neptune_project, args.target)
    print("done!")