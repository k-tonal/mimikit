import pytest
import numpy as np
import pandas as pd
import soundfile
import h5py

from mimikit.data.factory import AudioFileWalker, file_to_db, make_root_db
from mimikit.data import default_extract_func, DataObject, Database


@pytest.fixture
def audio_tree(tmp_path):
    root = (tmp_path / "audios")
    root.mkdir()
    dir1 = (root / "dir1")
    dir1.mkdir()
    dir2 = (root / "dir2")
    dir2.mkdir()
    extensions = [".aif",
                  ".wav",
                  ".mp3",
                  ".aiff",
                  ".notaudio",
                  ""]
    tests = [np.random.randn(10000) for _ in range(len(extensions))]
    for i, arr, ext in zip(range(len(extensions)), tests, extensions):
        soundfile.write(str([dir1, dir2][i % 2]) + "/test" + str(i) + ext, arr, 22050, 'PCM_24', format="WAV")
    return str(root)


def test_audio_file_walker(audio_tree):
    walker = AudioFileWalker(roots=audio_tree)
    assert len(list(walker)) == 4
    walker = AudioFileWalker(files=[audio_tree + "/dir2/test1.wav",
                                    audio_tree + "/dir1/test4.notaudio"])
    assert len(list(walker)) == 1


def test_default_extract_func(audio_tree):
    dic = default_extract_func(audio_tree + "/dir2/test1.wav")
    # we must have regions
    assert "regions" in dic and isinstance(dic["regions"][1], pd.DataFrame), dic["regions"]
    # regions + feature
    assert len(dic) == 2


def test_file_to_db(audio_tree):

    file_path = audio_tree + "/dir2/test1.wav"

    def faulty_extract_func(path):
        dic = default_extract_func(path)
        dic.pop("regions")
        return dic

    with pytest.raises(ValueError, match=r".*regions.*"):
        file_to_db(file_path, faulty_extract_func)

    file_to_db(file_path)
    with h5py.File(file_path.split(".")[0] + ".h5", "r") as f:
        assert f["regions"] is not None


def test_make_root_db_and_Database(audio_tree):
    make_root_db(audio_tree + "/test_db.h5", roots=audio_tree, extract_func=default_extract_func, n_cores=1)

    db = Database(audio_tree + "/test_db.h5")
    assert db.regions is not pd.DataFrame()
    assert len(db.regions) == 4, len(db.regions)
    assert np.any(db.fft[:4] != 0), db.fft[:4]
    assert isinstance(db.fft.get(db.regions.iloc[:1]), np.ndarray), db.fft.get(db.regions.iloc[:1])

    # test Dataset Integration
    ds = DataObject(db.fft)
    assert ds is not None, ds
    assert len(ds) == len(db.fft), (len(ds), len(db.fft))
    assert np.all(ds[:10] == db.fft[:10])
