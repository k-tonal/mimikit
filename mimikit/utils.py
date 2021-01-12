import librosa
import numpy as np
from librosa.display import specshow
import IPython.display as ipd
import matplotlib.pyplot as plt

from .data.transforms import HOP_LENGTH, SR

# Conversion

a2db = lambda S: librosa.amplitude_to_db(abs(S), ref=S.max())


# Debugging utils

def to_db(S):
    if S.dtype == np.complex64:
        S_hat = a2db(abs(S)) + 40
    elif S.min() >= 0 and S.dtype in (np.float, np.float32, np.float64, np.float_):
        S_hat = a2db(S) + 40
    else:
        S_hat = a2db(S) + 40
    return S_hat


def signal(S, hop_length=HOP_LENGTH):
    if S.dtype in (np.complex64, np.complex128):
        return librosa.istft(S, hop_length=hop_length)
    else:
        return librosa.griffinlim(S, hop_length=hop_length, n_iter=32)


def audio(S, hop_length=HOP_LENGTH, sr=SR):
    if len(S.shape) > 1:
        y = signal(S, hop_length)
        if y.size > 0:
            return ipd.display(ipd.Audio(y, rate=sr))
        else:
            return ipd.display(ipd.Audio(np.zeros(hop_length*2), rate=sr))
    else:
        return ipd.display(ipd.Audio(S, rate=sr))


def show(S, figsize=(), db_scale=True, title="", **kwargs):
    S_hat = to_db(S) if db_scale else S
    if figsize:
        plt.figure(figsize=figsize)
    if "x_axis" not in kwargs:
        kwargs["x_axis"] = "frames"
    if "y_axis" not in kwargs:
        kwargs["y_axis"] = "frames"
    ax = specshow(S_hat, sr=SR, **kwargs)
    plt.colorbar()
    plt.tight_layout()
    plt.title(title)
    return ax

