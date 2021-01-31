import numpy as np
from librosa.segment import recurrence_matrix
from librosa.util import localmax
from scipy.ndimage import convolve
from ..data.regions import Regions


def from_recurrence_matrix(X,
                           L=6,
                           k=None,
                           sym=True,
                           bandwidth=1.,
                           thresh=0.2,
                           min_dur=4):
    def checker(N):
        block = np.zeros((N * 2 + 1, N * 2 + 1), dtype=np.int32)
        for k in range(-N, N + 1):
            for l in range(-N, N + 1):
                block[k + N, l + N] = np.sign(k) * np.sign(l)
        return block / abs(block).sum()

    R = recurrence_matrix(
        X, metric="cosine", mode="affinity",
        k=k, sym=sym, bandwidth=bandwidth, self=True)
    # intensify checker-board-like entries
    R_hat = convolve(R, checker(L),
                     mode="constant")
    # extract them along the main diagonal
    dg = np.diag(R_hat, 0)
    mx = localmax(dg * (dg > thresh)).nonzero()[0]
    # filter out maxes less than min_dur frames away of the previous max
    mx = mx * (np.diff(mx, append=R.shape[0]) >= min_dur)
    mx = mx[mx > 0]
    stops = np.r_[mx, R.shape[0]]
    return Regions.from_stop(stops)