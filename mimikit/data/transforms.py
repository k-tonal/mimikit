import librosa
from .regions import Regions

N_FFT = 2048
HOP_LENGTH = 512
SR = 22050
MU = 255


def stft(file, n_fft=N_FFT, hop_length=HOP_LENGTH, sr=SR):
    y, sr = librosa.load(file, sr=sr)
    fft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    # returns the feature and its attributes
    return fft, dict(n_fft=n_fft, hop_length=hop_length, sr=sr)


def file_to_fft(abs_path, n_fft=N_FFT, hop_length=HOP_LENGTH, sr=SR):
    fft, params = stft(abs_path, n_fft, hop_length, sr)
    fft = abs(fft)
    regions = Regions.from_duration([fft.shape[1]])
    params.update(dict(time_axis=0))
    return dict(fft=(params, fft.T), regions=({}, regions))


def mu_law_compress(file, mu=MU, sr=SR):
    y, sr = librosa.load(file, sr=sr)
    y = librosa.util.normalize(y)
    qx = librosa.mu_compress(y, mu, quantize=True)
    qx = qx + (MU + 1) // 2
    return qx, dict(mu=mu, sr=sr)


def file_to_qx(abs_path, mu=MU, sr=SR):
    qx, params = mu_law_compress(abs_path, mu, sr)
    regions = Regions.from_duration([qx.shape[0]])
    return dict(qx=(params, qx.reshape(-1, 1)), regions=({}, regions))


default_extract_func = file_to_fft
