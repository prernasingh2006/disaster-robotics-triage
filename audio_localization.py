"""
Disaster-Response Triage sub-system 1: TDOA-based audio source localization,
paired with a mel-spectrogram sound classifier that flags candidate
"victim audio" (calls for help, tapping, breathing) vs. background noise.

Two independent pieces here, run together:
  1. localize_source()   -> estimates (x, y) of a sound source from a
                             4-mic array using Generalized Cross-Correlation
                             with Phase Transform (GCC-PHAT) for TDOA,
                             then multilateration.
  2. SoundEventClassifier -> extracts mel-spectrogram features from an
                             audio clip and classifies it. In the full
                             pipeline this is a CNN (torch); here it's a
                             lightweight sklearn MLP trained on the same
                             feature representation so the module runs
                             without a deep learning dependency / GPU.

Both feed victim_reid.py and urgency_scoring.py downstream.
"""

import numpy as np
from scipy.signal import fftconvolve
from scipy.optimize import least_squares
from sklearn.neural_network import MLPClassifier

SPEED_OF_SOUND = 343.0  # m/s


# ---------------------------------------------------------------------------
# 1. TDOA localization
# ---------------------------------------------------------------------------

def gcc_phat(sig, ref, fs, max_tau=None, interp=16):
    """Estimate the time delay between `sig` and `ref` using GCC-PHAT.

    Returns the delay (seconds) of `sig` relative to `ref`. Positive delay
    means `sig` arrived after `ref`.
    """
    n = sig.shape[0] + ref.shape[0]
    SIG = np.fft.rfft(sig, n=n)
    REF = np.fft.rfft(ref, n=n)
    R = SIG * np.conj(REF)

    denom = np.abs(R)
    denom[denom < 1e-12] = 1e-12
    R /= denom  # phase transform: keep phase, discard magnitude

    cc = np.fft.irfft(R, n=interp * n)
    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = min(int(interp * fs * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))
    shift = np.argmax(np.abs(cc)) - max_shift
    tau = shift / float(interp * fs)
    return tau


def localize_source(mic_positions, signals, fs, ref_idx=0):
    """Estimate a 2D sound source position from a synchronized mic array.

    mic_positions : (M, 2) array of mic (x, y) coordinates in meters
    signals       : list of M 1D numpy arrays, same length, synchronized
    fs            : sample rate (Hz)
    ref_idx       : index of the reference microphone for TDOA pairs

    Returns (x, y) estimate and the per-mic TDOA residual (for confidence).
    """
    mic_positions = np.asarray(mic_positions, dtype=float)
    ref_sig = signals[ref_idx]
    ref_pos = mic_positions[ref_idx]

    tdoas = []
    other_positions = []
    for i, sig in enumerate(signals):
        if i == ref_idx:
            continue
        tau = gcc_phat(sig, ref_sig, fs, max_tau=0.005)
        tdoas.append(tau)
        other_positions.append(mic_positions[i])
    tdoas = np.array(tdoas)
    other_positions = np.array(other_positions)

    def residuals(source):
        d_ref = np.linalg.norm(source - ref_pos)
        preds = []
        for pos in other_positions:
            d = np.linalg.norm(source - pos)
            preds.append((d - d_ref) / SPEED_OF_SOUND)
        return np.array(preds) - tdoas

    x0 = mic_positions.mean(axis=0)  # start the solve at array centroid
    result = least_squares(residuals, x0)
    residual_norm = float(np.linalg.norm(result.fun))
    return tuple(result.x), residual_norm


# ---------------------------------------------------------------------------
# 2. Sound event classification (mel-spectrogram features -> MLP)
# ---------------------------------------------------------------------------

def mel_filterbank(n_fft, n_mels, fs, fmin=20, fmax=None):
    fmax = fmax or fs / 2
    def hz_to_mel(f):
        return 2595 * np.log10(1 + f / 700)
    def mel_to_hz(m):
        return 700 * (10 ** (m / 2595) - 1)

    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_points = mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz_points / fs).astype(int)

    fbank = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        f_left, f_center, f_right = bins[m - 1], bins[m], bins[m + 1]
        for k in range(f_left, f_center):
            if 0 <= k < fbank.shape[1] and f_center != f_left:
                fbank[m - 1, k] = (k - f_left) / (f_center - f_left)
        for k in range(f_center, f_right):
            if 0 <= k < fbank.shape[1] and f_right != f_center:
                fbank[m - 1, k] = (f_right - k) / (f_right - f_center)
    return fbank


def mel_spectrogram(signal, fs, n_fft=512, hop=256, n_mels=32):
    window = np.hanning(n_fft)
    n_frames = 1 + (len(signal) - n_fft) // hop
    n_frames = max(n_frames, 1)
    spec = np.zeros((n_fft // 2 + 1, n_frames))
    for t in range(n_frames):
        start = t * hop
        frame = signal[start:start + n_fft]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        spec[:, t] = np.abs(np.fft.rfft(frame * window)) ** 2

    fbank = mel_filterbank(n_fft, n_mels, fs)
    mel_spec = fbank @ spec
    log_mel = np.log(mel_spec + 1e-8)
    return log_mel  # (n_mels, n_frames)


def clip_features(signal, fs):
    """Fixed-length feature vector for a clip: mean + std of each mel band."""
    m = mel_spectrogram(signal, fs)
    return np.concatenate([m.mean(axis=1), m.std(axis=1)])


class SoundEventClassifier:
    """Lightweight stand-in for the production CNN classifier.

    Same mel-spectrogram front end as the CNN pipeline; here it feeds a
    small MLP so this module has no deep-learning framework dependency.
    Swap `self.model` for a torch CNN over `mel_spectrogram()` output
    without changing anything downstream.
    """

    LABELS = ["background", "tapping", "voice_call", "breathing"]

    def __init__(self, seed=0):
        self.model = MLPClassifier(
            hidden_layer_sizes=(32, 16),
            max_iter=2000,
            random_state=seed,
        )
        self._fitted = False

    def fit(self, clips, labels, fs):
        X = np.array([clip_features(c, fs) for c in clips])
        self.model.fit(X, labels)
        self._fitted = True
        return self

    def predict(self, clip, fs):
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")
        x = clip_features(clip, fs).reshape(1, -1)
        label = self.model.predict(x)[0]
        proba = self.model.predict_proba(x)[0]
        confidence = float(proba.max())
        return label, confidence