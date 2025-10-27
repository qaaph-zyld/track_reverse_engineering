import os
import numpy as np
import librosa
from typing import Dict, Any
from scipy.signal import find_peaks

try:
    import pyloudnorm as pyln
except Exception:
    pyln = None


def _short_term_rms(x: np.ndarray, frame_samples: int, hop_samples: int) -> np.ndarray:
    if len(x) < frame_samples:
        return np.array([])
    frames = []
    for i in range(0, len(x) - frame_samples + 1, hop_samples):
        frames.append(np.sqrt(np.mean(x[i:i+frame_samples] ** 2)))
    return np.asarray(frames)


def estimate_rt60(audio: np.ndarray, sr: int) -> float:
    y = librosa.util.normalize(audio)
    energy = y ** 2
    edc = np.flip(np.cumsum(np.flip(energy)))
    edc_db = 10 * np.log10(edc + 1e-12)
    edc_db = edc_db - np.max(edc_db)
    idx_start = np.argmax(edc_db <= -5)
    idx_end = np.argmax(edc_db <= -35)
    if idx_start == 0 or idx_end == 0 or idx_end <= idx_start:
        return 0.0
    x = np.arange(idx_start, idx_end) / sr
    y_db = edc_db[idx_start:idx_end]
    A = np.vstack([x, np.ones_like(x)]).T
    m, c = np.linalg.lstsq(A, y_db, rcond=None)[0]
    if m == 0:
        return 0.0
    rt60 = -60.0 / m
    return float(max(rt60, 0.0))


def spectral_tilt(audio: np.ndarray, sr: int) -> float:
    S = np.abs(librosa.stft(audio, n_fft=4096, hop_length=1024))
    mag = np.mean(S, axis=1) + 1e-12
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
    x = np.log10(freqs[1:])
    y = 20 * np.log10(mag[1:])
    A = np.vstack([x, np.ones_like(x)]).T
    m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(m)


def harmonic_distortion(audio: np.ndarray, sr: int) -> float:
    S = np.abs(librosa.stft(audio, n_fft=8192, hop_length=2048))
    spec = np.mean(S, axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=8192)
    idx_peak = np.argmax(spec[1:]) + 1
    f0 = freqs[idx_peak]
    if f0 <= 20:
        return 0.0
    harmonics = []
    for k in [2, 3, 4, 5]:
        fk = k * f0
        if fk >= sr / 2:
            break
        idx = np.argmin(np.abs(freqs - fk))
        harmonics.append(spec[idx])
    if not harmonics:
        return 0.0
    thd = np.sqrt(np.sum(np.square(harmonics))) / (spec[idx_peak] + 1e-9)
    return float(thd)


def compression_index(audio: np.ndarray, sr: int) -> float:
    frame = int(0.1 * sr)
    hop = int(0.05 * sr)
    rms = _short_term_rms(audio, frame, hop)
    if rms.size == 0:
        return 0.0
    peak = np.max(np.abs(audio)) + 1e-9
    crest = peak / (np.mean(rms) + 1e-9)
    var = float(np.var(rms))
    idx = float(1.0 / (crest + 1e-9)) * (1.0 / (var + 1e-6))
    return idx


def loudness_metrics(audio: np.ndarray, sr: int) -> Dict[str, float]:
    if pyln is None:
        return {}
    # Ensure correct dtype for pyloudnorm
    y = np.asarray(audio, dtype=np.float64)
    meter = pyln.Meter(sr)
    loudness = float(meter.integrated_loudness(y))
    lra = None
    # Some pyloudnorm versions don't expose Meter.loudness_range
    if hasattr(meter, "loudness_range"):
        try:
            lra = float(meter.loudness_range(y))
        except Exception:
            lra = None
    # Try module-level loudness_range/lra if available
    if lra is None:
        try:
            from pyloudnorm import loudness as _pln_loud
            if hasattr(_pln_loud, "loudness_range"):
                lra = float(_pln_loud.loudness_range(y, sr))
            elif hasattr(_pln_loud, "lra"):
                lra = float(_pln_loud.lra(y, sr))
        except Exception:
            lra = None
    # Fallback: approximate LRA from short-term RMS distribution (3s windows)
    if lra is None:
        win = int(3.0 * sr)
        hop = int(1.0 * sr)
        st_rms = _short_term_rms(y, win, hop)
        if st_rms.size > 0:
            st_db = 20.0 * np.log10(st_rms + 1e-12)
            p95 = float(np.percentile(st_db, 95))
            p10 = float(np.percentile(st_db, 10))
            lra = max(p95 - p10, 0.0)
        else:
            lra = 0.0
    return {"loudness_lufs": loudness, "loudness_range": float(lra)}


def analyze_effects(audio: np.ndarray, sr: int) -> Dict[str, Any]:
    rt60 = estimate_rt60(audio, sr)
    tilt = spectral_tilt(audio, sr)
    thd = harmonic_distortion(audio, sr)
    comp = compression_index(audio, sr)
    loud = loudness_metrics(audio, sr)
    out = {
        "rt60_seconds": rt60,
        "spectral_tilt_db_per_decade": tilt,
        "thd_ratio": thd,
        "compression_index": comp,
    }
    out.update(loud)
    return out
