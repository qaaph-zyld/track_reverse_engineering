import numpy as np
import librosa
from typing import Dict, Any


def track_f0_torchcrepe(audio: np.ndarray, sr: int, hop_length: int = 160, model: str = 'full') -> Dict[str, Any]:
    """Track F0 using torchcrepe if available. Returns dict with times, f0_hz, periodicity.
    Audio is resampled to 16 kHz as required by torchcrepe.
    """
    try:
        import torch
        import torchcrepe
    except Exception:
        return {}

    # Resample to 16kHz for torchcrepe
    target_sr = 16000
    if sr != target_sr:
        y = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    else:
        y = audio

    # Prepare tensor (1, time)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(0)

    fmin, fmax = 50.0, 1100.0
    device = 'cpu'
    with torch.no_grad():
        f0, periodicity, _ = torchcrepe.predict(
            y_t, target_sr, hop_length, fmin, fmax,
            model=model, batch_size=1024, return_periodicity=True, device=device
        )

    f0 = f0.squeeze(0).cpu().numpy()
    periodicity = periodicity.squeeze(0).cpu().numpy()
    times = np.arange(len(f0)) * (hop_length / float(target_sr))

    return {
        'times': times.tolist(),
        'f0_hz': f0.tolist(),
        'periodicity': periodicity.tolist(),
        'sample_rate': target_sr,
        'hop_length': hop_length
    }
