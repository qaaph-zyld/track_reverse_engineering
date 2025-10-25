import os
from typing import Dict, Any


def separate_demucs(file_path: str, model_name: str = 'htdemucs') -> Dict[str, Any]:
    """Separate sources using Demucs if installed. Returns dict of stems or empty dict.
    This is a best-effort wrapper and will return {} if Demucs is unavailable.
    """
    try:
        import torch  # noqa: F401
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
        from demucs.audio import AudioFile
    except Exception:
        return {}

    try:
        model = get_model(model_name)
        model.to('cpu')
        wav = AudioFile(file_path).read(streams=0, samplerate=model.samplerate, channels=2)
        # wav shape: (channels, time)
        with torch.no_grad():
            # out shape: (sources, channels, time)
            out = apply_model(model, wav[None], device='cpu')[0]
        stems = {}
        for i, name in enumerate(model.sources):
            # Convert to mono
            mono = out[i].mean(0).cpu().numpy()
            stems[name] = mono
        stems['sample_rate'] = model.samplerate
        return stems
    except Exception:
        return {}
