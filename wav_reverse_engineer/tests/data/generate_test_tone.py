"""Generate a simple test tone WAV file for smoke testing."""
import os
import numpy as np
import soundfile as sf

os.makedirs(os.path.dirname(__file__) or '.', exist_ok=True)

sr = 22050
sec = 1.0
t = np.linspace(0, sec, int(sr * sec), endpoint=False)
x = 0.2 * np.sin(2 * np.pi * 440 * t)

out_path = os.path.join(os.path.dirname(__file__), 'test_tone.wav')
sf.write(out_path, x, sr)
print(f"Wrote {out_path}")
