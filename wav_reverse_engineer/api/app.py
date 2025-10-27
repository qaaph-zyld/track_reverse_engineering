"""
Simple FastAPI microservice for headless analysis.
Install with extras: pip install -e .[api]
Run: uvicorn wav_reverse_engineer.api.app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse

from ..audio_analyzer.audio_processor import AudioProcessor
from ..audio_analyzer.feature_extractor import FeatureExtractor
from ..audio_analyzer.effects_analyzer import analyze_effects
from ..audio_analyzer.instrument_recognizer import InstrumentRecognizer
from ..audio_analyzer.source_separation import separate_hpss
from ..audio_analyzer.backends.essentia_metrics import compute_essentia_metrics

app = FastAPI(title="WAV Reverse Engineering API", version="0.1.0")


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    effects: bool = Form(False),
    instruments: bool = Form(False),
    essentia: bool = Form(False),
):
    # Save upload to temp
    suffix = f".{file.filename.split('.')[-1].lower()}" if file.filename else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        temp_path = tmp.name

    try:
        ap = AudioProcessor()
        y, sr = ap.load_audio(temp_path, target_sr=22050, mono=True)
        fe = FeatureExtractor()
        features = fe.extract_features(y, sr)
        if essentia:
            try:
                ess = compute_essentia_metrics(y, sr)
                if ess:
                    features.update(ess)
            except Exception:
                pass
        # Chords (simple)
        chords = fe.detect_chords(y, sr)
        features['chords'] = [{
            'root': ch.root.name,
            'quality': ch.quality,
            'confidence': ch.confidence,
            'start_time': ch.start_time,
            'duration': ch.duration
        } for ch in chords]
        features['chord_progression'] = FeatureExtractor.summarize_chord_progression(chords)
        # Notes
        features['notes'] = fe.detect_notes(y, sr)
        if effects:
            try:
                features['effects'] = analyze_effects(y, sr)
            except Exception as e:
                features['effects_error'] = str(e)
        if instruments:
            recog = InstrumentRecognizer()
            features['instruments'] = recog.recognize(y, sr)
        return JSONResponse(features)
    finally:
        try:
            os.unlink(temp_path)
        except Exception:
            pass
