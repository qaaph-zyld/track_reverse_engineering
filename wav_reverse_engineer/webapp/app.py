import os
import io
import json
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

# Ensure we can import the local package
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audio_analyzer.audio_processor import AudioProcessor
from audio_analyzer.feature_extractor import FeatureExtractor
from audio_analyzer.visualizer import AudioVisualizer
from audio_analyzer.effects_analyzer import analyze_effects
from audio_analyzer.source_separation import separate_hpss, separate_spleeter, export_stems
from audio_analyzer.instrument_recognizer import InstrumentRecognizer

st.set_page_config(page_title="WAV Reverse Engineering Tool", layout="wide")
st.title("WAV Reverse Engineering Tool")
st.write("Analyze and reverse engineer WAV audio tracks: chords, tempo, key, instruments, effects, and stems.")

uploaded_file = st.file_uploader("Upload an audio file (WAV/MP3)", type=["wav", "mp3", "flac", "ogg"])

col1, col2, col3 = st.columns(3)
with col1:
    run_effects = st.checkbox("Advanced Effects", value=True)
with col2:
    run_instruments = st.checkbox("Instrument Recognition", value=True)
with col3:
    separation_method = st.selectbox("Source Separation", ["none", "hpss", "spleeter2", "spleeter4", "spleeter5"], index=1)

analyze_btn = st.button("Analyze")

if uploaded_file and analyze_btn:
    with st.spinner("Processing audio..."):
        # Save to temp file
        suffix = f".{uploaded_file.name.split('.')[-1].lower()}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name
        audio_bytes = None
        try:
            audio_bytes = Path(temp_path).read_bytes()
        except Exception:
            pass

        # Show audio player
        if audio_bytes:
            st.audio(audio_bytes)

        # Load and analyze
        ap = AudioProcessor()
        audio, sr = ap.load_audio(temp_path, target_sr=22050, mono=True)

        fe = FeatureExtractor()
        features = fe.extract_features(audio, sr)

        st.subheader("Key Metrics")
        km_cols = st.columns(4)
        km_cols[0].metric("Tempo (BPM)", f"{features.get('tempo', 0):.1f}")
        key_display = f"{features.get('key','?')} {features.get('mode','')}".strip()
        km_cols[1].metric("Key", key_display)
        km_cols[2].metric("Duration (s)", f"{features.get('duration', 0):.2f}")
        km_cols[3].metric("RMS", f"{features.get('rms_energy', 0):.4f}")

        st.subheader("Chords and Notes")
        chords = fe.detect_chords(audio, sr)
        features['chords'] = [{
            'root': ch.root.name,
            'quality': ch.quality,
            'confidence': ch.confidence,
            'start_time': ch.start_time,
            'duration': ch.duration
        } for ch in chords]
        progression = FeatureExtractor.summarize_chord_progression(chords)
        features['chord_progression'] = progression
        st.write("Chord progression (condensed):")
        st.dataframe(progression)

        notes = fe.detect_notes(audio, sr)
        features['notes'] = notes
        with st.expander("Detected notes"):
            st.dataframe(notes)

        if run_effects:
            st.subheader("Effects Analysis")
            eff = analyze_effects(audio, sr)
            features['effects'] = eff
            st.json(eff)

        if run_instruments:
            st.subheader("Instrument Recognition")
            recog = InstrumentRecognizer()
            inst = recog.recognize(audio, sr)
            features['instruments'] = inst
            st.table(inst)

        # Separation
        stems_paths = None
        if separation_method != 'none':
            st.subheader("Source Separation")
            if separation_method == 'hpss':
                stems = separate_hpss(audio)
                stems_sr = sr
            else:
                try:
                    stems_count = int(separation_method.replace('spleeter', ''))
                except Exception:
                    stems_count = 2
                stems = separate_spleeter(temp_path, stems=stems_count) or {}
                stems_sr = stems.get('sample_rate', sr)
            if stems:
                features['stems'] = [k for k in stems.keys() if k != 'sample_rate']
                stems_dir = os.path.join(tempfile.gettempdir(), 'wav_re_separation')
                stems_paths = export_stems(stems, stems_sr, stems_dir, prefix=Path(temp_path).stem)
                features['stems_paths'] = stems_paths
                st.write("Exported stems:")
                for name, path in stems_paths.items():
                    st.write(f"- {name}: {path}")
            else:
                st.info("Spleeter not available or separation failed; try HPSS.")

        # Visualizations
        st.subheader("Visualizations")
        vis_dir = os.path.join(tempfile.gettempdir(), 'wav_re_vis')
        os.makedirs(vis_dir, exist_ok=True)
        vis = AudioVisualizer()
        report_paths = vis.generate_analysis_report(audio, sr, output_dir=vis_dir, prefix=Path(temp_path).stem)
        cols = st.columns(2)
        cols[0].image(report_paths['waveform'], caption='Waveform', use_column_width=True)
        cols[1].image(report_paths['spectrogram'], caption='Spectrogram', use_column_width=True)
        cols = st.columns(2)
        cols[0].image(report_paths['mel_spectrogram'], caption='Mel Spectrogram', use_column_width=True)
        cols[1].image(report_paths['chroma'], caption='Chroma', use_column_width=True)
        st.image(report_paths['beat_tracking'], caption='Beat Tracking', use_column_width=True)
        st.image(report_paths['pitch_tracking'], caption='Pitch Tracking', use_column_width=True)

        # Download JSON
        st.subheader("Download Results")
        json_bytes = json.dumps(features, indent=2).encode('utf-8')
        st.download_button("Download analysis.json", data=json_bytes, file_name="analysis.json", mime="application/json")

        # Cleanup temp file left on disk
        # (Keep for reproducibility during session)
