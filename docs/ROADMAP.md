# Track Reverse Engineering: Roadmap

This document captures current capabilities, limitations, and the phased plan to reach production-grade quality using free and open-source components.

## Current capabilities (hard truth)

- **Core analysis**
  - Tempo/beat/onset (librosa/madmom)
  - Key/mode (HPCP-style heuristic)
  - Spectral features (centroid/bandwidth/contrast/MFCCs)
  - Chord detection (simple chroma + templates)
  - Note detection (onsets + YIN on segments)
- **Effects analysis**
  - RT60 decay estimate, spectral tilt, THD proxy, compression index
  - Integrated LUFS/LRA when `pyloudnorm` is available
- **Source separation**
  - HPSS always available
  - Optional Spleeter 2/4/5 stems (heavy dependency)
- **Instrument recognition**
  - PANNs (if installed) or heuristic fallback (coarse: drums, bass, vocals, guitar/piano)
- **Deliverables**
  - CLI with options for effects, instruments, separation, stems export
  - Streamlit web UI (local) with visuals and downloads
  - Static docs site (for Netlify)

## Known limitations

- Key/mode and chords are heuristic; unreliable under modulation, dense mixes, heavy distortion
- Note detection is monophonic-oriented; polyphonic transcription not solved yet
- Effects analysis is indicative, not engineering-grade; room/genre dependent
- PANNs and Spleeter are heavy; CPU can be slow; installation friction on Windows for some optional tools
- No plugin system; limited caching; evaluation not automated yet

## Phase 1: Quality jump with proven OSS (High priority)

1. Chords (robust)
   - Integrate Chordino (Vamp plugin via Sonic Annotator) for chord and key estimation
   - Provide wrapper and parsers; compare outputs vs internal method
2. Pitch/transcription (accurate)
   - Integrate CREPE or Basic Pitch for high-accuracy F0; evaluate monophonic performance
3. Separation (SOTA option)
   - Add Demucs/HTDemucs backend (torch); keep HPSS/Spleeter as fallbacks; cache models
4. Key/mode/loudness (robust)
   - Integrate Essentia for HPCP-based key, loudness, and descriptors; optional plugin

## Phase 2: Architecture and ops

- Plugin/registry to load optional providers lazily; settings file (YAML/TOML)
- Caching/artifact management using content hashes (.cache)
- Batch/CLI UX: resume, concurrency, progress, schema versioning
- Evaluation harness with `mir_eval` and public datasets (MedleyDB, Isophonics); CI jobs
- Packaging: `extras_require` groups, Windows notes, Dockerfile (CPU and CUDA)

## Phase 3: UI and deployment

- Host a CPU-friendly subset on Streamlit Community Cloud or Hugging Face Spaces
- Keep Netlify for docs; add guides and examples
- Optional API mode (FastAPI microservice) for headless runs

## Milestones and acceptance

- M1: Chordino wrapper merged; CLI flag to choose backend; tests; baseline metrics
- M2: CREPE/Basic Pitch integrated; improved note F0 accuracy; tests/benchmarks
- M3: Demucs backend available; model caching; performance checks
- M4: Essentia plugin; key and loudness metrics validated against datasets
- M5: Plugin system + caching; CI with evaluation harness; Docker images
- M6: Hosted demo app; comprehensive docs and examples

## Risks and mitigations

- Windows install friction (Spleeter, Essentia, Vamp): keep optional, document clearly, provide Docker
- Model sizes and runtime: cache models, allow CPU/GPU toggles, pre-download instructions
- Licensing: only use OSS-compatible models/tools; document licenses
