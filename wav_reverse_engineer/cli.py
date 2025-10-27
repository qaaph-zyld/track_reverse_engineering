"""
Command-line interface for the WAV Reverse Engineering Tool.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import matplotlib
import librosa
import os.path as _osp

# Use non-interactive backend for CLI
matplotlib.use('Agg')

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.absolute()))

from audio_analyzer.audio_processor import AudioProcessor
from audio_analyzer.feature_extractor import FeatureExtractor, DetectedChord, NoteName
from audio_analyzer.visualizer import AudioVisualizer
from audio_analyzer.utils import save_analysis_results, ensure_dir, get_file_hash, split_audio
from audio_analyzer.effects_analyzer import analyze_effects
from audio_analyzer.source_separation import separate_hpss, separate_spleeter, export_stems
from audio_analyzer.instrument_recognizer import InstrumentRecognizer
from audio_analyzer.backends.chordino import detect_chords_chordino
from audio_analyzer.backends.pitch_torchcrepe import track_f0_torchcrepe
from audio_analyzer.backends.demucs_backend import separate_demucs
from audio_analyzer.backends.essentia_metrics import compute_essentia_metrics

class AudioAnalyzerCLI:
    """Command-line interface for audio analysis."""
    
    def __init__(self):
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create the command-line argument parser."""
        parser = argparse.ArgumentParser(
            description='WAV Reverse Engineering Tool - Analyze and extract features from audio files.'
        )
        
        # Main command
        subparsers = parser.add_subparsers(dest='command', help='Command to execute')
        
        # Analyze command
        analyze_parser = subparsers.add_parser('analyze', help='Analyze an audio file')
        analyze_parser.add_argument('input_file', type=str, help='Input audio file (WAV format)')
        analyze_parser.add_argument('-o', '--output-dir', type=str, default='output',
                                  help='Output directory for analysis results (default: ./output)')
        analyze_parser.add_argument('--no-vis', action='store_true',
                                  help='Disable visualization generation')
        analyze_parser.add_argument('--segment-duration', type=float, default=30.0,
                                  help='Duration of audio segments in seconds (default: 30.0)')
        analyze_parser.add_argument('--export-json', action='store_true',
                                  help='Export analysis results as JSON')
        analyze_parser.add_argument('--export-audio', action='store_true',
                                  help='Export processed audio segments')
        analyze_parser.add_argument('--effects', action='store_true',
                                  help='Run advanced effects analysis')
        analyze_parser.add_argument('--instruments', action='store_true',
                                  help='Run instrument recognition')
        analyze_parser.add_argument('--separate', type=str, default='none',
                                  choices=['none', 'hpss', 'spleeter2', 'spleeter4', 'spleeter5', 'demucs'],
                                  help='Perform source separation method')
        analyze_parser.add_argument('--export-stems', action='store_true',
                                  help='Export separated stems as audio files')
        analyze_parser.add_argument('--chord-backend', type=str, default='simple',
                                  choices=['simple', 'chordino'],
                                  help='Chord detection backend (default: simple)')
        analyze_parser.add_argument('--pitch-backend', type=str, default='yin',
                                  choices=['yin', 'torchcrepe'],
                                  help='Pitch tracking backend for F0 curve (default: yin)')
        analyze_parser.add_argument('--essentia', action='store_true',
                                  help='Compute robust key/loudness using Essentia (if available)')
        analyze_parser.add_argument('--cache', action='store_true',
                                  help='Enable result caching to speed up repeated analyses')
        analyze_parser.add_argument('--cache-dir', type=str, default='.cache',
                                  help='Directory to store cache files (default: ./.cache)')
        analyze_parser.add_argument('--config', type=str, default=None,
                                  help='YAML config file to override options (backends, cache, etc.)')
        
        # Batch process command
        batch_parser = subparsers.add_parser('batch', help='Process multiple audio files')
        batch_parser.add_argument('input_dir', type=str, help='Input directory containing audio files')
        batch_parser.add_argument('-o', '--output-dir', type=str, default='batch_output',
                                help='Output directory for analysis results (default: ./batch_output)')
        batch_parser.add_argument('--ext', type=str, default='wav',
                                help='File extension to process (default: wav)')
        batch_parser.add_argument('--no-vis', action='store_true',
                                help='Disable visualization generation')
        batch_parser.add_argument('--export-json', action='store_true',
                                help='Export analysis results as JSON')
        batch_parser.add_argument('--jobs', type=int, default=1,
                                help='Number of parallel workers (default: 1)')
        batch_parser.add_argument('--resume', action='store_true',
                                help='Skip files that already have output JSON in their folder')
        
        # Version command
        subparsers.add_parser('version', help='Show version information')
        
        return parser
    
    def run(self, args=None):
        """Run the CLI with the given arguments."""
        if args is None:
            args = sys.argv[1:]
            
        if not args:  # No arguments provided
            self.parser.print_help()
            return 0
            
        args = self.parser.parse_args(args)
        
        if args.command == 'analyze':
            # Load config if provided and override CLI args
            if getattr(args, 'config', None):
                try:
                    import yaml
                    with open(args.config, 'r') as f:
                        cfg = yaml.safe_load(f) or {}
                    for k, v in cfg.get('analyze', {}).items():
                        if hasattr(args, k):
                            setattr(args, k, v)
                except Exception as _e:
                    print(f"Warning: failed to load config {args.config}: {_e}")
            return self.analyze_audio(
                args.input_file,
                output_dir=args.output_dir,
                generate_visualizations=not args.no_vis,
                segment_duration=args.segment_duration,
                export_json=args.export_json,
                export_audio=args.export_audio,
                effects=args.effects,
                instruments=args.instruments,
                separate=args.separate,
                export_stem_files=args.export_stems,
                chord_backend=args.chord_backend,
                pitch_backend=args.pitch_backend,
                use_essentia=args.essentia,
                use_cache=args.cache,
                cache_dir=args.cache_dir
            )
        elif args.command == 'batch':
            return self.batch_process(
                args.input_dir,
                output_dir=args.output_dir,
                file_extension=args.ext,
                generate_visualizations=not args.no_vis,
                export_json=args.export_json,
                jobs=args.jobs,
                resume=args.resume
            )
        elif args.command == 'version':
            self._print_version()
            return 0
        else:
            self.parser.print_help()
            return 1
    
    def analyze_audio(self, 
                     input_file: str, 
                     output_dir: str = 'output',
                     generate_visualizations: bool = True,
                     segment_duration: float = 30.0,
                     export_json: bool = False,
                     export_audio: bool = False,
                     effects: bool = False,
                     instruments: bool = False,
                     separate: str = 'none',
                     export_stem_files: bool = False,
                     chord_backend: str = 'simple',
                     pitch_backend: str = 'yin',
                     use_essentia: bool = False,
                     use_cache: bool = False,
                     cache_dir: str = '.cache') -> int:
        """Analyze a single audio file."""
        try:
            print(f"Analyzing audio file: {input_file}")
            
            # Create output directories
            ensure_dir(output_dir)
            audio_basename = os.path.splitext(os.path.basename(input_file))[0]
            
            # Load and process audio
            audio_processor = AudioProcessor()
            audio, sample_rate = audio_processor.load_audio(input_file)
            
            # Get audio info
            audio_info = audio_processor.get_audio_info(audio, sample_rate)
            print("\nAudio Information:")
            for key, value in audio_info.items():
                print(f"  {key.replace('_', ' ').title()}: {value}")
            
            # Initialize feature extractor
            feature_extractor = FeatureExtractor()
            
            # Cache pre-check
            cache_path = None
            features = None
            if use_cache:
                ensure_dir(cache_dir)
                file_hash = get_file_hash(input_file)
                # fingerprint minimal options that affect results
                fp = json.dumps({
                    'ver': 1,
                    'effects': effects,
                    'instruments': instruments,
                    'separate': separate,
                    'chord_backend': chord_backend,
                    'pitch_backend': pitch_backend,
                    'essentia': use_essentia
                }, sort_keys=True)
                import hashlib
                opt_hash = hashlib.md5(fp.encode('utf-8')).hexdigest()[:12]
                cache_basename = f"{os.path.splitext(os.path.basename(input_file))[0]}_{file_hash[:8]}_{opt_hash}.json"
                cache_path = os.path.join(cache_dir, cache_basename)
                if os.path.isfile(cache_path):
                    try:
                        with open(cache_path, 'r') as cf:
                            features = json.load(cf)
                        print(f"Loaded analysis from cache: {cache_path}")
                    except Exception:
                        features = None

            # Extract features from the entire audio (if not cached)
            if features is None:
                print("\nExtracting features...")
                features = feature_extractor.extract_features(audio, sample_rate)
            
            # Optional: Essentia metrics (robust key/loudness) if requested
            if use_essentia:
                try:
                    ess = compute_essentia_metrics(audio, sample_rate)
                    if ess:
                        features.update(ess)
                        # If Essentia provided key/mode, include a consolidated display field
                        if 'essentia_key' in ess or 'essentia_mode' in ess:
                            features['key_display'] = f"{ess.get('essentia_key', features.get('key','?'))} {ess.get('essentia_mode', features.get('mode',''))}".strip()
                except Exception:
                    pass
            
            # Detect chords (backend selectable)
            print("Detecting chords...")
            use_chordino = (chord_backend == 'chordino')
            chords_list_dict = []
            chords_for_summary: list = []
            if use_chordino:
                chords_list_dict = detect_chords_chordino(input_file)
                if chords_list_dict:
                    features['chords'] = chords_list_dict
                    # Convert to DetectedChord for summarization
                    note_names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
                    for ch in chords_list_dict:
                        try:
                            idx = note_names.index(ch['root'])
                        except ValueError:
                            continue
                        dc = DetectedChord(
                            root=NoteName(idx),
                            quality='min' if ch.get('quality','maj') == 'min' else 'maj',
                            confidence=float(ch.get('confidence', 0.0)),
                            start_time=float(ch.get('start_time', 0.0)),
                            duration=float(ch.get('duration', 0.0))
                        )
                        chords_for_summary.append(dc)
                else:
                    print("Chordino not available or failed; falling back to simple backend.")
                    use_chordino = False
            if not use_chordino:
                chords = feature_extractor.detect_chords(audio, sample_rate)
                features['chords'] = [{
                    'root': chord.root.name,
                    'quality': chord.quality,
                    'confidence': chord.confidence,
                    'start_time': chord.start_time,
                    'duration': chord.duration
                } for chord in chords]
                chords_for_summary = chords
            features['chord_progression'] = FeatureExtractor.summarize_chord_progression(chords_for_summary)
            
            # Detect notes
            print("Detecting notes...")
            notes = feature_extractor.detect_notes(audio, sample_rate)
            features['notes'] = notes

            # Pitch tracking (F0 curve)
            print("Tracking pitch (F0)...")
            if pitch_backend == 'torchcrepe':
                f0 = track_f0_torchcrepe(audio, sample_rate)
                if f0:
                    features['pitch_track'] = f0
                else:
                    print("torchcrepe not available; falling back to YIN.")
                    pitch_backend = 'yin'
            if pitch_backend == 'yin':
                f0 = librosa.pyin(
                    y=audio, fmin=50.0, fmax=1100.0, sr=sample_rate, frame_length=2048, hop_length=256
                )
                times = librosa.times_like(f0, sr=sample_rate, hop_length=256)
                features['pitch_track'] = {
                    'times': times.tolist(),
                    'f0_hz': np.nan_to_num(f0, nan=0.0).tolist(),
                    'sample_rate': sample_rate,
                    'hop_length': 256
                }

            if effects:
                print("Running effects analysis...")
                eff = analyze_effects(audio, sample_rate)
                features['effects'] = eff

            if instruments:
                print("Running instrument recognition...")
                recog = InstrumentRecognizer()
                inst = recog.recognize(audio, sample_rate)
                features['instruments'] = inst

            stems_info = None
            if separate and separate != 'none':
                print(f"Performing source separation: {separate}")
                stems = {}
                stems_sr = sample_rate
                if separate == 'hpss':
                    stems = separate_hpss(audio)
                elif separate == 'demucs':
                    stems = separate_demucs(input_file)
                    stems_sr = stems.get('sample_rate', stems_sr)
                else:
                    try:
                        stems_count = int(separate.replace('spleeter', ''))
                    except Exception:
                        stems_count = 2
                    stems = separate_spleeter(input_file, stems=stems_count) or {}
                    stems_sr = stems.get('sample_rate', stems_sr)
                if stems:
                    features['stems'] = [k for k in stems.keys() if k != 'sample_rate']
                    if export_audio or export_stem_files:
                        stems_dir = os.path.join(output_dir, 'stems')
                        stems_info = export_stems(stems, stems_sr, stems_dir, prefix=audio_basename)
                        features['stems_paths'] = stems_info
                        print(f"Exported stems to: {stems_dir}")
                else:
                    print("Separation skipped or required package not available.")
            
            # Generate visualizations if requested
            if generate_visualizations:
                print("\nGenerating visualizations...")
                vis_dir = os.path.join(output_dir, 'visualizations')
                ensure_dir(vis_dir)
                
                visualizer = AudioVisualizer()
                visualizer.generate_analysis_report(
                    audio, sample_rate, 
                    output_dir=vis_dir,
                    prefix=audio_basename
                )
                print(f"Visualizations saved to: {vis_dir}")
            
            # Export analysis results as JSON if requested
            if export_json:
                json_path = os.path.join(output_dir, f"{audio_basename}_analysis.json")
                save_analysis_results(features, json_path)
                print(f"\nAnalysis results saved to: {json_path}")

            # Save to cache if enabled
            if use_cache and cache_path is not None:
                try:
                    with open(cache_path, 'w') as cf:
                        json.dump(features, cf, indent=2)
                    print(f"Cached analysis at: {cache_path}")
                except Exception:
                    pass
            
            # Export processed audio segments if requested
            if export_audio:
                audio_dir = os.path.join(output_dir, 'audio_segments')
                ensure_dir(audio_dir)
                
                # Split audio into segments
                segments = split_audio(audio, sample_rate, segment_duration)
                
                for i, segment in enumerate(segments):
                    segment_path = os.path.join(
                        audio_dir, 
                        f"{audio_basename}_segment_{i+1:03d}.wav"
                    )
                    audio_processor.save_audio(
                        segment, sample_rate, segment_path
                    )
                
                print(f"Exported {len(segments)} audio segments to: {audio_dir}")
            
            print("\nAnalysis complete!")
            return 0
            
        except Exception as e:
            print(f"\nError: {str(e)}", file=sys.stderr)
            return 1
    
    def batch_process(self, 
                     input_dir: str, 
                     output_dir: str = 'batch_output',
                     file_extension: str = 'wav',
                     generate_visualizations: bool = True,
                     export_json: bool = False,
                     jobs: int = 1,
                     resume: bool = False) -> int:
        """Process multiple audio files in a directory."""
        try:
            if not os.path.isdir(input_dir):
                print(f"Error: Input directory not found: {input_dir}", file=sys.stderr)
                return 1
            
            # Get list of audio files
            audio_files = [
                os.path.join(input_dir, f) for f in os.listdir(input_dir)
                if f.lower().endswith(f'.{file_extension.lower()}')
            ]
            
            if not audio_files:
                print(f"No .{file_extension} files found in {input_dir}")
                return 0
            
            print(f"Found {len(audio_files)} audio files to process.")
            
            # Resume: skip files that already have JSON output
            if resume and export_json:
                to_process = []
                for f in audio_files:
                    base = os.path.splitext(os.path.basename(f))[0]
                    out_dir = os.path.join(output_dir, base)
                    json_path = os.path.join(out_dir, f"{base}_analysis.json")
                    if os.path.isfile(json_path):
                        print(f"Skipping (resume): {f}")
                        continue
                    to_process.append(f)
                audio_files = to_process
                if not audio_files:
                    print("Nothing to process after resume filter.")
                    return 0

            # Process each file (optionally in parallel)
            results = {}
            from concurrent.futures import ThreadPoolExecutor, as_completed
            def _process_one(audio_file: str) -> tuple:
                file_basename = os.path.splitext(os.path.basename(audio_file))[0]
                file_output_dir = os.path.join(output_dir, file_basename)
                ensure_dir(file_output_dir)
                ret = self.analyze_audio(
                    audio_file,
                    output_dir=file_output_dir,
                    generate_visualizations=generate_visualizations,
                    export_json=export_json
                )
                return audio_file, ret

            total = len(audio_files)
            print(f"Processing {total} files with jobs={jobs}...")
            if jobs <= 1:
                for i, f in enumerate(audio_files, 1):
                    print(f"\n[{i}/{total}] {os.path.basename(f)}")
                    k, ret = _process_one(f)
                    results[k] = 'Success' if ret == 0 else 'Failed'
            else:
                with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
                    futs = {ex.submit(_process_one, f): f for f in audio_files}
                    done = 0
                    for fut in as_completed(futs):
                        k, ret = fut.result()
                        done += 1
                        print(f"[{done}/{total}] {os.path.basename(k)} -> {'OK' if ret == 0 else 'FAIL'}")
                        results[k] = 'Success' if ret == 0 else 'Failed'
            
            # Print summary
            print("\nBatch processing complete!")
            print("\nSummary:")
            for file, status in results.items():
                print(f"  {os.path.basename(file)}: {status}")
            
            return 0
            
        except Exception as e:
            print(f"\nError during batch processing: {str(e)}", file=sys.stderr)
            return 1
    
    def _print_version(self):
        """Print version information."""
        from audio_analyzer import __version__
        print(f"WAV Reverse Engineering Tool v{__version__}")
        print("Copyright (c) 2023 Your Name. All rights reserved.")

def main():
    """Main entry point for the CLI."""
    cli = AudioAnalyzerCLI()
    sys.exit(cli.run())

if __name__ == '__main__':
    main()
