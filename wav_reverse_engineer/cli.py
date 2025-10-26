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

# Use non-interactive backend for CLI
matplotlib.use('Agg')

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.absolute()))

from audio_analyzer.audio_processor import AudioProcessor
from audio_analyzer.feature_extractor import FeatureExtractor, DetectedChord, NoteName
from audio_analyzer.visualizer import AudioVisualizer
from audio_analyzer.utils import save_analysis_results, ensure_dir
from audio_analyzer.effects_analyzer import analyze_effects
from audio_analyzer.source_separation import separate_hpss, separate_spleeter, export_stems
from audio_analyzer.instrument_recognizer import InstrumentRecognizer
from audio_analyzer.backends.chordino import detect_chords_chordino

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
                                  choices=['none', 'hpss', 'spleeter2', 'spleeter4', 'spleeter5'],
                                  help='Perform source separation method')
        analyze_parser.add_argument('--export-stems', action='store_true',
                                  help='Export separated stems as audio files')
        analyze_parser.add_argument('--chord-backend', type=str, default='simple',
                                  choices=['simple', 'chordino'],
                                  help='Chord detection backend (default: simple)')
        
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
                chord_backend=args.chord_backend
            )
        elif args.command == 'batch':
            return self.batch_process(
                args.input_dir,
                output_dir=args.output_dir,
                file_extension=args.ext,
                generate_visualizations=not args.no_vis,
                export_json=args.export_json
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
                     chord_backend: str = 'simple') -> int:
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
            
            # Extract features from the entire audio
            print("\nExtracting features...")
            features = feature_extractor.extract_features(audio, sample_rate)
            
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
            
            # Export processed audio segments if requested
            if export_audio:
                audio_dir = os.path.join(output_dir, 'audio_segments')
                ensure_dir(audio_dir)
                
                # Split audio into segments
                segments = audio_processor.split_audio(audio, sample_rate, segment_duration)
                
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
                     export_json: bool = False) -> int:
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
            
            # Process each file
            results = {}
            for i, audio_file in enumerate(audio_files, 1):
                print(f"\nProcessing file {i}/{len(audio_files)}: {os.path.basename(audio_file)}")
                
                # Create a subdirectory for this file's output
                file_basename = os.path.splitext(os.path.basename(audio_file))[0]
                file_output_dir = os.path.join(output_dir, file_basename)
                ensure_dir(file_output_dir)
                
                # Analyze the file
                result = self.analyze_audio(
                    audio_file,
                    output_dir=file_output_dir,
                    generate_visualizations=generate_visualizations,
                    export_json=export_json
                )
                
                if result != 0:
                    print(f"Warning: Failed to process {audio_file}", file=sys.stderr)
                
                results[audio_file] = 'Success' if result == 0 else 'Failed'
            
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
