"""
YouTube audio ingestion module.

Downloads audio from YouTube URLs using yt-dlp and prepares them for analysis.
"""

import os
import tempfile
import hashlib
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False


class YouTubeIngestion:
    """
    Downloads and prepares audio from YouTube URLs for analysis.
    """

    def __init__(self, output_dir: Optional[str] = None, keep_files: bool = False):
        """
        Initialize the YouTube ingestion module.

        Args:
            output_dir: Directory to store downloaded audio files.
                        If None, uses a temp directory.
            keep_files: If True, keeps downloaded files after processing.
                        If False, files are deleted after use.
        """
        if not YT_DLP_AVAILABLE:
            raise ImportError(
                "yt-dlp is required for YouTube ingestion. "
                "Install with: pip install yt-dlp"
            )
        
        self.output_dir = output_dir or tempfile.gettempdir()
        self.keep_files = keep_files
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def _get_ydl_opts(self, output_path: str) -> Dict[str, Any]:
        """Get yt-dlp options for audio extraction."""
        return {
            'format': 'bestaudio/best',
            'outtmpl': output_path,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
        }

    def get_video_info(self, url: str) -> Dict[str, Any]:
        """
        Get metadata about a YouTube video without downloading.

        Args:
            url: YouTube video URL

        Returns:
            Dictionary containing video metadata (title, duration, etc.)
        """
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            
        return {
            'id': info.get('id', ''),
            'title': info.get('title', 'Unknown'),
            'duration': info.get('duration', 0),
            'uploader': info.get('uploader', 'Unknown'),
            'view_count': info.get('view_count', 0),
            'upload_date': info.get('upload_date', ''),
            'description': info.get('description', '')[:500],  # Truncate
            'thumbnail': info.get('thumbnail', ''),
            'url': url,
        }

    def download_audio(self, url: str, filename: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Download audio from a YouTube URL.

        Args:
            url: YouTube video URL
            filename: Optional custom filename (without extension).
                      If None, uses video ID.

        Returns:
            Tuple of (path_to_wav_file, video_metadata)
        """
        # Get video info first
        info = self.get_video_info(url)
        video_id = info['id']
        
        # Generate filename
        if filename is None:
            # Use video ID + hash of URL for uniqueness
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            filename = f"yt_{video_id}_{url_hash}"
        
        # Clean filename
        filename = "".join(c for c in filename if c.isalnum() or c in "._-")
        
        output_template = os.path.join(self.output_dir, filename)
        output_wav = output_template + ".wav"
        
        # Check if already downloaded
        if os.path.exists(output_wav):
            return output_wav, info
        
        # Download with yt-dlp
        ydl_opts = self._get_ydl_opts(output_template)
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        # yt-dlp may add extension, find the actual file
        if not os.path.exists(output_wav):
            # Try common patterns
            for ext in ['.wav', '.mp3', '.m4a', '.webm']:
                candidate = output_template + ext
                if os.path.exists(candidate):
                    # Convert to wav if needed
                    if ext != '.wav':
                        self._convert_to_wav(candidate, output_wav)
                        os.unlink(candidate)
                    break
        
        if not os.path.exists(output_wav):
            raise FileNotFoundError(
                f"Failed to download or convert audio from {url}"
            )
        
        return output_wav, info

    def _convert_to_wav(self, input_path: str, output_path: str) -> None:
        """Convert an audio file to WAV format using pydub."""
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(input_path)
            audio.export(output_path, format="wav")
        except Exception as e:
            raise RuntimeError(f"Failed to convert {input_path} to WAV: {e}")

    def cleanup(self, file_path: str) -> None:
        """Remove a downloaded file if keep_files is False."""
        if not self.keep_files and os.path.exists(file_path):
            try:
                os.unlink(file_path)
            except Exception:
                pass

    @staticmethod
    def is_youtube_url(url: str) -> bool:
        """Check if a URL is a valid YouTube URL."""
        youtube_patterns = [
            'youtube.com/watch',
            'youtu.be/',
            'youtube.com/shorts/',
            'youtube.com/v/',
            'youtube.com/embed/',
        ]
        return any(pattern in url.lower() for pattern in youtube_patterns)


def download_youtube_audio(url: str, output_dir: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    """
    Convenience function to download YouTube audio.

    Args:
        url: YouTube video URL
        output_dir: Optional output directory

    Returns:
        Tuple of (path_to_wav_file, video_metadata)
    """
    ingestion = YouTubeIngestion(output_dir=output_dir, keep_files=True)
    return ingestion.download_audio(url)
