"""
Transcription Processor

Uses OpenAI Whisper and other open-source speech recognition models
to transcribe audio and video content.
"""

import asyncio
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import tempfile
import shutil

import whisper
from pydub import AudioSegment
from pydub.silence import split_on_silence
import torch

from ..utils.logger import get_logger
from ..utils.config import TranscriptionConfig

logger = get_logger(__name__)


class TranscriptionProcessor:
    """Audio and video transcription using Whisper and other ASR models."""
    
    def __init__(self, config: Optional[TranscriptionConfig] = None):
        self.config = config or TranscriptionConfig()
        
        # Load Whisper model
        self.model = None
        self.model_name = self.config.model_size or 'base'
        
        # Supported audio formats
        self.supported_formats = {
            '.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma', '.m4a', '.mp4', '.avi', '.mov', '.mkv', '.webm'
        }
        
    async def __aenter__(self):
        """Async context manager entry - load Whisper model."""
        await self._load_model()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup if needed."""
        pass
        
    async def _load_model(self):
        """Load the Whisper model."""
        try:
            logger.info(f"Loading Whisper model: {self.model_name}")
            
            # Check if CUDA is available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")
            
            # Load model
            self.model = whisper.load_model(self.model_name, device=device)
            
            logger.info(f"Whisper model '{self.model_name}' loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
            
    async def transcribe_audio(self, 
                             audio_path: str,
                             language: Optional[str] = None,
                             task: str = "transcribe",
                             verbose: bool = False) -> Dict[str, Any]:
        """
        Transcribe audio file to text.
        
        Args:
            audio_path: Path to audio file
            language: Language code (auto-detect if None)
            task: Task type ('transcribe' or 'translate')
            verbose: Whether to include detailed output
            
        Returns:
            Dictionary with transcription results
        """
        start_time = datetime.now()
        
        try:
            if not self.model:
                raise RuntimeError("Whisper model not loaded")
                
            # Validate audio file
            if not Path(audio_path).exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
                
            # Load and preprocess audio
            audio = await self._load_audio(audio_path)
            
            # Transcribe using Whisper
            options = {
                "language": language,
                "task": task,
                "verbose": verbose
            }
            
            # Run transcription
            result = self.model.transcribe(audio, **options)
            
            # Process and format results
            transcription_result = {
                'text': result['text'].strip(),
                'language': result.get('language', 'unknown'),
                'task': task,
                'segments': [],
                'words': [],
                'audio_path': audio_path,
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'processed_at': start_time.isoformat()
            }
            
            # Process segments with timestamps
            if 'segments' in result:
                transcription_result['segments'] = [
                    {
                        'start': segment['start'],
                        'end': segment['end'],
                        'text': segment['text'].strip(),
                        'confidence': segment.get('avg_logprob', 0)
                    }
                    for segment in result['segments']
                ]
                
            # Process word-level timestamps if available
            if 'chunks' in result:  # Whisper doesn't provide word-level timestamps by default
                transcription_result['words'] = []
                
            # Calculate additional metrics
            transcription_result.update({
                'word_count': len(transcription_result['text'].split()),
                'character_count': len(transcription_result['text']),
                'duration_seconds': self._get_audio_duration(audio_path),
                'words_per_minute': self._calculate_wpm(transcription_result['text'], transcription_result['duration_seconds'])
            })
            
            return transcription_result
            
        except Exception as e:
            logger.error(f"Transcription failed for {audio_path}: {e}")
            return {
                'text': '',
                'language': 'unknown',
                'error': str(e),
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'processed_at': start_time.isoformat()
            }
            
    async def transcribe_batch(self, 
                             audio_paths: List[str],
                             language: Optional[str] = None,
                             task: str = "transcribe") -> List[Dict[str, Any]]:
        """
        Transcribe multiple audio files concurrently.
        
        Args:
            audio_paths: List of audio file paths
            language: Language code
            task: Task type
            
        Returns:
            List of transcription results
        """
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        async def transcribe_single_file(audio_path: str):
            async with semaphore:
                return await self.transcribe_audio(audio_path, language, task)
                
        # Process all files concurrently
        tasks = [transcribe_single_file(path) for path in audio_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log them
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to transcribe {audio_paths[i]}: {result}")
            else:
                valid_results.append(result)
                
        return valid_results
        
    async def transcribe_video(self, 
                             video_path: str,
                             extract_audio: bool = True,
                             language: Optional[str] = None,
                             task: str = "transcribe") -> Dict[str, Any]:
        """
        Transcribe video file by extracting audio first.
        
        Args:
            video_path: Path to video file
            extract_audio: Whether to extract audio from video
            language: Language code
            task: Task type
            
        Returns:
            Dictionary with transcription results
        """
        start_time = datetime.now()
        
        try:
            if not extract_audio:
                # Try to transcribe video directly (limited support)
                return await self.transcribe_audio(video_path, language, task)
                
            # Extract audio from video
            audio_path = await self._extract_audio_from_video(video_path)
            
            try:
                # Transcribe the extracted audio
                result = await self.transcribe_audio(str(audio_path), language, task)
                
                # Add video-specific information
                result.update({
                    'video_path': video_path,
                    'extracted_audio_path': str(audio_path),
                    'video_duration': self._get_video_duration(video_path),
                    'video_metadata': await self._get_video_metadata(video_path)
                })
                
                return result
                
            finally:
                # Clean up extracted audio file
                if audio_path.exists():
                    audio_path.unlink()
                    
        except Exception as e:
            logger.error(f"Video transcription failed for {video_path}: {e}")
            return {
                'text': '',
                'language': 'unknown',
                'error': str(e),
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'processed_at': start_time.isoformat()
            }
            
    async def transcribe_with_speaker_diarization(self, 
                                                audio_path: str,
                                                language: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribe audio with speaker diarization.
        
        Args:
            audio_path: Path to audio file
            language: Language code
            
        Returns:
            Dictionary with transcription and speaker information
        """
        try:
            # Basic transcription first
            result = await self.transcribe_audio(audio_path, language)
            
            # For now, return basic result without speaker diarization
            # Speaker diarization would require additional libraries like pyannote.audio
            result['speaker_diarization'] = {
                'enabled': False,
                'message': 'Speaker diarization not implemented in this version'
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Transcription with speaker diarization failed: {e}")
            return await self.transcribe_audio(audio_path, language)  # Fallback to basic transcription
            
    async def translate_audio(self, 
                            audio_path: str,
                            target_language: str = "en",
                            source_language: Optional[str] = None) -> Dict[str, Any]:
        """
        Translate audio content to target language.
        
        Args:
            audio_path: Path to audio file
            target_language: Target language code
            source_language: Source language code
            
        Returns:
            Dictionary with translation results
        """
        try:
            # Use Whisper's translation capability
            result = await self.transcribe_audio(
                audio_path, 
                language=source_language, 
                task="translate"
            )
            
            # Add translation-specific metadata
            result.update({
                'translation': {
                    'source_language': result.get('language', 'unknown'),
                    'target_language': target_language,
                    'translated_text': result['text']
                }
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Audio translation failed for {audio_path}: {e}")
            return {
                'text': '',
                'translation': {
                    'source_language': 'unknown',
                    'target_language': target_language,
                    'error': str(e)
                },
                'error': str(e)
            }
            
    async def _load_audio(self, audio_path: str):
        """Load and preprocess audio file."""
        try:
            # Convert to WAV format for Whisper
            audio = whisper.load_audio(audio_path)
            return audio
            
        except Exception as e:
            logger.error(f"Failed to load audio {audio_path}: {e}")
            raise
            
    async def _extract_audio_from_video(self, video_path: Path) -> Path:
        """Extract audio from video file using ffmpeg."""
        try:
            # Create temporary audio file
            audio_path = video_path.with_suffix('.temp.wav')
            
            # Extract audio using ffmpeg
            cmd = [
                'ffmpeg', '-i', str(video_path), 
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # PCM audio codec
                '-ar', '16000',  # 16kHz sample rate (Whisper optimized)
                '-ac', '1',  # Mono
                '-y',  # Overwrite output file
                str(audio_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(f"ffmpeg failed: {result.stderr}")
                
            return audio_path
            
        except Exception as e:
            logger.error(f"Audio extraction failed: {e}")
            raise
            
    def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration in seconds."""
        try:
            audio = AudioSegment.from_file(audio_path)
            return len(audio) / 1000.0  # Convert milliseconds to seconds
        except Exception:
            return 0.0
            
    def _get_video_duration(self, video_path: str) -> float:
        """Get video duration in seconds using ffprobe."""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                return float(result.stdout.strip())
                
        except Exception:
            pass
            
        return 0.0
        
    async def _get_video_metadata(self, video_path: str) -> Dict[str, Any]:
        """Get video metadata using ffprobe."""
        try:
            import json
            
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                metadata = json.loads(result.stdout)
                
                # Extract relevant video information
                video_stream = None
                audio_stream = None
                
                for stream in metadata.get('streams', []):
                    if stream.get('codec_type') == 'video':
                        video_stream = stream
                    elif stream.get('codec_type') == 'audio':
                        audio_stream = stream
                        
                return {
                    'duration': float(metadata.get('format', {}).get('duration', 0)),
                    'video': {
                        'codec': video_stream.get('codec_name', '') if video_stream else '',
                        'width': video_stream.get('width', 0) if video_stream else 0,
                        'height': video_stream.get('height', 0) if video_stream else 0,
                        'frame_rate': video_stream.get('r_frame_rate', '') if video_stream else ''
                    } if video_stream else None,
                    'audio': {
                        'codec': audio_stream.get('codec_name', '') if audio_stream else '',
                        'sample_rate': audio_stream.get('sample_rate', 0) if audio_stream else 0,
                        'channels': audio_stream.get('channels', 0) if audio_stream else 0
                    } if audio_stream else None
                }
                
        except Exception as e:
            logger.error(f"Failed to get video metadata: {e}")
            
        return {}
        
    def _calculate_wpm(self, text: str, duration_seconds: float) -> float:
        """Calculate words per minute."""
        if duration_seconds <= 0:
            return 0.0
        
        word_count = len(text.split())
        duration_minutes = duration_seconds / 60.0
        return word_count / duration_minutes if duration_minutes > 0 else 0.0
        
    async def detect_language(self, audio_path: str) -> Dict[str, Any]:
        """Detect the language of audio content."""
        try:
            # Load audio
            audio = await self._load_audio(audio_path)
            
            # Use Whisper to detect language
            # This is a simplified approach - Whisper's language detection
            # happens during transcription, but we can estimate it
            
            # For now, we'll do a basic transcription and use the detected language
            result = await self.transcribe_audio(audio_path)
            
            return {
                'detected_language': result.get('language', 'unknown'),
                'confidence': 1.0,  # Whisper doesn't provide language confidence directly
                'method': 'transcription_based'
            }
            
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return {
                'detected_language': 'unknown',
                'confidence': 0.0,
                'error': str(e)
            }
            
    async def preprocess_audio(self, 
                             input_path: str,
                             output_path: Optional[str] = None,
                             target_sample_rate: int = 16000,
                             normalize: bool = True) -> str:
        """
        Preprocess audio for better transcription results.
        
        Args:
            input_path: Input audio file path
            output_path: Output file path (auto-generated if None)
            target_sample_rate: Target sample rate
            normalize: Whether to normalize audio volume
            
        Returns:
            Path to preprocessed audio file
        """
        try:
            # Load audio
            audio = AudioSegment.from_file(input_path)
            
            # Convert to mono if not already
            if audio.channels > 1:
                audio = audio.set_channels(1)
                
            # Resample to target sample rate
            if audio.frame_rate != target_sample_rate:
                audio = audio.set_frame_rate(target_sample_rate)
                
            # Normalize audio if requested
            if normalize:
                audio = audio.normalize()
                
            # Generate output path if not provided
            if output_path is None:
                input_path_obj = Path(input_path)
                output_path = str(input_path_obj.with_suffix('.preprocessed.wav'))
                
            # Export preprocessed audio
            audio.export(output_path, format="wav")
            
            logger.info(f"Audio preprocessed: {input_path} -> {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {e}")
            raise
            
    async def split_long_audio(self, 
                             audio_path: str,
                             max_duration_seconds: float = 300,  # 5 minutes
                             silence_threshold: int = -40,
                             silence_duration: float = 1.0) -> List[str]:
        """
        Split long audio files into smaller segments for better transcription.
        
        Args:
            audio_path: Path to audio file
            max_duration_seconds: Maximum duration per segment
            silence_threshold: Silence threshold in dB
            silence_duration: Minimum silence duration in seconds
            
        Returns:
            List of output file paths
        """
        try:
            # Load audio
            audio = AudioSegment.from_file(audio_path)
            
            # Split on silence
            chunks = split_on_silence(
                audio,
                min_silence_len=int(silence_duration * 1000),  # Convert to milliseconds
                silence_thresh=silence_threshold,
                keep_silence=500  # Keep 500ms of silence at the beginning and end
            )
            
            if not chunks:
                # If no silence-based split, split by duration
                chunk_length_ms = int(max_duration_seconds * 1000)
                chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
                
            # Export chunks
            input_path_obj = Path(audio_path)
            chunk_paths = []
            
            for i, chunk in enumerate(chunks):
                chunk_path = str(input_path_obj.with_name(f"{input_path_obj.stem}_chunk_{i:03d}.wav"))
                chunk.export(chunk_path, format="wav")
                chunk_paths.append(chunk_path)
                
            logger.info(f"Audio split into {len(chunk_paths)} segments")
            return chunk_paths
            
        except Exception as e:
            logger.error(f"Audio splitting failed: {e}")
            return [audio_path]  # Return original file if splitting fails
            
    async def validate_whisper(self) -> Dict[str, Any]:
        """Validate Whisper installation and model."""
        try:
            # Check PyTorch installation
            torch_info = {
                'version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'cuda_version': torch.version.cuda if torch.cuda.is_available() else None
            }
            
            # Check if model is loaded
            model_loaded = self.model is not None
            model_info = {}
            
            if model_loaded:
                model_info = {
                    'name': self.model_name,
                    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
                }
                
            return {
                'whisper_available': True,
                'model_loaded': model_loaded,
                'torch_info': torch_info,
                'model_info': model_info
            }
            
        except Exception as e:
            logger.error(f"Whisper validation failed: {e}")
            return {
                'whisper_available': False,
                'error': str(e)
            }