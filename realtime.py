"""
Real-time Audio Classification Pipeline for AI Voice Detection

This module provides streaming inference for detecting synthetic/deepfake audio
in real-time using microphone input or audio streams.

Features:
- Microphone capture with configurable sample rate
- Sliding window inference with overlap
- Smoothed predictions using exponential moving average
- Configurable confidence thresholding
- Low-latency inference with TorchScript support
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from collections import deque

import numpy as np
import torch
import torch.nn as nn

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    print("Warning: sounddevice not available. Install with: pip install sounddevice")

from data_utils import extract_feature, FEATURE_TYPES


class StreamingDetector:
    """
    Real-time streaming audio classifier for deepfake detection.
    
    Uses a sliding window approach with overlapping segments to provide
    continuous predictions on incoming audio streams.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        feature_type: int = 1,
        sample_rate: int = 16000,
        window_duration: float = 2.0,
        hop_duration: float = 0.5,
        smoothing_alpha: float = 0.3,
        threshold: float = 0.5,
    ):
        """
        Initialize the streaming detector.
        
        Args:
            model: Trained PyTorch model for classification
            device: Device to run inference on (cpu/cuda)
            feature_type: Feature type for extraction (0=raw, 1=mel, etc.)
            sample_rate: Audio sample rate in Hz
            window_duration: Duration of each classification window in seconds
            hop_duration: Duration between consecutive windows in seconds
            smoothing_alpha: EMA smoothing factor (0-1, higher = less smoothing)
            threshold: Classification threshold (predictions >= threshold are "real")
        """
        self.model = model
        self.device = device
        self.feature_type = feature_type
        self.sample_rate = sample_rate
        self.window_duration = window_duration
        self.hop_duration = hop_duration
        self.smoothing_alpha = smoothing_alpha
        self.threshold = threshold
        
        # Calculate sample counts
        self.window_samples = int(window_duration * sample_rate)
        self.hop_samples = int(hop_duration * sample_rate)
        
        # Audio buffer for accumulating samples
        self.audio_buffer = np.zeros(0, dtype=np.float32)
        
        # Smoothed prediction score
        self.smoothed_score = 0.5
        
        # Prediction history for visualization
        self.prediction_history = deque(maxlen=100)
        
        # Set model to eval mode
        self.model.eval()
        
        # Try to compile model for faster inference (PyTorch 2.0+)
        self._optimize_model()
    
    def _optimize_model(self):
        """Apply optimizations for faster inference."""
        try:
            # Try torch.compile for PyTorch 2.0+
            if hasattr(torch, 'compile'):
                self.model = torch.compile(self.model, mode='reduce-overhead')
                print("Model compiled with torch.compile for faster inference")
        except Exception as e:
            print(f"torch.compile not available: {e}")
    
    def reset(self):
        """Reset internal buffers and state."""
        self.audio_buffer = np.zeros(0, dtype=np.float32)
        self.smoothed_score = 0.5
        self.prediction_history.clear()
    
    def _extract_features(self, audio: np.ndarray) -> torch.Tensor:
        """Extract features from audio segment."""
        # Extract features
        features = extract_feature(audio, feature_type=self.feature_type, sr=self.sample_rate)
        
        # Convert to tensor and add batch dimension
        if features.ndim == 1:
            # Raw waveform
            x = torch.from_numpy(features).float().unsqueeze(0)
        else:
            # Spectrogram features (freq x time)
            x = torch.from_numpy(features).float().unsqueeze(0)
        
        return x.to(self.device)
    
    @torch.no_grad()
    def _predict(self, x: torch.Tensor) -> Tuple[float, int]:
        """
        Run inference on a single batch.
        
        Returns:
            (score, prediction): Score in [0, 1] and binary prediction
        """
        # Forward pass
        embeddings, logits = self.model(x)
        
        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=-1)
        
        # Score for "real" class (assuming class 1 = real, class 0 = fake)
        score = probs[0, 1].item()
        
        return score, int(score >= self.threshold)
    
    def process_chunk(self, audio_chunk: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Process an incoming audio chunk.
        
        Args:
            audio_chunk: Audio samples as numpy array
            
        Returns:
            Dictionary with prediction info if a window was processed, None otherwise
        """
        # Append to buffer
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk.flatten()])
        
        # Check if we have enough samples for a window
        if len(self.audio_buffer) < self.window_samples:
            return None
        
        # Extract the window
        window = self.audio_buffer[:self.window_samples]
        
        # Remove processed samples (hop forward)
        self.audio_buffer = self.audio_buffer[self.hop_samples:]
        
        # Extract features and predict
        x = self._extract_features(window)
        raw_score, prediction = self._predict(x)
        
        # Apply exponential moving average smoothing
        self.smoothed_score = (
            self.smoothing_alpha * raw_score + 
            (1 - self.smoothing_alpha) * self.smoothed_score
        )
        
        # Determine smoothed prediction
        smoothed_prediction = int(self.smoothed_score >= self.threshold)
        
        # Build result
        result = {
            'raw_score': raw_score,
            'smoothed_score': self.smoothed_score,
            'raw_prediction': prediction,
            'smoothed_prediction': smoothed_prediction,
            'label': 'REAL' if smoothed_prediction == 1 else 'FAKE',
            'confidence': abs(self.smoothed_score - 0.5) * 2,  # 0-1 confidence
            'timestamp': time.time(),
        }
        
        # Add to history
        self.prediction_history.append(result)
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics from prediction history."""
        if not self.prediction_history:
            return {'num_predictions': 0}
        
        scores = [p['smoothed_score'] for p in self.prediction_history]
        predictions = [p['smoothed_prediction'] for p in self.prediction_history]
        
        return {
            'num_predictions': len(self.prediction_history),
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'real_ratio': np.mean(predictions),
            'fake_ratio': 1 - np.mean(predictions),
        }


class MicrophoneStream:
    """
    Context manager for microphone audio streaming.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration: float = 0.1,
        device: Optional[int] = None,
    ):
        """
        Initialize microphone stream.
        
        Args:
            sample_rate: Sample rate in Hz
            chunk_duration: Duration of each audio chunk in seconds
            device: Audio device index (None = default)
        """
        if not SOUNDDEVICE_AVAILABLE:
            raise RuntimeError("sounddevice is required for microphone input")
        
        self.sample_rate = sample_rate
        self.chunk_size = int(chunk_duration * sample_rate)
        self.device = device
        self.stream = None
        self.audio_queue = deque()
    
    def _callback(self, indata, frames, time_info, status):
        """Callback for audio stream."""
        if status:
            print(f"Audio stream status: {status}", file=sys.stderr)
        self.audio_queue.append(indata.copy())
    
    def __enter__(self):
        """Start the audio stream."""
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            blocksize=self.chunk_size,
            device=self.device,
            callback=self._callback,
        )
        self.stream.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the audio stream."""
        if self.stream:
            self.stream.stop()
            self.stream.close()
    
    def read(self) -> Optional[np.ndarray]:
        """Read available audio data."""
        if self.audio_queue:
            return self.audio_queue.popleft()
        return None


def load_model_for_realtime(
    model_path: str,
    config_path: str,
    device: torch.device,
) -> nn.Module:
    """
    Load a trained model for real-time inference.
    
    Args:
        model_path: Path to model weights (.pth file)
        config_path: Path to model config (.conf file)
        device: Device to load model on
        
    Returns:
        Loaded model ready for inference
    """
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Get model architecture
    model_config = config.get('model_config', config)
    architecture = model_config.get('architecture', 'EfficientNetB2')
    
    # Import the appropriate model
    if architecture == 'EfficientNetB2':
        from models.EfficientNetB2 import Model
    elif architecture == 'LCNN':
        from models.LCNN import Model
    elif architecture == 'AudioViT':
        from models.AudioViT import Model
    elif architecture == 'Conformer':
        from models.Conformer import Model
    elif architecture == 'SEResNet':
        from models.SEResNet import Model
    elif architecture == 'SimpleCNN':
        from models.SimpleCNN import Model
    elif architecture == 'RawNet3':
        from models.RawNet3 import Model
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    # Initialize model
    model = Model(model_config).to(device)
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def run_realtime_detection(
    model_path: str,
    config_path: str,
    feature_type: int = 1,
    threshold: float = 0.5,
    duration: float = 60.0,
    device_str: str = 'cpu',
):
    """
    Run real-time detection from microphone input.
    
    Args:
        model_path: Path to trained model weights
        config_path: Path to model config
        feature_type: Feature type for extraction
        threshold: Classification threshold
        duration: Duration to run in seconds (0 = indefinite)
        device_str: Device string ('cpu' or 'cuda')
    """
    device = torch.device(device_str)
    
    print(f"Loading model from {model_path}...")
    model = load_model_for_realtime(model_path, config_path, device)
    
    print("Initializing streaming detector...")
    detector = StreamingDetector(
        model=model,
        device=device,
        feature_type=feature_type,
        threshold=threshold,
    )
    
    print("\n" + "="*60)
    print("REAL-TIME DEEPFAKE AUDIO DETECTION")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Feature type: {FEATURE_TYPES.get(feature_type, feature_type)}")
    print(f"Threshold: {threshold}")
    print(f"Duration: {duration if duration > 0 else 'Indefinite'}s")
    print("="*60)
    print("\nListening... Press Ctrl+C to stop.\n")
    
    start_time = time.time()
    
    try:
        with MicrophoneStream() as mic:
            while True:
                # Check duration
                if duration > 0 and (time.time() - start_time) >= duration:
                    print("\nDuration reached. Stopping...")
                    break
                
                # Read audio chunk
                audio = mic.read()
                if audio is None:
                    time.sleep(0.01)
                    continue
                
                # Process chunk
                result = detector.process_chunk(audio)
                
                if result:
                    # Display result
                    label = result['label']
                    score = result['smoothed_score']
                    conf = result['confidence']
                    
                    # Color the output based on prediction
                    if label == 'REAL':
                        color = '\033[92m'  # Green
                    else:
                        color = '\033[91m'  # Red
                    reset = '\033[0m'
                    
                    # Create progress bar
                    bar_len = 30
                    filled = int(bar_len * score)
                    bar = '█' * filled + '░' * (bar_len - filled)
                    
                    print(f"\r{color}[{label:4s}]{reset} Score: {score:.3f} [{bar}] Conf: {conf:.1%}", end='', flush=True)
    
    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    
    finally:
        # Print statistics
        stats = detector.get_statistics()
        print("\n" + "="*60)
        print("SESSION STATISTICS")
        print("="*60)
        print(f"Total predictions: {stats.get('num_predictions', 0)}")
        if stats.get('num_predictions', 0) > 0:
            print(f"Mean score: {stats['mean_score']:.3f} ± {stats['std_score']:.3f}")
            print(f"Real ratio: {stats['real_ratio']:.1%}")
            print(f"Fake ratio: {stats['fake_ratio']:.1%}")
        print("="*60)


def process_audio_file(
    audio_path: str,
    model_path: str,
    config_path: str,
    feature_type: int = 1,
    threshold: float = 0.5,
    device_str: str = 'cpu',
) -> Dict[str, Any]:
    """
    Process an audio file for deepfake detection.
    
    Args:
        audio_path: Path to audio file
        model_path: Path to model weights
        config_path: Path to model config
        feature_type: Feature type
        threshold: Classification threshold
        device_str: Device string
        
    Returns:
        Detection results
    """
    import soundfile as sf
    
    device = torch.device(device_str)
    model = load_model_for_realtime(model_path, config_path, device)
    
    detector = StreamingDetector(
        model=model,
        device=device,
        feature_type=feature_type,
        threshold=threshold,
        window_duration=2.0,
        hop_duration=0.5,
    )
    
    # Load audio
    audio, sr = sf.read(audio_path)
    if sr != detector.sample_rate:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=detector.sample_rate)
    
    # Process in chunks
    chunk_size = int(0.1 * detector.sample_rate)
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i+chunk_size]
        detector.process_chunk(chunk)
    
    stats = detector.get_statistics()
    final_prediction = 'REAL' if stats.get('mean_score', 0.5) >= threshold else 'FAKE'
    
    return {
        'file': audio_path,
        'prediction': final_prediction,
        'mean_score': stats.get('mean_score', 0.5),
        'std_score': stats.get('std_score', 0.0),
        'num_windows': stats.get('num_predictions', 0),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Real-time Deepfake Audio Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Real-time from microphone
  python realtime.py --model_path exp_result/run1/weights/best.pth --config config/EfficientNetB2.conf
  
  # Process a file
  python realtime.py --file audio.wav --model_path model.pth --config config.conf
  
  # Run for specific duration
  python realtime.py --model_path model.pth --config config.conf --duration 30
        """
    )
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model weights (.pth)')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to model config (.conf)')
    parser.add_argument('--file', type=str, default=None,
                       help='Audio file to process (if not provided, uses microphone)')
    parser.add_argument('--feature_type', type=int, default=1,
                       help='Feature type (0=raw, 1=mel, 2=lfcc, 3=mfcc, 4=cqt)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Classification threshold')
    parser.add_argument('--duration', type=float, default=0,
                       help='Duration in seconds (0 = indefinite)')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device for inference')
    
    args = parser.parse_args()
    
    if args.file:
        # Process file
        result = process_audio_file(
            audio_path=args.file,
            model_path=args.model_path,
            config_path=args.config,
            feature_type=args.feature_type,
            threshold=args.threshold,
            device_str=args.device,
        )
        print(f"\nFile: {result['file']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Mean Score: {result['mean_score']:.3f} ± {result['std_score']:.3f}")
        print(f"Windows analyzed: {result['num_windows']}")
    else:
        # Real-time from microphone
        run_realtime_detection(
            model_path=args.model_path,
            config_path=args.config,
            feature_type=args.feature_type,
            threshold=args.threshold,
            duration=args.duration,
            device_str=args.device,
        )


if __name__ == '__main__':
    main()