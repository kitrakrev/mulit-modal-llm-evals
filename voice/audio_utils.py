"""Audio utility functions for loading, resampling, and format conversion."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class AudioData:
    """Container for audio data with metadata."""

    samples: np.ndarray
    sample_rate: int
    channels: int = 1
    duration_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.duration_seconds == 0.0 and self.sample_rate > 0:
            self.duration_seconds = len(self.samples) / self.sample_rate


def load_audio(
    path: str | Path,
    target_sample_rate: int = 16000,
    mono: bool = True,
) -> AudioData:
    """Load an audio file and optionally resample.

    Args:
        path: Path to the audio file.
        target_sample_rate: Target sample rate for resampling.
        mono: Whether to convert to mono.

    Returns:
        An AudioData instance with the loaded audio.

    Raises:
        FileNotFoundError: If the audio file does not exist.
    """
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    import soundfile as sf

    data, sample_rate = sf.read(str(filepath))

    # Convert to mono if requested
    if mono and data.ndim > 1:
        data = np.mean(data, axis=1)

    # Resample if needed
    if sample_rate != target_sample_rate:
        data = resample_audio(data, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate

    return AudioData(
        samples=data.astype(np.float32),
        sample_rate=sample_rate,
        channels=1 if mono else (data.shape[1] if data.ndim > 1 else 1),
    )


def resample_audio(
    data: np.ndarray,
    original_rate: int,
    target_rate: int,
) -> np.ndarray:
    """Resample audio data to a target sample rate.

    Uses linear interpolation for resampling. For higher quality, install librosa.

    Args:
        data: The audio samples as a numpy array.
        original_rate: The original sample rate.
        target_rate: The target sample rate.

    Returns:
        The resampled audio data.
    """
    if original_rate == target_rate:
        return data

    try:
        import librosa

        return librosa.resample(data, orig_sr=original_rate, target_sr=target_rate)
    except ImportError:
        # Fallback to linear interpolation
        duration = len(data) / original_rate
        target_length = int(duration * target_rate)
        indices = np.linspace(0, len(data) - 1, target_length)
        return np.interp(indices, np.arange(len(data)), data).astype(np.float32)


def save_audio(
    path: str | Path,
    data: np.ndarray,
    sample_rate: int,
) -> None:
    """Save audio data to a file.

    Args:
        path: Output file path. Format is inferred from extension.
        data: The audio samples as a numpy array.
        sample_rate: The sample rate.
    """
    import soundfile as sf

    sf.write(str(path), data, sample_rate)


def get_audio_duration(path: str | Path) -> float:
    """Get the duration of an audio file in seconds.

    Args:
        path: Path to the audio file.

    Returns:
        Duration in seconds.
    """
    import soundfile as sf

    info = sf.info(str(path))
    return info.duration


def convert_format(
    input_path: str | Path,
    output_path: str | Path,
    target_sample_rate: int | None = None,
) -> None:
    """Convert an audio file to a different format.

    Args:
        input_path: Path to the input audio file.
        output_path: Path for the output file. Format is inferred from extension.
        target_sample_rate: Optional target sample rate for resampling.
    """
    audio = load_audio(str(input_path), mono=False)

    data = audio.samples
    sample_rate = audio.sample_rate

    if target_sample_rate and target_sample_rate != sample_rate:
        data = resample_audio(data, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate

    save_audio(output_path, data, sample_rate)
