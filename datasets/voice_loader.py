"""Voice dataset loading utilities."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class VoiceSample:
    """A single voice evaluation sample."""

    audio_path: str
    reference_text: str
    language: str = "en"
    speaker_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def load_voice_dataset(path: str | Path) -> list[VoiceSample]:
    """Load a voice evaluation dataset from a JSON manifest file.

    The JSON file should contain a list of objects, each with at minimum
    'audio_path' and 'reference_text' fields.

    Args:
        path: Path to the JSON manifest file.

    Returns:
        A list of VoiceSample objects.

    Raises:
        FileNotFoundError: If the manifest file does not exist.
    """
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"Voice dataset manifest not found: {path}")

    with open(filepath) as f:
        data = json.load(f)

    samples: list[VoiceSample] = []
    base_dir = filepath.parent

    for item in data:
        audio_path = item["audio_path"]
        # Resolve relative paths against the manifest directory
        if not Path(audio_path).is_absolute():
            audio_path = str(base_dir / audio_path)

        sample = VoiceSample(
            audio_path=audio_path,
            reference_text=item["reference_text"],
            language=item.get("language", "en"),
            speaker_id=item.get("speaker_id", ""),
            metadata=item.get("metadata", {}),
        )
        samples.append(sample)

    return samples


def load_voice_dataset_from_directory(
    directory: str | Path,
    transcript_extension: str = ".txt",
    audio_extensions: tuple[str, ...] = (".wav", ".mp3", ".flac"),
) -> list[VoiceSample]:
    """Load a voice dataset from a directory of audio files with matching transcripts.

    Expects audio files and transcript files with matching basenames.

    Args:
        directory: Path to the directory containing audio and transcript files.
        transcript_extension: Extension for transcript files.
        audio_extensions: Tuple of valid audio file extensions.

    Returns:
        A list of VoiceSample objects.
    """
    dirpath = Path(directory)
    if not dirpath.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    samples: list[VoiceSample] = []

    for audio_file in sorted(dirpath.iterdir()):
        if audio_file.suffix.lower() not in audio_extensions:
            continue

        transcript_file = audio_file.with_suffix(transcript_extension)
        if not transcript_file.exists():
            continue

        reference_text = transcript_file.read_text().strip()
        samples.append(
            VoiceSample(
                audio_path=str(audio_file),
                reference_text=reference_text,
            )
        )

    return samples
