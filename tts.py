import os
import subprocess
import sys
import re
import wave
from pathlib import Path
from tempfile import TemporaryDirectory
from concurrent.futures import ThreadPoolExecutor

# --- Configuration ---
voice_path = "voices/en_US-danny-low.onnx"
output_dir = "dialogue_audio"
os.makedirs(output_dir, exist_ok=True)

VOICES = {
    'Yoon_Jae': 'voices/en_US-john-medium.onnx',
    'Seo_Yeon': 'voices/en_US-amy-medium.onnx',
    'Dr_Noh': 'voices/en_US-bryce-medium.onnx',
    'Han_Jun': 'voices/en_US-norman-medium.onnx',
    'Chef_Baek': 'voices/en_US-kusal-medium.onnx',
    'Minji_Kang': 'voices/en_US-kristin-medium.onnx',
    'Dr_Shin': 'voices/en_US-amy-medium.onnx',
    'Jay_Park': 'voices/en_US-norman-medium.onnx',
    'president': 'voices/en_US-ryan-high.onnx',
    'Riley_Stone': 'voices/en_US-kristin-medium.onnx',
    'Dr_Harold': 'voices/en_US-john-medium.onnx',
    'Senator_Clark': 'voices/en_US-kusal-medium.onnx',
    'Tommy_Reed': 'voices/en_US-bryce-medium.onnx',
    'Evelyn_Moss': 'voices/en_US-amy-medium.onnx',
    'Jordan_Fields': 'voices/en_US-norman-medium.onnx',
}

# Run Piper in this interpreter so its NumPy/ONNX dependencies come from the
# project's virtual environment, rather than a different global installation.
def _synthesize(text: str, speaker: str, output_filename: str) -> str:
    model_path = VOICES[speaker]
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Missing Piper voice model: {model_path}")
    os.makedirs(os.path.dirname(output_filename) or ".", exist_ok=True)
    com_env = os.environ.copy()
    com_env["PYTHONUTF8"] = "1"
    com_env["PYTHONIOENCODING"] = "utf-8"
    try:
        subprocess.run(
            [sys.executable, "-m", "piper", "--model", model_path,
             "--output-file", output_filename],
            input=text.encode("utf-8"), env=com_env, capture_output=True,
            check=True, timeout=120,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        if os.path.exists(output_filename):
            os.remove(output_filename)
        raise RuntimeError("Piper synthesis failed; check piper-tts and numpy<2 in the active environment") from error
    return output_filename


def split_sentences(text):
    """Conservative English boundaries; retain punctuation and the final fragment."""
    text = " ".join(text.split())
    if not text:
        raise ValueError("Cannot synthesize empty speech")
    sentences, start = [], 0
    for boundary in re.finditer(r"[.!?]+[\"'”’)]*(?:\s+|$)", text):
        end = boundary.end()
        candidate = text[start:end].strip()
        # Titles, initials and common abbreviations are not sentence boundaries.
        if end < len(text) and re.search(
            r"(?:\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc)\.|\b[A-Z]\.|\b[A-Za-z]\.[A-Za-z]\.)$",
            candidate,
        ):
            continue
        sentences.append(candidate)
        start = end
    if text[start:].strip():
        sentences.append(text[start:].strip())
    return sentences


def _combine_wavs(paths, output):
    """Write PCM in input order; reject empty, truncated or incompatible segments."""
    with wave.open(str(paths[0]), "rb") as first:
        parameters = first.getparams()
    expected = (parameters.nchannels, parameters.sampwidth, parameters.framerate)
    total = 0
    with wave.open(str(output), "wb") as destination:
        destination.setparams(parameters)
        for path in paths:
            with wave.open(str(path), "rb") as source:
                shape = (source.getnchannels(), source.getsampwidth(), source.getframerate())
                frames = source.getnframes()
                if source.getcomptype() != "NONE" or frames <= 0:
                    raise ValueError("Sentence WAV must contain uncompressed audio")
                if shape != expected:
                    raise ValueError("Incompatible sentence WAV formats")
                data = source.readframes(frames)
                if len(data) != frames * shape[0] * shape[1]:
                    raise ValueError("Truncated sentence WAV")
                destination.writeframesraw(data)
                total += frames
    return total


def tts_generate(text: str, speaker: str, output_filename: str) -> str:
    """Synthesize sentences and atomically publish one complete, ordered WAV.

    The first sentence is generated first, then at most two Piper processes run
    concurrently. Temporary sentence files are removed on success or failure.
    Callers receive only the complete WAV, after validated speech is available.
    """
    sentences = split_sentences(text)
    target = Path(output_filename)
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="tts-", dir=str(target.parent)) as directory:
        paths = [Path(directory) / ("sentence-%03d.wav" % index)
                 for index in range(len(sentences))]
        _synthesize(sentences[0], speaker, str(paths[0]))
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(_synthesize, sentence, speaker, str(path))
                       for sentence, path in zip(sentences[1:], paths[1:])]
            for future in futures:
                future.result()  # Propagate every worker error before publication.
        combined = Path(directory) / "complete.wav"
        _combine_wavs(paths, combined)
        combined.replace(target)
    return output_filename
