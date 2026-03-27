import os
import subprocess
import sys

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
def tts_generate(text: str, speaker: str, output_filename: str) -> str:
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
