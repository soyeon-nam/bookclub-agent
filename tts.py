import os
import subprocess
import sys

# --- Configuration ---
voice_path = "voices/en_US-danny-low.onnx"
output_dir = "dialogue_audio"
os.makedirs(output_dir, exist_ok=True)

VOICES={
    "Riley_Stone": "voices/en_US-kristin-medium.onnx",
    "Dr_Harold": "voices/en_US-john-medium.onnx",
    "Senator_Clark": "voices/en_US-kusal-medium.onnx",
    "Tommy_Reed": "voices/en_US-bryce-medium.onnx",
    "president": "voices/en_US-ryan-high.onnx",
    "Evelyn_Moss": "voices/en_US-amy-medium.onnx",
    "Jordan_Fields": "voices/en_US-norman-medium.onnx",
}

# --- piper ---
if sys.platform == "win32":
    scripts_dir = os.path.join(sys.prefix, 'Scripts')
    PIPER_EXE_PATH = os.path.join(scripts_dir, 'piper.exe')
else:
    bin_dir = os.path.join(sys.prefix, 'bin')
    PIPER_EXE_PATH = os.path.join(bin_dir, 'piper')

if not os.path.exists(PIPER_EXE_PATH):
    print(f"ERROR: 'piper' executable not found at path: {PIPER_EXE_PATH}")
    print("Please run 'pip install piper-tts' to install it")

# --- TTS ---
def tts_generate(text: str, speaker: str, output_filename: str) -> str:
    """
    Generate speech audio from text using Piper TTS.

    Args:
        text (str): The text to convert to speech.
        speaker (str): The path to the speaker model file.
        output_filename (str): The filename to save the output audio.

    Returns:
        str: The path to the generated audio file.
    """
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    # cmd = f'piper --model {VOICES[speaker]} --output-file {output_filename}  "{text}"'
    # os.system(cmd)
    # return output_filename

    com_env = os.environ.copy()
    com_env["PYTHONUTF8"] = "1"
    com_env["PYTHONIOENCODING"] = "utf-8"

    try:
        result = subprocess.run(
            [
                PIPER_EXE_PATH,
                '--model', VOICES[speaker],
                '--output-file', output_filename
            ],
            input=text.encode('utf-8', 'ignore'),
            env=com_env,
            capture_output=True,
            check=True
        )
        return output_filename
    except FileNotFoundError:
        print(f"FATAL ERROR: 'piper' executable not found at path: {PIPER_EXE_PATH}")
        raise
    except subprocess.CalledProcessError as e:
        print(f"PIPER TTS ERROR: {e.stderr.decode('utf-8', errors='ignore')}")
        raise


# tts_generate("Hello, this is a test.", "Riley_Stone", os.path.join(output_dir, "test_riley_stone.wav"))
# tts_generate("Hello, this is a test.", "Dr_Harold", os.path.join(output_dir, "test_dr_harold.wav"))
# tts_generate("Hello, this is a test.", "Senator_Clark", os.path.join(output_dir, "test_senator_clark.wav"))
# tts_generate("Hello, this is a test.", "Tommy_Reed", os.path.join(output_dir, "test_tommy_reed.wav"))
# tts_generate("Hello, this is a test.", "president", os.path.join(output_dir, "test_president.wav"))
