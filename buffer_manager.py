import struct
import os
import numpy as np
from typing import Union, Optional, Tuple, List
from ring_buffer import TTSHeader, STTHeader, RingBufferWithMetadata


# ============================================
# TTS Manager (Python → Unity)
# ============================================

class TTSManager:
    """Send TTS audio to Unity via ring buffer"""

    def __init__(self, ring_buffer: RingBufferWithMetadata):
        self.ring_buffer = ring_buffer
        self.sequence_counter = 0

    def send_tts_audio(self, audio_file_path: str, speaker: int, animation_type: int,
                       user_sequence: int, flag: int, sample_rate: int = 22050) -> bool:
        """
        Send generated TTS audio file to Unity

        Args:
            audio_file_path: Path to WAV audio file
            speaker: Speaker index
            animation_type: Animation type
            sample_rate: Target sample rate

        Returns:
            True if successful, False otherwise
        """
        import wave

        # Increment sequence number
        self.sequence_counter += 1
        seq = self.sequence_counter

        print(f"\n[TTS seq={seq} user={user_sequence}] Loading generated: {audio_file_path}")
        print(f"  speaker={speaker}, anim={animation_type}, flag={flag}")

        try:
            # Check if file exists
            if not os.path.exists(audio_file_path):
                print(f"  Error: File not found: {audio_file_path}")
                return False

            # Read WAV file
            with wave.open(audio_file_path, 'rb') as wav_file:
                n_channels = wav_file.getnchannels()
                sampwidth = wav_file.getsampwidth()
                framerate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                audio_bytes = wav_file.readframes(n_frames)

            # Convert to float32
            if sampwidth == 2:  # 16-bit
                audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
                audio_float = audio_data.astype(np.float32) / 32768.0
            elif sampwidth == 4:  # 32-bit
                audio_data = np.frombuffer(audio_bytes, dtype=np.int32)
                audio_float = audio_data.astype(np.float32) / 2147483648.0
            else:
                raise ValueError(f"Unsupported sample width: {sampwidth}")

            # Convert stereo to mono
            if n_channels == 2:
                audio_float = audio_float.reshape(-1, 2).mean(axis=1)

            # Resample if needed
            if framerate != sample_rate:
                from scipy import signal
                num_samples = int(len(audio_float) * sample_rate / framerate)
                audio_float = signal.resample(audio_float, num_samples)

            # Convert to bytes
            audio_bytes = audio_float.astype(np.float32).tobytes()

            # Create header
            header = TTSHeader()
            header.sequence_number = seq
            header.total_length = len(audio_bytes)
            header.user_sequence = user_sequence
            header.speaker = speaker
            header.animation_type = animation_type
            header.flag = flag

            # Combine header + audio data
            full_data = header.pack() + audio_bytes

            # Write to ring buffer
            success = self.ring_buffer.write_data(full_data)

            if success:
                print(f"  Sent: {len(audio_bytes)} bytes ({len(audio_float)/sample_rate:.2f}s)")
            else:
                print(f"  Failed to write to buffer")
            print("\n" + "=" * 16 + "\n")

            return success

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            return False


# ============================================
# STT Manager (Unity → Python)
# ============================================

class STTManager:
    """Receive audio from Unity and perform STT"""

    def __init__(self, ring_buffer: RingBufferWithMetadata):
        self.ring_buffer = ring_buffer
        self.whisper_model = None
        self._load_whisper_model()

    def _load_whisper_model(self):
        """Load Whisper model"""
        try:
            import whisper
            print("Loading Whisper model...")
            self.whisper_model = whisper.load_model("medium")
            print("Whisper model loaded")
        except ImportError:
            print("Warning: Whisper not installed. Install with: pip install openai-whisper")
            self.whisper_model = None
        except Exception as e:
            print(f"Warning: Failed to load Whisper model: {e}")
            self.whisper_model = None

    def process_stt_buffer(self) -> Union[None, List[int], Tuple[int, str]]:
        available = self.ring_buffer.get_available_bytes()

        if available < STTHeader.SIZE:
            return None

        # Read header
        header_bytes = self.ring_buffer.read_data(STTHeader.SIZE)
        if header_bytes is None:
            return None

        header = STTHeader.unpack(header_bytes)
        print(f"\n[STT seq={header.sequence_number}] Processing {header.data_length} bytes...")

        # participant list
        if header.sequence_number == 0:
            return self.receive_participant_list(header.data_length)
        else:
            return self.process_user_audio(header.sequence_number, header.data_length, header.user_sequence)

    def receive_participant_list(self, data_length: int) -> Optional[List[int]]:
        """
        Receive participant list from Unity (sequence_number=0)

        Args:
            data_length

        Returns:
            List of participant indices [int, int, int] or None
        """
        # Read data (3 int32 = 12 bytes)
        if data_length != 12:
            print(f"Warning: Invalid participant list size: {data_length}")
            return None

        data_bytes = self.ring_buffer.read_data(12)
        if data_bytes is None:
            return None

        participants = list(struct.unpack('<III', data_bytes))
        print(f"Received participants: {participants}")
        return participants

    def process_user_audio(self, sequence_number: int, data_length: int, user_seq: int) -> Optional[Tuple[int, int, str]]:
        """
        Receive user audio from Unity and perform STT

        Args:
            sequence_number
            data_length
            user_seq

        Returns:
            (sequence_number, user_sequence, transcribed_text) or None
        """
        # Read audio data
        audio_bytes = self.ring_buffer.read_data(data_length)
        if audio_bytes is None:
            return None

        # Convert to float32 array
        audio_data = np.frombuffer(audio_bytes, dtype=np.float32)

        # Perform STT
        text = self._transcribe_audio(audio_data, sample_rate=22050)

        if text:
            print(f"  Transcribed: {text}")
            return (sequence_number, user_seq, text)
        else:
            return None

    def _transcribe_audio(self, audio_data: np.ndarray, sample_rate: int) -> Optional[str]:
        """Perform Whisper STT"""
        if self.whisper_model is None:
            return "[Whisper not loaded]"

        try:
            # Resample to 16kHz
            if sample_rate != 16000:
                from scipy import signal
                num_samples = int(len(audio_data) * 16000 / sample_rate)
                audio_resampled = signal.resample(audio_data, num_samples)
            else:
                audio_resampled = audio_data

            # Run Whisper
            result = self.whisper_model.transcribe(
                audio_resampled,
                language='en',
                fp16=False
            )

            return result["text"].strip()
        except Exception as e:
            print(f"  Warning: STT error: {e}")
            return None
