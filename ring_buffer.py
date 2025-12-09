import struct
import os
from typing import Optional, Tuple, List
from multiprocessing import shared_memory


# ============================================
# Protocol Structures
# ============================================

class TTSHeader:
    """
    TTS Header (32 bytes)

    [0-3]   sequence_number (uint32)
    [4-7]   total_length (uint32) - audio data length in bytes
    [8-11]  user_sequence (uint32)
    [12]    speaker (uint8)
    [13]     animation_type (uint8)
    [14]    flag (uint8)
    [15-31] reserved (17 bytes)
    """
    SIZE = 32

    def __init__(self):
        self.sequence_number = 0
        self.total_length = 0
        self.user_sequence = 0
        self.speaker = 0
        self.animation_type = 0
        self.flag = 0

    def pack(self) -> bytes:
        return struct.pack(
            '<IIIBBB17x',
            self.sequence_number,
            self.total_length,
            self.user_sequence,
            self.speaker,
            self.animation_type,
            self.flag
        )

    @staticmethod
    def unpack(data: bytes) -> 'TTSHeader':
        values = struct.unpack('<IIIBBB17x', data[:TTSHeader.SIZE])
        header = TTSHeader()
        header.sequence_number = values[0]
        header.total_length = values[1]
        header.user_sequence = values[2]
        header.speaker = values[3]
        header.animation_type = values[4]
        header.flag = values[5]
        return header


class STTHeader:
    """
    STT Header (32 bytes)

    [0-3]   sequence_number (uint32) - 0 for participant list, else audio seq
    [4-7]   data_length (uint32) - data length in bytes
    [8-11]  user_sequence (uint32)
    [12-31] reserved (20 bytes)
    """
    SIZE = 32

    def __init__(self):
        self.sequence_number = 0
        self.data_length = 0
        self.user_sequence = 0

    def pack(self) -> bytes:
        return struct.pack(
            '<III20x',
            self.sequence_number,
            self.data_length,
            self.user_sequence
        )

    @staticmethod
    def unpack(data: bytes) -> 'STTHeader':
        values = struct.unpack('<III20x', data[:STTHeader.SIZE])
        header = STTHeader()
        header.sequence_number = values[0]
        header.data_length = values[1]
        header.user_sequence = values[2]
        return header


# ============================================
# Ring Buffer with Metadata
# ============================================

class RingBufferWithMetadata:
    """Ring Buffer with Metadata"""

    def __init__(self, name: str, buffer_size: int = 20 * 1024 * 1024):
        """
        Args:
            name: Shared memory's name
            buffer_size: Actual audio data buffer size (bytes)
        """
        self.name = name
        self.buffer_size = buffer_size
        # Metadata: write_ptr(8) + read_ptr(8) = 16 bytes
        self.metadata_size = 16
        self.total_size = self.metadata_size + buffer_size
        self.shm: Optional[shared_memory.SharedMemory] = None

    def create(self):
        """Create and Initialize Shared Memory"""
        print(f"Creating ring buffer: {self.name}")

        self.shm = shared_memory.SharedMemory(
            name=self.name,
            create=True,
            size=self.total_size
        )

        # Metadata initialization (write_ptr=0, read_ptr=0)
        self.shm.buf[:self.metadata_size] = b'\x00' * self.metadata_size

        print(f"Created: {self.name} ({self.buffer_size / 1024 / 1024:.1f}MB)")

    def connect(self):
        """Connect to existing shared memory"""
        print(f"Connecting to ring buffer: {self.name}")
        self.shm = shared_memory.SharedMemory(name=self.name)
        print(f"Connected: {self.name}")

    def _get_write_ptr(self) -> int:
        return struct.unpack('<Q', bytes(self.shm.buf[0:8]))[0]

    def _set_write_ptr(self, value: int):
        self.shm.buf[0:8] = struct.pack('<Q', value)

    def _get_read_ptr(self) -> int:
        return struct.unpack('<Q', bytes(self.shm.buf[8:16]))[0]

    def _set_read_ptr(self, value: int):
        self.shm.buf[8:16] = struct.pack('<Q', value)

    def write_data(self, data: bytes) -> bool:
        """
        Write data to the ring buffer

        Args:
            data: Data to be written (header + payload; bytes)

        Returns:
            Success or Fail
        """
        if self.shm is None:
            return False

        data_size = len(data)
        write_ptr = self._get_write_ptr()
        read_ptr = self._get_read_ptr()

        used = int(write_ptr - read_ptr)
        available = self.buffer_size - used

        if data_size > available:
            print(f"Buffer full: need {data_size}, available {available}")
            return False

        # Write to the ring buffer
        write_pos = write_ptr % self.buffer_size
        buffer_start = self.metadata_size

        if write_pos + data_size <= self.buffer_size:
            self.shm.buf[buffer_start + write_pos:buffer_start + write_pos + data_size] = data
        else:
            first_part = self.buffer_size - write_pos
            self.shm.buf[buffer_start + write_pos:buffer_start + self.buffer_size] = data[:first_part]
            self.shm.buf[buffer_start:buffer_start + data_size - first_part] = data[first_part:]

        # Update write pointer
        self._set_write_ptr(write_ptr + data_size)

        return True

    def read_data(self, size: int) -> Optional[bytes]:
        """
        Read data from the ring buffer

        Args:
            size: Size to read (bytes)

        Returns:
            Read data or None
        """
        if self.shm is None:
            return None

        write_ptr = self._get_write_ptr()
        read_ptr = self._get_read_ptr()

        available = int(write_ptr - read_ptr)

        if available < size:
            return None

        # Read from the ring buffer
        read_pos = read_ptr % self.buffer_size
        buffer_start = self.metadata_size

        data = bytearray(size)

        if read_pos + size <= self.buffer_size:
            data[:] = self.shm.buf[buffer_start + read_pos:buffer_start + read_pos + size]
        else:
            first_part = self.buffer_size - read_pos
            data[:first_part] = self.shm.buf[buffer_start + read_pos:buffer_start + self.buffer_size]
            data[first_part:] = self.shm.buf[buffer_start:buffer_start + size - first_part]

        # Update read pointer
        self._set_read_ptr(read_ptr + size)

        return bytes(data)

    def get_available_bytes(self) -> int:
        if self.shm is None:
            return 0
        write_ptr = self._get_write_ptr()
        read_ptr = self._get_read_ptr()
        return int(write_ptr - read_ptr)

    def skip_to_latest(self):
        """Move the read pointer to the write pointer (skipping all data)"""
        if self.shm is None:
            return
        write_ptr = self._get_write_ptr()
        self._set_read_ptr(write_ptr)
        print(f"Skipped all buffered data in {self.name}")

    def cleanup(self):
        """Resource cleanup"""
        if self.shm:
            try:
                self.shm.close()
                self.shm.unlink()
            except:
                pass
