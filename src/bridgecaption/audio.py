"""Capture authorized Linux microphone or output-monitor audio via FFmpeg/PulseAudio."""
import array
import math
import shutil
import subprocess
import threading

RATE = 16000

def sources():
    """Return (display_name, pulse_source) pairs; monitor devices carry playback audio."""
    if not shutil.which("pactl"):
        return []
    try:
        sink = subprocess.run(["pactl", "get-default-sink"], capture_output=True, text=True, timeout=4).stdout.strip()
        result = subprocess.run(["pactl", "list", "short", "sources"], capture_output=True,
                                text=True, timeout=5, check=True)
    except (OSError, subprocess.SubprocessError):
        return []
    devices = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        name = parts[1]
        monitor = name.endswith(".monitor")
        label = f"{'System audio' if monitor else 'Microphone'}: {name}"
        if name == sink + ".monitor":
            label = "System audio (current output)"
        devices.append((label, name))
    devices.sort(key=lambda item: (item[0] != "System audio (current output)", item[0]))
    return devices

def audible(raw: bytes, threshold: float = .006) -> bool:
    if len(raw) < 2:
        return False
    samples = array.array("h")
    samples.frombytes(raw[:len(raw) // 2 * 2])
    values = samples[::16]
    if not values:
        return False
    return math.sqrt(sum(v * v for v in values) / len(values)) / 32768.0 > threshold

class AudioCapture:
    def __init__(self, source: str, seconds: float = 4):
        if not source or not 1 <= seconds <= 12:
            raise ValueError("Invalid audio source or chunk duration")
        self.source, self.seconds = source, seconds
        self.process = None
        self.stopped = threading.Event()

    def start(self):
        if not shutil.which("ffmpeg"):
            raise RuntimeError("Install ffmpeg to capture audio.")
        self.process = subprocess.Popen(
            ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-f", "pulse",
             "-i", self.source, "-ac", "1", "-ar", str(RATE), "-f", "s16le", "pipe:1"],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        return self

    def chunks(self):
        if self.process is None or self.process.stdout is None:
            raise RuntimeError("Capture has not started")
        required = int(RATE * self.seconds) * 2
        while not self.stopped.is_set():
            data = bytearray()
            while len(data) < required and not self.stopped.is_set():
                piece = self.process.stdout.read(required - len(data))
                if not piece:
                    if self.stopped.is_set():
                        return
                    raise RuntimeError("Audio source disconnected or cannot be opened")
                data.extend(piece)
            if len(data) == required:
                yield bytes(data)

    def stop(self):
        self.stopped.set()
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)

class SpeechRecognizer:
    def __init__(self, model: str = "base"):
        if model not in {"tiny", "base", "small", "medium"}:
            raise ValueError("Invalid Whisper model setting")
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model, device="cpu", compute_type="int8")

    def transcribe(self, raw: bytes, language: str = "auto") -> tuple[str, str]:
        import numpy as np
        samples = np.frombuffer(raw, dtype="<i2").astype("float32") / 32768.0
        if samples.size < RATE:
            return "", language
        segments, info = self.model.transcribe(samples, language=None if language == "auto" else language,
                                                beam_size=3, vad_filter=True, condition_on_previous_text=False)
        return " ".join(segment.text.strip() for segment in segments).strip(), info.language or language
