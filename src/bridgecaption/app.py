"""Visible draggable translation overlay for authorized Linux audio.

The application starts no recording until START LISTENING is clicked.
The text translator sends recognized text to Gemini; microphone/system audio
remains on the local computer.
"""
import os
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox

from dotenv import load_dotenv

from .audio import AudioCapture, SpeechRecognizer, audible, sources
from .context import parse_glossary
from .pipeline import Pipeline

LANGUAGES = {
    "auto": "Auto detect", "en": "English", "es": "Spanish", "fr": "French",
    "de": "German", "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
    "ar": "Arabic", "hi": "Hindi", "pt": "Portuguese", "ru": "Russian",
    "it": "Italian", "sw": "Swahili", "yo": "Yoruba", "am": "Amharic",
    "bn": "Bengali", "ta": "Tamil", "te": "Telugu", "ur": "Urdu",
    "tr": "Turkish", "fa": "Persian", "id": "Indonesian", "vi": "Vietnamese",
    "th": "Thai", "uk": "Ukrainian", "pl": "Polish", "nl": "Dutch",
    "sv": "Swedish", "da": "Danish", "fi": "Finnish", "no": "Norwegian",
    "he": "Hebrew", "el": "Greek", "cs": "Czech", "hu": "Hungarian",
    "ro": "Romanian", "bg": "Bulgarian", "hr": "Croatian", "sr": "Serbian",
    "sk": "Slovak", "sl": "Slovenian", "lt": "Lithuanian", "lv": "Latvian",
    "et": "Estonian", "ms": "Malay", "tl": "Tagalog", "ne": "Nepali",
    "mr": "Marathi", "gu": "Gujarati", "pa": "Punjabi", "kn": "Kannada",
    "ml": "Malayalam", "my": "Burmese", "km": "Khmer", "lo": "Lao",
    "ka": "Georgian", "hy": "Armenian", "az": "Azerbaijani",
    "kk": "Kazakh", "uz": "Uzbek", "mn": "Mongolian",
    "af": "Afrikaans", "sq": "Albanian", "ca": "Catalan", "eu": "Basque",
    "gl": "Galician", "is": "Icelandic", "cy": "Welsh", "so": "Somali",
    "ha": "Hausa", "si": "Sinhala", "jw": "Javanese", "su": "Sundanese",
    "mt": "Maltese", "mk": "Macedonian", "be": "Belarusian",
}
TOPICS = ["auto", "general", "technology", "business", "education", "healthcare", "gaming", "casual"]
BG = "#121b20"
PANEL = "#19272d"
TEXT = "#eaf4ed"
MUTED = "#a0b8af"
ACCENT = "#a9e4c4"

class Application:
    def __init__(self, root):
        self.root = root
        self.root.title("BridgeCaption | AI translation")
        self.root.geometry("860x400+220+380")
        self.root.minsize(650, 300)
        self.root.configure(bg=BG)
        self.root.attributes("-topmost", True)
        self.root.overrideredirect(True)
        self.worker = None
        self.stop_event = threading.Event()
        self.capture = None
        self.events = queue.Queue()
        self.pipeline = Pipeline()
        self.context_enabled = tk.BooleanVar(value=True)
        self.glossary = {}
        self.devices = []
        self._drag_xy = None
        self.is_pinned = True
        self._ui()
        self.refresh_sources()
        self.root.after(100, self.poll)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def label(self, parent, text, size=11, color=TEXT, bold=False):
        return tk.Label(parent, text=text, bg=parent.cget("bg"), fg=color,
                        font=("DejaVu Sans", size, "bold" if bold else "normal"),
                        anchor="w")

    def button(self, parent, text, command, accent=False):
        b = tk.Button(parent, text=text, command=command, relief="flat", bd=0,
                      bg=ACCENT if accent else "#30443f",
                      fg="#14211a" if accent else TEXT, activebackground="#d5f5e2",
                      font=("DejaVu Sans", 10, "bold"), padx=12, pady=8, cursor="hand2")
        return b

    def picker(self, parent, values, width=15):
        box = ttk.Combobox(parent, state="readonly", values=values, width=width)
        box.current(0)
        box.pack(side="left", padx=(0, 8), pady=4)
        return box

    def _ui(self):
        top = tk.Frame(self.root, bg=BG, padx=18, pady=13)
        top.pack(fill="x")
        self.label(top, "BRIDGE / CAPTION", 15, ACCENT, True).pack(side="left")
        self.button(top, "×", self.close).pack(side="right", padx=(6, 0))
        self.button(top, "—", self.root.iconify).pack(side="right", padx=(6, 0))
        self.pin_button = self.button(top, "PINNED", self.toggle_pin)
        self.pin_button.pack(side="right", padx=4)
        self.label(top, "AI LIVE TRANSLATION", 9, MUTED, True).pack(side="right", padx=15)
        for widget in (top,):
            widget.bind("<ButtonPress-1>", self.drag_start)
            widget.bind("<B1-Motion>", self.drag_move)

        controls = tk.Frame(self.root, bg=BG, padx=18)
        controls.pack(fill="x")
        self.source_box = self.picker(controls, ["No audio sources"], width=28)
        self.source_box.bind("<<ComboboxSelected>>", lambda _event: self.reset_history())
        self.source_labels = sorted(LANGUAGES.items(), key=lambda kv: kv[1])
        self.source_picker = self.picker(controls, [f"{name} [{code}]" for code, name in self.source_labels], 15)
        self.target_labels = [item for item in self.source_labels if item[0] != "auto"]
        self.target_picker = self.picker(controls, [f"{name} [{code}]" for code, name in self.target_labels], 14)
        self.source_picker.set("Auto detect [auto]")
        self.target_picker.set("English [en]")
        self.topic_box = self.picker(controls, TOPICS, width=12)
        self.start_button = self.button(controls, "START LISTENING", self.toggle_capture, accent=True)
        self.start_button.pack(side="right")
        settings = tk.Frame(self.root, bg=BG, padx=18, pady=7)
        settings.pack(fill="x")
        tk.Checkbutton(settings, text="Context + slang", variable=self.context_enabled,
                       bg=BG, fg=TEXT, selectcolor=PANEL, activebackground=BG,
                       activeforeground=TEXT).pack(side="left")
        self.button(settings, "GLOSSARY", self.edit_glossary).pack(side="left", padx=8)
        self.button(settings, "REFRESH AUDIO", self.refresh_sources).pack(side="left", padx=8)
        self.button(settings, "CLEAR CONTEXT", self.reset_history).pack(side="left")

        caption_panel = tk.Frame(self.root, bg=PANEL, padx=20, pady=14)
        caption_panel.pack(fill="both", expand=True, padx=18, pady=7)
        self.label(caption_panel, "ORIGINAL SPEECH", 9, MUTED, True).pack(fill="x")
        self.source_text = self.label(caption_panel, "Your transcript appears here.", 14, MUTED)
        self.source_text.configure(wraplength=760, justify="left")
        self.source_text.pack(fill="x", pady=(6, 10))
        self.label(caption_panel, "LIVE TRANSLATION", 9, ACCENT, True).pack(fill="x")
        self.translated_text = self.label(caption_panel, "Play a video or join a meeting.", 20, TEXT, True)
        self.translated_text.configure(wraplength=760, justify="left")
        self.translated_text.pack(fill="both", expand=True, pady=(6, 0))

        bottom = tk.Frame(self.root, bg=BG, padx=18)
        bottom.pack(fill="x")
        self.test_text = tk.Entry(bottom, bg="#27383d", fg=TEXT, insertbackground=TEXT,
                                  relief="flat", font=("DejaVu Sans", 12))
        self.test_text.pack(side="left", fill="x", expand=True, ipady=8, padx=(0, 8))
        self.test_text.insert(0, "Ese código está chido")
        self.test_text.bind("<Return>", lambda _event: self.test_translate())
        self.button(bottom, "TRANSLATE TEXT", self.test_translate).pack(side="right")
        self.status_var = tk.StringVar(value="Select an audio source. Requires a Gemini API key.")
        tk.Label(self.root, textvariable=self.status_var, bg=BG, fg=MUTED,
                 anchor="w", wraplength=790, padx=18, pady=8).pack(fill="x")

    def source_lang(self):
        return self.source_picker.get().rsplit("[", 1)[-1].rstrip("]")

    def target_lang(self):
        return self.target_picker.get().rsplit("[", 1)[-1].rstrip("]")

    def drag_start(self, event):
        self._drag_xy = (event.x_root - self.root.winfo_x(),
                         event.y_root - self.root.winfo_y())

    def drag_move(self, event):
        if self._drag_xy:
            self.root.geometry(f"+{event.x_root - self._drag_xy[0]}+{event.y_root - self._drag_xy[1]}")

    def toggle_pin(self):
        self.is_pinned = not self.is_pinned
        self.root.attributes("-topmost", self.is_pinned)
        self.pin_button.config(text="PINNED" if self.is_pinned else "UNPINNED")

    def reset_history(self):
        self.pipeline.reset()
        self.status_var.set("Conversation context cleared.")

    def refresh_sources(self):
        self.devices = sources()
        labels = [name for name, _device in self.devices]
        self.source_box["values"] = labels or ["No audio sources: check pactl"]
        self.source_box.current(0)
        if not labels:
            self.status_var.set("No audio source found; install pulseaudio-utils, FFmpeg and check PipeWire/PulseAudio.")

    def edit_glossary(self):
        popup = tk.Toplevel(self.root)
        popup.title("Meeting glossary")
        popup.geometry("500x340")
        popup.configure(bg=BG)
        self.label(popup, "One line per term: spoken = preferred translation", 11).pack(padx=12, pady=12)
        editor = tk.Text(popup, bg=PANEL, fg=TEXT, insertbackground=TEXT, relief="flat")
        editor.pack(fill="both", expand=True, padx=12)
        editor.insert("1.0", "\n".join(f"{k} = {v}" for k, v in self.glossary.items()))
        def save():
            self.glossary = parse_glossary(editor.get("1.0", "end"))
            self.status_var.set(f"{len(self.glossary)} glossary terms active; stored in memory only.")
            popup.destroy()
        self.button(popup, "SAVE GLOSSARY", save, True).pack(pady=12)

    def publish(self, kind, *values):
        self.events.put((kind, values))

    def test_translate(self):
        text = self.test_text.get().strip()
        if not text:
            return
        language = self.source_lang()
        if language == "auto":
            self.status_var.set("For typed text, select the source language manually.")
            return
        def task():
            try:
                self.publish("status", "Translating typed text...")
                result = self.pipeline.translate(text, language, self.target_lang(),
                                                 self.topic_box.get(), self.glossary,
                                                 self.context_enabled.get())
                self.publish("caption", result.original, result.translated, result.source_language, "Text")
            except Exception as exc:
                self.publish("error", str(exc))
        if self.worker and self.worker.is_alive():
            self.status_var.set("Busy; try after current translation completes.")
            return
        self.worker = threading.Thread(target=task, daemon=True)
        self.worker.start()

    def toggle_capture(self):
        if self.capture is not None:
            self.stop_event.set()
            self.capture.stop()
            self.start_button.config(text="STOPPING...")
            self.status_var.set("Stopping audio capture...")
            return
        if not self.devices:
            self.status_var.set("No audio devices available.")
            return
        if not os.getenv("GEMINI_API_KEY"):
            self.status_var.set("Set GEMINI_API_KEY in .env before listening.")
            return
        if self.worker and self.worker.is_alive():
            self.status_var.set("Wait until the previous translation finishes.")
            return
        self.stop_event.clear()
        device = self.devices[self.source_box.current()][1]
        language = self.source_lang()
        target = self.target_lang()
        topic = self.topic_box.get()
        glossary = dict(self.glossary)
        contextual = self.context_enabled.get()
        self.start_button.config(text="STOP LISTENING")
        def task():
            capture = None
            try:
                self.publish("status", "Loading Whisper (first run downloads model weights)...")
                recognizer = SpeechRecognizer(os.getenv("WHISPER_MODEL", "base"))
                capture = AudioCapture(device, float(os.getenv("AUDIO_CHUNK_SECONDS", "4"))).start()
                self.capture = capture
                self.publish("status", "Listening to selected audio; original audio remains local.")
                for raw in capture.chunks():
                    if self.stop_event.is_set():
                        break
                    if not audible(raw):
                        continue
                    begin = time.monotonic()
                    original, detected = recognizer.transcribe(raw, language)
                    if not original or self.stop_event.is_set():
                        continue
                    self.publish("heard", original, detected)
                    result = self.pipeline.translate(original, detected, target, topic,
                                                     glossary, contextual)
                    if self.stop_event.is_set():
                        break
                    self.publish("caption", result.original, result.translated,
                                 detected, f"{time.monotonic() - begin:.1f}s after audio chunk")
            except Exception as exc:
                if not self.stop_event.is_set():
                    self.publish("error", str(exc))
            finally:
                if capture:
                    capture.stop()
                self.publish("stopped")
        self.worker = threading.Thread(target=task, daemon=True)
        self.worker.start()

    def poll(self):
        try:
            while True:
                kind, values = self.events.get_nowait()
                if kind == "status":
                    self.status_var.set(values[0])
                elif kind == "heard":
                    self.source_text.config(text=values[0])
                    self.translated_text.config(text="Translating…")
                elif kind == "caption":
                    original, translated, lang, latency = values
                    self.source_text.config(text=f"[{lang}]  {original[:700]}")
                    self.translated_text.config(text=translated[:900])
                    self.status_var.set(f"Translation ready · {latency}")
                elif kind == "error":
                    self.status_var.set("ERROR · " + values[0])
                elif kind == "stopped":
                    self.capture = None
                    self.start_button.config(text="START LISTENING")
        except queue.Empty:
            pass
        self.root.after(100, self.poll)

    def close(self):
        self.stop_event.set()
        if self.capture is not None:
            self.capture.stop()
        self.root.destroy()

def main():
    load_dotenv()
    root = tk.Tk()
    Application(root)
    root.mainloop()

if __name__ == "__main__":
    main()
