"""Visible draggable translation overlay for authorized Linux audio.

The application starts no recording until START LISTENING is clicked.
The text translator sends recognized text to Gemini; microphone/system audio
remains on the local computer.
"""
import os
import queue
import threading
import tkinter as tk
from tkinter import ttk, messagebox

from dotenv import load_dotenv

from .audio import sources
from .session import LiveSession
from .context import parse_glossary
from .pipeline import Pipeline
from .provider import TranslationRequest
from .providers import PROVIDERS, key_available, needs_cloud, create_translator
from .overlay import apply_opacity, opacity_percent, window_size, MIN_OPACITY_PERCENT, MAX_OPACITY_PERCENT

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
        self.root.geometry("760x282+220+340")
        self.root.minsize(560, 245)
        self.root.configure(bg=BG)
        self.root.attributes("-topmost", True)
        self.root.overrideredirect(False)
        self.worker = None
        self.text_generation = 0
        self.events = queue.Queue()
        self.engine_selected = os.getenv("BRIDGE_TRANSLATOR", "ollama").strip().lower()
        if self.engine_selected not in PROVIDERS:
            self.engine_selected = "ollama"
        self.pipeline = Pipeline(translator=create_translator(self.engine_selected))
        # Typed-text history is never shared with a live listening session.
        self.live_session = None
        self.session_generation = 0
        self.consent_given = False
        self.consent_provider = None
        self.closed = False
        self.context_enabled = tk.BooleanVar(value=True)
        self.glossary = {}
        self.devices = []
        self._drag_xy = None
        self.is_pinned = True
        self.compact = True
        self.opacity_value = tk.DoubleVar(value=opacity_percent(
            os.getenv("BRIDGE_OVERLAY_OPACITY", "92")))
        self._ui()
        self._apply_overlay_mode()
        self._apply_opacity(self.opacity_value.get(), notify=False)
        self.refresh_sources()
        self.root.bind("<Configure>", self._resize_caption_wrap)
        self.root.bind("<Control-comma>", lambda _event: self.toggle_settings())
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
        top = tk.Frame(self.root, bg=BG, padx=12, pady=7)
        top.pack(fill="x")
        self.label(top, "BRIDGE / CAPTION", 13, ACCENT, True).pack(side="left")
        self.button(top, "×", self.close).pack(side="right", padx=(4, 0))
        self.button(top, "—", self.root.iconify).pack(side="right", padx=(4, 0))
        self.pin_button = self.button(top, "PINNED", self.toggle_pin)
        self.pin_button.pack(side="right", padx=3)
        self.start_button = self.button(top, "START", self.toggle_capture, accent=True)
        self.start_button.pack(side="right", padx=3)
        self.settings_button = self.button(top, "SETTINGS", self.toggle_settings)
        self.settings_button.pack(side="right", padx=3)
        top.bind("<ButtonPress-1>", self.drag_start)
        top.bind("<B1-Motion>", self.drag_move)

        self.summary_var = tk.StringVar(value="Ready · choose a source in Settings")
        self.summary_label = tk.Label(self.root, textvariable=self.summary_var,
                                      bg=BG, fg=MUTED, anchor="w",
                                      font=("DejaVu Sans", 9), padx=16, pady=1)
        self.summary_label.pack(fill="x")

        controls = tk.Frame(self.root, bg=BG, padx=12)
        self.controls_frame = controls
        controls.pack(fill="x")
        self.source_box = self.picker(controls, ["No audio sources"], width=28)
        self.source_box.bind("<<ComboboxSelected>>", lambda _event: self._on_source_change())
        self.source_labels = sorted(LANGUAGES.items(), key=lambda kv: kv[1])
        self.source_picker = self.picker(controls, [f"{name} [{code}]" for code, name in self.source_labels], 15)
        self.target_labels = [item for item in self.source_labels if item[0] != "auto"]
        self.target_picker = self.picker(controls, [f"{name} [{code}]" for code, name in self.target_labels], 14)
        self.source_picker.set("Auto detect [auto]")
        self.target_picker.set("English [en]")
        self.topic_box = self.picker(controls, TOPICS, width=12)
        self.source_picker.bind("<<ComboboxSelected>>", lambda _event: self._on_source_change())
        self.target_picker.bind("<<ComboboxSelected>>", lambda _event: self._on_source_change())
        settings = tk.Frame(self.root, bg=BG, padx=12, pady=5)
        self.settings_frame = settings
        settings.pack(fill="x")
        tk.Checkbutton(settings, text="Context + slang", variable=self.context_enabled,
                       bg=BG, fg=TEXT, selectcolor=PANEL, activebackground=BG,
                       activeforeground=TEXT).pack(side="left")
        self.button(settings, "GLOSSARY", self.edit_glossary).pack(side="left", padx=8)
        self.button(settings, "REFRESH AUDIO", self.refresh_sources).pack(side="left", padx=8)
        self.button(settings, "CLEAR CONTEXT", self.reset_history).pack(side="left")
        self.label(settings, "Opacity", 9, MUTED).pack(side="left", padx=(15, 4))
        ttk.Scale(settings, from_=MIN_OPACITY_PERCENT, to=MAX_OPACITY_PERCENT,
                  orient="horizontal", length=105, variable=self.opacity_value,
                  command=self._on_opacity_change).pack(side="left", padx=(0, 5))
        self.opacity_label = tk.StringVar(value=f"{opacity_percent(self.opacity_value.get())}%")
        tk.Label(settings, textvariable=self.opacity_label, bg=BG, fg=TEXT,
                 font=("DejaVu Sans", 9)).pack(side="left")
        engine_controls = tk.Frame(self.root, bg=BG, padx=12, pady=2)
        self.engine_controls_frame = engine_controls
        engine_controls.pack(fill="x")
        self.label(engine_controls, "TRANSLATION ENGINE", 9, MUTED, True).pack(side="left", padx=(0, 8))
        self.engine_box = self.picker(engine_controls, PROVIDERS, width=12)
        self.engine_box.set(self.engine_selected)
        self.engine_box.bind("<<ComboboxSelected>>", lambda _event: self.change_engine())
        self.button(engine_controls, "CHECK ENGINE", self.check_engine).pack(side="left", padx=8)
        self.label(engine_controls, "Local / cloud selection is explicit", 9, MUTED).pack(side="left", padx=8)

        caption_panel = tk.Frame(self.root, bg=PANEL, padx=15, pady=9)
        self.caption_panel = caption_panel
        caption_panel.pack(fill="both", expand=True, padx=12, pady=(6, 4))
        self.label(caption_panel, "ORIGINAL SPEECH", 9, MUTED, True).pack(fill="x")
        self.source_text = self.label(caption_panel, "Your transcript appears here.", 14, MUTED)
        self.source_text.configure(wraplength=760, justify="left")
        self.source_text.pack(fill="x", pady=(6, 10))
        self.label(caption_panel, "LIVE TRANSLATION", 9, ACCENT, True).pack(fill="x")
        self.translated_text = self.label(caption_panel, "Play a video or join a meeting.", 20, TEXT, True)
        self.translated_text.configure(wraplength=760, justify="left")
        self.translated_text.pack(fill="both", expand=True, pady=(6, 0))

        bottom = tk.Frame(self.root, bg=BG, padx=12)
        self.bottom_frame = bottom
        bottom.pack(fill="x")
        self.test_text = tk.Entry(bottom, bg="#27383d", fg=TEXT, insertbackground=TEXT,
                                  relief="flat", font=("DejaVu Sans", 12))
        self.test_text.pack(side="left", fill="x", expand=True, ipady=8, padx=(0, 8))
        self.test_text.insert(0, "Ese código está chido")
        self.test_text.bind("<Return>", lambda _event: self.test_translate())
        self.button(bottom, "TRANSLATE TEXT", self.test_translate).pack(side="right")
        self.status_var = tk.StringVar(value="Select a microphone/system monitor and a translation engine.")
        self.status_label = tk.Label(self.root, textvariable=self.status_var, bg=BG,
                                     fg=MUTED, anchor="w", justify="left",
                                     wraplength=700, padx=15, pady=4,
                                     font=("DejaVu Sans", 9))
        self.status_label.pack(fill="x")

    def _apply_overlay_mode(self):
        """Collapse configuration without losing captions or Start/Stop controls."""
        for panel in (self.controls_frame, self.settings_frame,
                      self.engine_controls_frame, self.bottom_frame):
            panel.pack_forget()
        if not self.compact:
            for panel in (self.controls_frame, self.settings_frame,
                          self.engine_controls_frame):
                panel.pack(fill="x", before=self.caption_panel)
            self.bottom_frame.pack(fill="x", before=self.status_label)
        self.settings_button.config(text="SETTINGS" if self.compact else "HIDE SETTINGS")
        self.source_text.config(font=("DejaVu Sans", 11 if self.compact else 14))
        self.translated_text.config(font=("DejaVu Sans", 17 if self.compact else 20, "bold"))
        width, height = window_size(self.compact)
        self.root.minsize(560 if self.compact else 760,
                          245 if self.compact else 400)
        self.root.geometry(f"{width}x{height}")
        self._resize_caption_wrap()

    def toggle_settings(self):
        self.compact = not self.compact
        self._apply_overlay_mode()

    def _on_opacity_change(self, value):
        self._apply_opacity(value, notify=True)

    def _apply_opacity(self, value, *, notify=True):
        percent = opacity_percent(value)
        self.opacity_label.set(f"{percent}%")
        available = apply_opacity(self.root, percent)
        if notify and not available:
            self.status_var.set("Window transparency unavailable in this desktop session.")
        return available

    def _resize_caption_wrap(self, event=None):
        if event is not None and event.widget is not self.root:
            return
        width = max(280, self.root.winfo_width() - 64)
        self.source_text.config(wraplength=width)
        self.translated_text.config(wraplength=width)
        self.status_label.config(wraplength=width)

    def _update_overlay_summary(self):
        if not hasattr(self, "summary_var"):
            return
        engine = self.engine_box.get().upper()
        source = self.source_lang().upper()
        target = self.target_lang().upper()
        device = self.source_box.get().strip()
        if len(device) > 55:
            device = device[:52] + "..."
        live = "LIVE" if self.live_session is not None else "READY"
        self.summary_var.set(f"{live}  ·  {engine}  ·  {source} → {target}  ·  {device}")

    def _on_source_change(self):
        self.reset_history()
        self._update_overlay_summary()

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
        self.text_generation += 1
        self.pipeline.reset()
        if self.live_session is not None:
            self.live_session.reset_context()
        # Remove queued old captions, but preserve lifecycle/errors (especially
        # the "stopped" event that re-enables the controls).
        retained = []
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            kind, values = event
            if kind == "live" and len(values) > 1 and values[1] in ("heard", "caption"):
                continue
            if kind == "caption" or kind == "heard":
                continue
            retained.append(event)
        for event in retained:
            self.events.put_nowait(event)
        self.source_text.config(text="Previous transcript cleared.")
        self.translated_text.config(text="Context cleared.")
        self.status_var.set("Context and pending older captions cleared.")

    def change_engine(self):
        if self.live_session is not None or (self.worker and self.worker.is_alive()):
            self.engine_box.set(self.engine_selected)
            self.status_var.set("Stop listening or finish the current translation before changing engines.")
            return
        next_engine = self.engine_box.get()
        try:
            new_pipeline = Pipeline(translator=create_translator(next_engine))
        except (ValueError, OSError) as exc:
            self.engine_box.set(self.engine_selected)
            self.status_var.set(f"Cannot select engine: {exc}")
            return
        self.text_generation += 1
        self.pipeline.reset()
        self.pipeline = new_pipeline
        self.engine_selected = next_engine
        self._update_overlay_summary()
        self.consent_given = False
        self.consent_provider = None
        self.source_text.config(text="Engine changed. Prior conversation cleared.")
        self.translated_text.config(text="Start a new authorized translation.")
        self.status_var.set(
            "Azure is context-free for baseline comparisons." if next_engine == "azure"
            else f"Engine set to {next_engine}. Context selection available.")

    def check_engine(self):
        """Check one provider with a public synthetic phrase, never meeting text."""
        if self.live_session is not None or (self.worker and self.worker.is_alive()):
            self.status_var.set("Stop the current task before checking the engine.")
            return
        engine = self.engine_box.get()
        if not key_available(engine):
            self.status_var.set(f"Configure the {engine} key in .env, then restart the app.")
            return
        def task():
            try:
                model = create_translator(engine)
                response = model.translate(TranslationRequest(
                    text="Hola", source="es", target="en", context_enabled=False))
                self.publish("engine_check", engine, response[:80])
            except Exception as exc:
                self.publish("engine_check", engine, f"ERROR: {exc}")
        self.status_var.set(f"Checking {engine} with a public example...")
        self.worker = threading.Thread(target=task, daemon=True)
        self.worker.start()

    def check_consent(self):
        engine = self.engine_box.get()
        if self.consent_given and self.consent_provider == engine:
            return True
        destination = (f"the {engine.upper()} cloud service" if needs_cloud(engine)
                       else "the local Ollama service on this computer")
        details = ("Azure receives only the current transcript and has no context support."
                   if engine == "azure" else
                   "Selected preceding utterances and glossary terms may also be sent when context is enabled.")
        permission = messagebox.askokcancel(
            "Audio & translation privacy",
            "Capture starts only when you click Start Listening. Whisper processes audio locally. "
            "When translation is required, text is sent to " + destination + ".\n\n"
            + details + "\n\n"
            "Do not use recordings or meeting content without the participants' permission; "
            "do not use confidential material unless this destination is approved.\n\n"
            "Continue?",
            parent=self.root,
        )
        if permission:
            self.consent_given = True
            self.consent_provider = engine
        return permission

    def refresh_sources(self):
        self.devices = sources()
        labels = [name for name, _device in self.devices]
        self.source_box["values"] = labels or ["No audio sources: check pactl"]
        self.source_box.current(0)
        if not labels:
            self.status_var.set("No audio source found; install pulseaudio-utils, FFmpeg and check PipeWire/PulseAudio.")
        self._update_overlay_summary()

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
        if not text or self.live_session is not None:
            return
        if self.source_lang() != self.target_lang() and not key_available(self.engine_box.get()):
            self.status_var.set(f"Configure the key for {self.engine_box.get()} in .env.")
            return
        if not self.check_consent():
            return
        language = self.source_lang()
        if language == "auto":
            self.status_var.set("For typed text, select the source language manually.")
            return
        target = self.target_lang()
        topic = self.topic_box.get()
        glossary = dict(self.glossary)
        contextual = self.context_enabled.get()
        generation = self.text_generation
        active_pipeline = self.pipeline
        def task():
            try:
                self.publish("status", "Translating typed text...")
                result = active_pipeline.translate(text, language, target,
                                                   topic, glossary, contextual)
                self.publish("typed_caption", generation, result.original, result.translated,
                             result.source_language, "Text")
            except Exception as exc:
                self.publish("typed_error", generation, str(exc))
        if self.worker and self.worker.is_alive():
            self.status_var.set("Busy; try after current translation completes.")
            return
        self.worker = threading.Thread(target=task, daemon=True)
        self.worker.start()

    def toggle_capture(self):
        if self.live_session is not None:
            self.live_session.stop()
            self.source_text.config(text="Audio capture stopping.")
            self.translated_text.config(text="Previous captions hidden.")
            self.start_button.config(text="STOPPING", state="disabled")
            self.status_var.set("Stopping. Waiting for current processing to finish.")
            return
        if not self.devices:
            self.status_var.set("No audio sources found. Try REFRESH AUDIO.")
            return
        if (self.source_lang() != self.target_lang()
                and not key_available(self.engine_box.get())):
            self.status_var.set(f"Configure the API key for {self.engine_box.get()} in .env.")
            return
        if self.worker and self.worker.is_alive():
            self.status_var.set("Finish the current typed translation before listening.")
            return
        if not self.check_consent():
            return
        index = self.source_box.current()
        if index < 0 or index >= len(self.devices):
            self.status_var.set("Select a valid audio source.")
            return
        try:
            duration = float(os.getenv("AUDIO_CHUNK_SECONDS", "4"))
            self.pipeline.reset()
            self.session_generation += 1
            generation = self.session_generation
            session = LiveSession(
                source=self.devices[index][1],
                source_language=self.source_lang(),
                target_language=self.target_lang(),
                topic=self.topic_box.get(),
                glossary=dict(self.glossary),
                contextual=self.context_enabled.get(),
                pipeline_factory=lambda: Pipeline(
                    translator=create_translator(self.engine_box.get())),
                chunk_seconds=duration,
                on_event=lambda kind, *data: self.publish(
                    "live", generation, kind, *data),
            )
        except (ValueError, TypeError, OSError) as exc:
            self.status_var.set(f"Cannot start audio session: {exc}")
            return
        self.live_session = session
        self._update_overlay_summary()
        self.start_button.config(text="STOP", state="normal")
        for selector in (self.source_box, self.source_picker,
                         self.target_picker, self.topic_box, self.engine_box):
            selector.config(state="disabled")
        session.start()

    def poll(self):
        try:
            while True:
                kind, values = self.events.get_nowait()
                if kind == "status":
                    self.status_var.set(values[0])
                elif kind == "heard":
                    self.source_text.config(text=values[0])
                    self.translated_text.config(text="Translating…")
                elif kind == "typed_caption":
                    generation, original, translated, lang, latency = values
                    if generation != self.text_generation:
                        continue
                    self.source_text.config(text=f"[{lang}]  {original[:700]}")
                    self.translated_text.config(text=translated[:900])
                    self.status_var.set(f"Translation ready · {latency}")
                elif kind == "typed_error":
                    if values[0] == self.text_generation:
                        self.status_var.set("ERROR · " + values[1])
                elif kind == "engine_check":
                    self.status_var.set(f"{values[0]} test: {values[1]}")
                elif kind == "error":
                    self.status_var.set("ERROR · " + values[0])
                elif kind == "live":
                    generation, event, *data = values
                    if generation != self.session_generation:
                        continue
                    if event == "status":
                        self.status_var.set(data[0])
                    elif event == "heard":
                        original, language = data
                        self.source_text.config(text=f"[{language}]  {original[:700]}")
                        self.translated_text.config(text="Translating...")
                    elif event == "caption":
                        original, translated, lang, latency, *details = data
                        self.source_text.config(text=f"[{lang}]  {original[:700]}")
                        self.translated_text.config(text=translated[:900])
                        self.status_var.set(
                            f"Translation ready · {latency:.1f}s since chunk ended"
                            + (f" · {details[0]}" if details else ""))
                    elif event == "backlog":
                        self.status_var.set(
                            f"Processing behind playback · skipped {data[0]} older chunk(s)")
                    elif event == "error":
                        self.status_var.set("ERROR · " + data[0])
                    elif event == "stopped":
                        self.live_session = None
                        self._update_overlay_summary()
                        self.pipeline.reset()
                        for selector in (self.source_box, self.source_picker,
                                         self.target_picker, self.topic_box, self.engine_box):
                            selector.config(state="readonly")
                        self.start_button.config(text="START", state="normal")
                        self.source_text.config(text="Listening stopped.")
                        self.translated_text.config(text="Start another authorized session to see captions.")
                        self.status_var.set(
                            f"Stopped. Session memory cleared. Skipped chunks: {data[0]}.")
        except queue.Empty:
            pass
        if not self.closed:
            self.root.after(100, self.poll)

    def close(self):
        self.closed = True
        if self.live_session is not None:
            self.live_session.stop()
        self.pipeline.reset()
        self.root.destroy()

def main():
    load_dotenv()
    root = tk.Tk()
    Application(root)
    root.mainloop()

if __name__ == "__main__":
    main()
