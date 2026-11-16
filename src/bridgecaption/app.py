"""Visible draggable translation overlay for authorized Linux audio.

The application starts no recording until START LISTENING is clicked.
The selected translator receives recognized text; microphone/system audio
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
BG = "#101923"
PANEL = "#1c2b38"
CONTROL = "#273b49"
BORDER = "#3c5565"
TEXT = "#f4f8fb"
MUTED = "#bed0dd"
ACCENT = "#5de1b3"
ALERT = "#ffb7a8"

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
        self.last_caption = None
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
        # On Tk/X11, alpha set before the window is mapped may silently be lost.
        self.root.after(180, lambda: self._apply_opacity(
            self.opacity_value.get(), notify=False))
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def label(self, parent, text, size=11, color=TEXT, bold=False):
        return tk.Label(parent, text=text, bg=parent.cget("bg"), fg=color,
                        font=("DejaVu Sans", size, "bold" if bold else "normal"),
                        anchor="w")

    def button(self, parent, text, command, accent=False):
        background = ACCENT if accent else CONTROL
        foreground = BG if accent else TEXT
        hover = "#87f0ca" if accent else "#395362"
        return tk.Button(
            parent, text=text, command=command, relief="flat", bd=0,
            bg=background, fg=foreground, activebackground=hover,
            activeforeground=BG if accent else TEXT,
            disabledforeground="#35544e" if accent else MUTED,
            font=("DejaVu Sans", 10, "bold"),
            padx=13, pady=8, cursor="hand2", takefocus=True,
            highlightthickness=1, highlightbackground=BG,
            highlightcolor=ACCENT,
        )

    def picker(self, parent, values, width=15):
        box = ttk.Combobox(parent, state="readonly", values=values,
                           width=width, style="Bridge.TCombobox")
        box.current(0)
        return box

    def _configure_styles(self):
        """Keep Tk and ttk controls legible on light or dark Linux themes."""
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure(
            "Bridge.TCombobox", font=("DejaVu Sans", 11), padding=6,
            foreground=TEXT, fieldbackground=CONTROL, background=CONTROL,
            selectforeground=TEXT, selectbackground=CONTROL,
            bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER,
            arrowcolor=TEXT,
        )
        style.map(
            "Bridge.TCombobox",
            fieldbackground=[("disabled", PANEL), ("readonly", CONTROL)],
            foreground=[("disabled", MUTED), ("readonly", TEXT)],
            selectforeground=[("readonly", TEXT)],
            selectbackground=[("readonly", CONTROL)],
        )
        style.configure("Bridge.Horizontal.TScale", background=BG,
                        troughcolor=CONTROL, sliderlength=18)

    def _ui(self):
        self._configure_styles()

        # Caption controls stay visible when Settings is collapsed.
        top = tk.Frame(self.root, bg=BG, padx=18, pady=10)
        top.pack(fill="x")
        self.label(top, "BridgeCaption", 15, TEXT, True).pack(side="left")
        self.button(top, "Close", self.close).pack(side="right", padx=(6, 0))
        self.button(top, "Minimize", self.root.iconify).pack(side="right", padx=(6, 0))
        self.pin_button = self.button(top, "Pinned", self.toggle_pin)
        self.pin_button.pack(side="right", padx=(6, 0))
        self.settings_button = self.button(top, "Settings", self.toggle_settings)
        self.settings_button.pack(side="right", padx=(6, 0))
        self.start_button = self.button(top, "Start listening", self.toggle_capture, accent=True)
        self.start_button.pack(side="right", padx=(6, 0))
        top.bind("<ButtonPress-1>", self.drag_start)
        top.bind("<B1-Motion>", self.drag_move)

        self.summary_var = tk.StringVar(value="Ready  ·  Open Settings to choose languages")
        self.summary_label = tk.Label(
            self.root, textvariable=self.summary_var, bg=BG, fg=MUTED,
            anchor="w", font=("DejaVu Sans", 10), padx=18, pady=4,
        )
        self.summary_label.pack(fill="x")

        self.controls_frame = tk.Frame(self.root, bg=BG, padx=18, pady=7)
        self.controls_frame.pack(fill="x")
        self.label(self.controls_frame, "Audio & languages", 11, TEXT, True).pack(
            anchor="w", pady=(0, 8))
        language_row = tk.Frame(self.controls_frame, bg=BG)
        language_row.pack(fill="x")
        for column, weight in ((0, 3), (1, 2), (2, 2)):
            language_row.grid_columnconfigure(column, weight=weight)

        def field(parent, title, column, last=False):
            group = tk.Frame(parent, bg=BG)
            group.grid(row=0, column=column, sticky="ew",
                       padx=(0, 0 if last else 14))
            self.label(group, title, 9, MUTED, True).pack(
                anchor="w", pady=(0, 5))
            return group

        audio_group = field(language_row, "Audio source", 0)
        self.source_box = self.picker(audio_group, ["No audio sources"], width=29)
        self.source_box.pack(fill="x")
        self.source_box.bind("<<ComboboxSelected>>",
                             lambda _event: self._on_source_change())

        self.source_labels = sorted(LANGUAGES.items(), key=lambda kv: kv[1])
        from_group = field(language_row, "From", 1)
        self.source_picker = self.picker(
            from_group, [f"{name} [{code}]" for code, name in self.source_labels], 18)
        self.source_picker.pack(fill="x")
        self.target_labels = [item for item in self.source_labels if item[0] != "auto"]
        to_group = field(language_row, "To", 2, last=True)
        self.target_picker = self.picker(
            to_group, [f"{name} [{code}]" for code, name in self.target_labels], 18)
        self.target_picker.pack(fill="x")
        self.source_picker.set("Auto detect [auto]")
        self.target_picker.set("English [en]")
        self.source_picker.bind("<<ComboboxSelected>>",
                                lambda _event: self._on_source_change())
        self.target_picker.bind("<<ComboboxSelected>>",
                                lambda _event: self._on_source_change())

        self.engine_controls_frame = tk.Frame(self.root, bg=BG, padx=18, pady=7)
        self.engine_controls_frame.pack(fill="x")
        self.label(self.engine_controls_frame, "Translation", 11, TEXT, True).pack(
            anchor="w", pady=(0, 8))
        engine_row = tk.Frame(self.engine_controls_frame, bg=BG)
        engine_row.pack(fill="x")
        engine_row.grid_columnconfigure(0, weight=2)
        engine_row.grid_columnconfigure(2, weight=2)
        engine_group = field(engine_row, "Engine", 0)
        self.engine_box = self.picker(engine_group, PROVIDERS, width=20)
        self.engine_box.set(self.engine_selected)
        self.engine_box.pack(fill="x")
        self.engine_box.bind("<<ComboboxSelected>>",
                             lambda _event: self.change_engine())
        self.button(engine_row, "Check connection", self.check_engine).grid(
            row=0, column=1, sticky="s", padx=(0, 14))
        topic_group = field(engine_row, "Topic", 2, last=True)
        self.topic_box = self.picker(topic_group, TOPICS, width=18)
        self.topic_box.pack(fill="x")
        self.label(
            self.engine_controls_frame,
            "Cloud engines receive text only after you approve the destination.",
            9, MUTED,
        ).pack(anchor="w", pady=(7, 0))

        self.settings_frame = tk.Frame(self.root, bg=BG, padx=18, pady=7)
        self.settings_frame.pack(fill="x")
        self.label(self.settings_frame, "Preferences", 11, TEXT, True).pack(
            anchor="w", pady=(0, 8))
        options_row = tk.Frame(self.settings_frame, bg=BG)
        options_row.pack(fill="x")
        tk.Checkbutton(
            options_row, text="Context & slang", variable=self.context_enabled,
            bg=BG, fg=TEXT, selectcolor=PANEL, activebackground=BG,
            activeforeground=TEXT, font=("DejaVu Sans", 10), takefocus=True,
            highlightcolor=ACCENT,
        ).pack(side="left")
        self.button(options_row, "Glossary", self.edit_glossary).pack(
            side="left", padx=(10, 0))
        self.button(options_row, "Clear context", self.reset_history).pack(
            side="left", padx=(8, 0))
        self.button(options_row, "Refresh audio", self.refresh_sources).pack(
            side="left", padx=(8, 0))
        opacity_group = tk.Frame(self.settings_frame, bg=BG)
        opacity_group.pack(anchor="w", pady=(9, 0))
        self.label(opacity_group, "Opacity", 9, MUTED).pack(
            side="left", padx=(0, 9))
        ttk.Scale(
            opacity_group, from_=MIN_OPACITY_PERCENT, to=MAX_OPACITY_PERCENT,
            orient="horizontal", length=94, variable=self.opacity_value,
            command=self._on_opacity_change, style="Bridge.Horizontal.TScale",
        ).pack(side="left", padx=(0, 8))
        self.opacity_label = tk.StringVar(
            value=f"{opacity_percent(self.opacity_value.get())}%")
        tk.Label(
            opacity_group, textvariable=self.opacity_label, bg=BG, fg=TEXT,
            font=("DejaVu Sans", 10),
        ).pack(side="left")

        self.caption_panel = tk.Frame(self.root, bg=PANEL, padx=20, pady=14)
        self.caption_panel.pack(fill="both", expand=True, padx=18, pady=(8, 6))
        self.label(self.caption_panel, "Original speech", 10, MUTED, True).pack(
            anchor="w")
        self.source_text = self.label(
            self.caption_panel, "Your transcript will appear here.", 12, MUTED)
        self.source_text.configure(wraplength=650, justify="left")
        self.source_text.pack(fill="x", pady=(7, 8))
        tk.Frame(self.caption_panel, bg=BORDER, height=1).pack(
            fill="x", pady=(0, 10))
        self.label(self.caption_panel, "Translation", 10, ACCENT, True).pack(
            anchor="w")
        self.translated_text = self.label(
            self.caption_panel, "Waiting for speech or a text translation.", 20,
            TEXT, True)
        self.translated_text.configure(wraplength=650, justify="left")
        self.translated_text.pack(fill="both", expand=True, pady=(8, 0))

        self.bottom_frame = tk.Frame(self.root, bg=BG, padx=18, pady=5)
        self.bottom_frame.pack(fill="x")
        self.label(self.bottom_frame, "Try a sentence", 10, MUTED, True).pack(
            anchor="w", pady=(0, 6))
        text_row = tk.Frame(self.bottom_frame, bg=BG)
        text_row.pack(fill="x")
        self.test_text = tk.Entry(
            text_row, bg=CONTROL, fg=TEXT, insertbackground=TEXT,
            relief="flat", font=("DejaVu Sans", 12),
            highlightthickness=1, highlightbackground=BORDER,
            highlightcolor=ACCENT,
        )
        self.test_text.pack(side="left", fill="x", expand=True, ipady=9,
                            padx=(0, 10))
        self.test_text.insert(0, "Ese código está chido")
        self.test_text.bind("<Return>", lambda _event: self.test_translate())
        self.button(text_row, "Translate text", self.test_translate, True).pack(
            side="right")

        self.status_var = tk.StringVar(
            value="Choose an audio source and translation engine in Settings.")
        self.status_label = tk.Label(
            self.root, textvariable=self.status_var, bg=BG, fg=MUTED,
            anchor="w", justify="left", wraplength=700, padx=18, pady=7,
            font=("DejaVu Sans", 10),
        )
        self.status_label.pack(fill="x")
        self.status_var.trace_add("write", self._style_status)

    def _style_status(self, *_args):
        """Make errors identifiable without relying on status text alone."""
        message = self.status_var.get().casefold()
        error = message.startswith("error") or message.startswith("no audio")
        self.status_label.config(fg=ALERT if error else MUTED)

    def _apply_overlay_mode(self):
        """Collapse configuration without losing captions or Start/Stop controls."""
        for panel in (self.controls_frame, self.engine_controls_frame,
                      self.settings_frame, self.bottom_frame):
            panel.pack_forget()
        if not self.compact:
            for panel in (self.controls_frame, self.engine_controls_frame,
                          self.settings_frame):
                panel.pack(fill="x", before=self.caption_panel)
            self.bottom_frame.pack(fill="x", before=self.status_label)
        self.settings_button.config(
            text="Settings" if self.compact else "Hide settings")
        self.source_text.config(
            font=("DejaVu Sans", 12 if self.compact else 13))
        self.translated_text.config(
            font=("DejaVu Sans", 18 if self.compact else 21, "bold"))
        width, height = window_size(self.compact)
        self.root.minsize(700 if self.compact else 780,
                          278 if self.compact else 575)
        self.root.geometry(f"{width}x{height}")
        self._resize_caption_wrap()
        if self.last_caption is not None:
            self._render_caption(*self.last_caption)

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
        width = max(280, self.root.winfo_width() - 110)
        self.source_text.config(wraplength=width)
        self.translated_text.config(wraplength=width)
        self.status_label.config(wraplength=max(280, self.root.winfo_width() - 52))

    def _render_caption(self, original, translated, lang):
        """Display bounded subtitles without changing the underlying translation.

        Compact mode intentionally shows a preview. Expanding Settings can
        reveal more of the last caption without another provider/API call.
        """
        self.last_caption = (original, translated, lang)
        original_limit = 90 if self.compact else 700
        translated_limit = 125 if self.compact else 900

        def clipped(text, limit):
            return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"

        self.source_text.config(text=f"[{lang}]  {clipped(original, original_limit)}")
        self.translated_text.config(text=clipped(translated, translated_limit))

    def _update_overlay_summary(self):
        if not hasattr(self, "summary_var"):
            return
        engine = self.engine_box.get().capitalize()
        source = LANGUAGES.get(self.source_lang(), self.source_lang())
        target = LANGUAGES.get(self.target_lang(), self.target_lang())
        live = "Listening" if self.live_session is not None else "Ready"
        self.summary_var.set(f"{live}  ·  {source} → {target}  ·  {engine}")

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
        self.pin_button.config(text="Pinned" if self.is_pinned else "Unpinned")

    def reset_history(self):
        self.last_caption = None
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
        self.last_caption = None
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
            self.last_caption = None
            self.source_text.config(text="Audio capture stopping.")
            self.translated_text.config(text="Previous captions hidden.")
            self.start_button.config(text="Stopping...", state="disabled")
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
        self.start_button.config(text="Stop listening", state="normal")
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
                    self.last_caption = None
                    self.source_text.config(text=values[0])
                    self.translated_text.config(text="Translating…")
                elif kind == "typed_caption":
                    generation, original, translated, lang, latency = values
                    if generation != self.text_generation:
                        continue
                    self._render_caption(original, translated, lang)
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
                        self.last_caption = None
                        self.source_text.config(text=f"[{language}]  {original[:700]}")
                        self.translated_text.config(text="Translating...")
                    elif event == "caption":
                        original, translated, lang, latency, *details = data
                        self._render_caption(original, translated, lang)
                        self.status_var.set(
                            f"Translation ready · {latency:.1f}s since chunk ended"
                            + (f" · {details[0]}" if details else ""))
                    elif event == "backlog":
                        self.status_var.set(
                            f"Processing behind playback · skipped {data[0]} older chunk(s)")
                    elif event == "error":
                        self.status_var.set("ERROR · " + data[0])
                    elif event == "stopped":
                        self.last_caption = None
                        self.live_session = None
                        self._update_overlay_summary()
                        self.pipeline.reset()
                        for selector in (self.source_box, self.source_picker,
                                         self.target_picker, self.topic_box, self.engine_box):
                            selector.config(state="readonly")
                        self.start_button.config(text="Start listening", state="normal")
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
