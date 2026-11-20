"""Visible draggable translation overlay for authorized Linux audio.

The application starts no recording until START LISTENING is clicked.
The selected translator receives recognized text; microphone/system audio
remains on the local computer.
"""
import os
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox

from dotenv import load_dotenv

from .updates import (find_update, current_version, download_update, discard_update,
                      installed_app, install_update_on_exit)
from .audio import sources
from .session import LiveSession
from .context import parse_glossary
from .pipeline import Pipeline
from .provider import TranslationRequest
from .providers import key_available, needs_cloud, create_translator
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
BG = "#F7F3EA"
PANEL = "#FFFFFF"
CONTROL = "#EEE8DD"
BORDER = "#D6D0C5"
TEXT = "#174C52"
MUTED = "#45686B"
ACCENT = "#C66B4E"
ALERT = "#A53B2B"

class Application:
    def __init__(self, root):
        self.root = root
        self.root.title("BridgeCaption")
        self.root.geometry("760x282+220+340")
        self.root.minsize(560, 245)
        self.root.configure(bg=BG)
        self.root.attributes("-topmost", True)
        self.root.overrideredirect(False)
        self.worker = None
        self.text_generation = 0
        self.events = queue.Queue()
        self.engine_selected = os.getenv("BRIDGE_TRANSLATOR", "gemini").strip().lower()
        self.pipeline = Pipeline(translator=create_translator(self.engine_selected))
        # Typed-text history is never shared with a live listening session.
        self.live_session = None
        self.session_generation = 0
        self.consent_given = False
        self.consent_provider = None
        self.closed = False
        self.update_checking = False
        self.update_popup = None
        self.update_downloading = False
        self.update_preparing = False
        self.update_package = None
        self.update_cancel = threading.Event()
        self.update_lock = threading.Lock()
        self.context_enabled = tk.BooleanVar(value=True)
        self.glossary = {}
        self.devices = []
        self.last_caption = None
        self._dismiss_full_caption()
        self._drag_xy = None
        self.is_pinned = True
        self.compact = True
        self.capture_started_at = None
        self.opacity_value = tk.DoubleVar(value=opacity_percent(
            os.getenv("BRIDGE_OVERLAY_OPACITY", "92")))
        self._ui()
        self._apply_overlay_mode()
        self._apply_opacity(self.opacity_value.get(), notify=False)
        self.refresh_sources()
        self.root.bind("<Configure>", self._resize_caption_wrap)
        self.root.bind("<Control-comma>", lambda _event: self.toggle_settings())
        self.root.after(100, self.poll)
        self.root.after(5000, self._scheduled_update_check)
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
        foreground = "#FFFFFF" if accent else TEXT
        hover = "#AE553C" if accent else "#E2D9CB"
        return tk.Button(
            parent, text=text, command=command, relief="flat", bd=0,
            bg=background, fg=foreground, activebackground=hover,
            activeforeground="#FFFFFF" if accent else TEXT,
            disabledforeground="#765348" if accent else MUTED,
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
        # Two rows keep primary actions from being squeezed by window buttons.
        top = tk.Frame(self.root, bg=BG, padx=12, pady=5)
        self.top_frame = top
        top.pack(fill="x")
        title_row = tk.Frame(top, bg=BG)
        self.title_row = title_row
        title_row.pack(fill="x")
        title = self.label(title_row, "BridgeCaption", 15, TEXT, True)
        title.pack(side="left")
        self.button(title_row, "Close", self.close).pack(side="right", padx=(6, 0))
        self.button(title_row, "Minimize", self.root.iconify).pack(side="right", padx=(6, 0))
        self.pin_button = self.button(title_row, "Pinned", self.toggle_pin)
        self.pin_button.pack(side="right", padx=(6, 0))
        action_row = tk.Frame(top, bg=BG)
        self.action_row = action_row
        action_row.pack(fill="x", pady=(5, 0))
        self.start_button = self.button(
            action_row, "Start listening", self.toggle_capture, accent=True)
        self.start_button.pack(side="left", padx=(0, 8))
        self.settings_button = self.button(action_row, "Settings", self.toggle_settings)
        self.settings_button.pack(side="left")
        self.full_caption_button = self.button(action_row, "Full caption", self.show_full_caption)
        self.full_caption_button.pack(side="left", padx=(8, 0))
        for draggable in (top, title_row, title):
            draggable.bind("<ButtonPress-1>", self.drag_start)
            draggable.bind("<B1-Motion>", self.drag_move)

        self.summary_var = tk.StringVar(value="Ready  ·  Open Settings to choose languages")
        self.summary_label = tk.Label(
            self.root, textvariable=self.summary_var, bg=BG, fg=MUTED,
            anchor="w", font=("DejaVu Sans", 10), padx=18, pady=4,
        )
        self.summary_label.pack(fill="x")
        self.live_indicator = self.label(title_row, "", 10, MUTED)
        self.live_indicator.pack(side="left", padx=(14, 0))

        # Scroll configuration independently so captions retain usable height.
        self.settings_workspace = tk.Frame(self.root, bg=BG)
        self.settings_canvas = tk.Canvas(
            self.settings_workspace, bg=BG, bd=0,
            highlightthickness=0, height=220)
        self.settings_scrollbar = tk.Scrollbar(
            self.settings_workspace, orient="vertical",
            command=self.settings_canvas.yview, bg=CONTROL,
            troughcolor=BG, activebackground=ACCENT)
        self.settings_scrollbar.pack(side="right", fill="y")
        self.settings_canvas.pack(side="left", fill="both", expand=True)
        self.settings_canvas.configure(yscrollcommand=self.settings_scrollbar.set)
        self.settings_content = tk.Frame(self.settings_canvas, bg=BG)
        self.settings_window = self.settings_canvas.create_window(
            (0, 0), window=self.settings_content, anchor="nw")
        self.settings_content.bind(
            "<Configure>", lambda _e: self.settings_canvas.configure(
                scrollregion=self.settings_canvas.bbox("all")))
        self.settings_canvas.bind(
            "<Configure>", lambda e: self.settings_canvas.itemconfigure(
                self.settings_window, width=e.width))
        self.settings_canvas.bind("<MouseWheel>", self._scroll_settings)
        self.settings_canvas.bind("<Button-4>", self._scroll_settings)
        self.settings_canvas.bind("<Button-5>", self._scroll_settings)

        self.controls_frame = tk.Frame(self.settings_content, bg=BG, padx=18, pady=7)
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

        self.engine_controls_frame = tk.Frame(self.settings_content, bg=BG, padx=18, pady=7)
        self.engine_controls_frame.pack(fill="x")
        self.label(self.engine_controls_frame, "Translation", 11, TEXT, True).pack(
            anchor="w", pady=(0, 8))
        engine_row = tk.Frame(self.engine_controls_frame, bg=BG)
        engine_row.pack(fill="x")
        engine_row.grid_columnconfigure(2, weight=2)
        self.button(engine_row, "Check connection", self.check_engine).grid(
            row=0, column=1, sticky="s", padx=(0, 14))
        topic_group = field(engine_row, "Topic", 2, last=True)
        self.topic_box = self.picker(topic_group, TOPICS, width=18)
        self.topic_box.pack(fill="x")
        self.engine_note = self.label(self.engine_controls_frame, "", 9, MUTED)
        self.engine_note.configure(justify="left", wraplength=650)
        self.engine_note.pack(fill="x", pady=(8, 0))
        self.updates_frame = tk.Frame(self.settings_content, bg=BG, padx=18, pady=7)
        self.updates_frame.pack(fill="x")
        self.label(self.updates_frame, "Application updates", 11, TEXT, True).pack(anchor="w", pady=(0, 8))
        self.label(self.updates_frame, f"Current version: v{current_version()}", 9, MUTED).pack(side="left", padx=(0, 12))
        self.update_button = self.button(self.updates_frame, "Check for updates", lambda: self.check_updates(manual=True))
        self.update_button.pack(side="left")
        self.settings_frame = tk.Frame(self.settings_content, bg=BG, padx=18, pady=7)
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
        opacity_group = tk.Frame(options_row, bg=BG)
        opacity_group.pack(side="right")
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

        self.caption_panel = tk.Frame(self.root, bg=PANEL, padx=18, pady=12)
        self.caption_panel.pack(fill="both", expand=True, padx=18, pady=(6, 4))
        self.original_heading = self.label(self.caption_panel, "Original", 9, MUTED)
        self.original_heading.pack(anchor="w")
        self.source_text = self.label(
            self.caption_panel, "Your transcript will appear here.", 12, MUTED)
        self.source_text.configure(wraplength=650, justify="left")
        self.source_text.pack(fill="x", pady=(7, 8))
        self.caption_divider = tk.Frame(self.caption_panel, bg=BORDER, height=1)
        self.caption_divider.pack(fill="x", pady=(0, 10))
        self.translation_heading = self.label(self.caption_panel, "Translation", 9, MUTED)
        self.translation_heading.pack(anchor="w")
        self.translated_text = self.label(
            self.caption_panel, "Waiting for speech or a text translation.", 20,
            TEXT, True)
        self.translated_text.configure(wraplength=650, justify="left")
        self.translated_text.pack(fill="both", expand=True, pady=(8, 0))

        self.bottom_frame = tk.Frame(self.settings_content, bg=BG, padx=18, pady=5)
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
            value="Choose an audio source and languages in Settings.")
        self.status_label = tk.Label(
            self.root, textvariable=self.status_var, bg=BG, fg=MUTED,
            anchor="w", justify="left", wraplength=700, padx=18, pady=7,
            font=("DejaVu Sans", 10),
        )
        self.status_label.pack(fill="x", side="bottom", before=self.caption_panel)
        self.status_var.trace_add("write", self._style_status)
        self._bind_settings_scroll(self.settings_content)

    def _bind_settings_scroll(self, widget):
        """Wheel-scroll configuration without capturing wheel events globally."""
        for child in widget.winfo_children():
            if not isinstance(child, ttk.Combobox):
                for event in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                    child.bind(event, self._scroll_settings, add="+")
            self._bind_settings_scroll(child)

    def _scroll_settings(self, event):
        if self.compact:
            return
        number = getattr(event, "num", None)
        if number == 4:
            direction = -1
        elif number == 5:
            direction = 1
        else:
            delta = getattr(event, "delta", 0)
            direction = -1 if delta > 0 else 1 if delta < 0 else 0
        if direction:
            self.settings_canvas.yview_scroll(3 * direction, "units")
            return "break"

    def _style_status(self, *_args):
        """Make errors identifiable without relying on status text alone."""
        message = self.status_var.get().casefold()
        error = message.startswith("error") or message.startswith("no audio")
        self.status_label.config(fg=ALERT if error else MUTED)

    def _dismiss_full_caption(self):
        popup = getattr(self, "caption_popup", None)
        if popup is not None:
            popup.destroy()
        self.caption_popup = None

    def show_full_caption(self):
        """Read the complete last result without another translation request."""
        if self.last_caption is None:
            self.status_var.set("A completed translation will be available here after listening or a text test.")
            return
        original, translated, language = self.last_caption
        self._dismiss_full_caption()
        popup = tk.Toplevel(self.root)
        self.caption_popup = popup
        popup.transient(self.root)
        popup.title("Full caption | BridgeCaption")
        popup.geometry("720x480")
        popup.minsize(440, 300)
        popup.configure(bg=BG)
        self.label(popup, "Full caption", 15, ACCENT, True).pack(anchor="w", padx=16, pady=(16, 4))
        self.label(popup, "Snapshot of the last completed translation", 9, MUTED).pack(anchor="w", padx=16)
        body = tk.Frame(popup, bg=PANEL)
        body.pack(fill="both", expand=True, padx=16, pady=12)
        scrollbar = ttk.Scrollbar(body)
        scrollbar.pack(side="right", fill="y")
        text = tk.Text(body, wrap="word", bg=PANEL, fg=TEXT, relief="flat",
                       font=("DejaVu Sans", 12), padx=14, pady=14,
                       yscrollcommand=scrollbar.set, highlightthickness=1,
                       highlightbackground=BORDER, highlightcolor=ACCENT)
        text.pack(fill="both", expand=True)
        scrollbar.config(command=text.yview)
        text.tag_configure("heading", foreground=ACCENT, font=("DejaVu Sans", 9, "bold"))
        text.insert("end", f"ORIGINAL · {language.upper()}\n\n", "heading")
        text.insert("end", original + "\n\n")
        text.insert("end", "TRANSLATION\n\n", "heading")
        text.insert("end", translated)
        text.config(state="disabled")
        self.button(popup, "CLOSE", self._dismiss_full_caption).pack(anchor="e", padx=16, pady=(0, 12))
        popup.protocol("WM_DELETE_WINDOW", self._dismiss_full_caption)
        popup.bind("<Escape>", lambda _event: self._dismiss_full_caption())
        text.focus_set()

    def _apply_overlay_mode(self):
        """Switch between a caption-first floating overlay and full controls."""
        for panel in (self.controls_frame, self.engine_controls_frame,
                      self.settings_frame, self.updates_frame, self.bottom_frame):
            panel.pack_forget()
        self.settings_workspace.pack_forget()
        if not self.compact:
            for panel in (self.controls_frame, self.engine_controls_frame,
                          self.settings_frame, self.bottom_frame):
                panel.pack(fill="x")
            self.settings_workspace.pack(fill="x", before=self.caption_panel)
            self.settings_canvas.yview_moveto(0)

        dark = "#202327"
        surface = dark if self.compact else BG
        caption_surface = dark if self.compact else PANEL
        primary = "#F6F4F0" if self.compact else TEXT
        secondary = "#B8BABD" if self.compact else MUTED
        self.root.configure(bg=surface)
        for widget in (self.top_frame, self.title_row, self.action_row):
            widget.configure(bg=surface)
        for widget in self.title_row.winfo_children():
            if isinstance(widget, tk.Label):
                widget.configure(bg=surface, fg=primary)
        self.caption_panel.configure(bg=caption_surface)
        for widget in (self.original_heading, self.translation_heading,
                       self.source_text, self.translated_text):
            widget.configure(bg=caption_surface)
        self.original_heading.configure(fg=secondary)
        self.translation_heading.configure(fg=secondary)
        self.source_text.configure(fg=secondary,
                                   font=("DejaVu Sans", 10 if self.compact else 13))
        self.translated_text.configure(fg=primary,
                                       font=("DejaVu Sans", 16 if self.compact else 21))
        self.caption_divider.configure(bg="#42464A" if self.compact else BORDER)
        self.summary_label.pack_forget()
        self.status_label.pack_forget()
        if not self.compact:
            self.summary_label.pack(fill="x", before=self.caption_panel)
            self.status_label.pack(fill="x", side="bottom", before=self.caption_panel)
        # Tk widgets cannot be repacked into a different parent. Keep the
        # capture controls in their original row and hide secondary actions.
        self.action_row.pack_forget()
        self.action_row.pack(fill="x", pady=(2, 0), after=self.title_row)
        self.full_caption_button.pack_forget()
        self.settings_button.config(text="Expand" if self.compact else "Hide settings")
        if self.compact:
            self.settings_button.pack_forget()
            self.settings_button.pack(side="left", padx=(0, 8))
        else:
            self.settings_button.pack_forget()
            self.settings_button.pack(side="left")
            self.full_caption_button.pack(side="left", padx=(8, 0))
        width, height = window_size(self.compact)
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        minimum_width = 420 if self.compact else 780
        minimum_height = 180 if self.compact else min(575, max(430, sh - 96))
        width = min(width, max(minimum_width, sw - 36))
        height = min(height, max(minimum_height, sh - 72))
        if not self.compact:
            self.settings_canvas.config(height=min(220, max(140, height // 3)))
        self.root.minsize(minimum_width, minimum_height)
        if self.root.winfo_ismapped():
            x, y = self.root.winfo_x(), self.root.winfo_y()
        else:
            x, y = (sw - width) // 2, max(24, sh // 5)
        x = max(0, min(x, max(0, sw - width - 24)))
        y = max(0, min(y, max(0, sh - height - 48)))
        self.root.geometry(f"{width}x{height}+{x}+{y}")
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
        self.engine_note.config(wraplength=width)

    def _render_caption(self, original, translated, lang):
        """Display bounded subtitles without changing the underlying translation.

        Compact mode intentionally shows a preview. Expanding Settings can
        reveal more of the last caption without another provider/API call.
        """
        self.last_caption = (original, translated, lang)
        original_limit = 200 if self.compact else 700
        translated_limit = 320 if self.compact else 900

        def clipped(text, limit):
            return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"

        self.source_text.config(text=clipped(original, original_limit))
        self.translated_text.config(text=clipped(translated, translated_limit))

    def _update_overlay_summary(self):
        if not hasattr(self, "summary_var"):
            return
        engine = self.engine_selected.capitalize()
        source = LANGUAGES.get(self.source_lang(), self.source_lang())
        target = LANGUAGES.get(self.target_lang(), self.target_lang())
        live = "Listening" if self.live_session is not None else "Ready"
        self.summary_var.set(f"{live}  ·  {source} → {target}")
        self.engine_note.config(text=(
            "Cloud translation · recognized text is sent to the configured service."
            if needs_cloud(self.engine_selected) else
            "Local translation · recognized text is sent to your Ollama service."))

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
        self._dismiss_full_caption()
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

    def check_engine(self):
        """Check one provider with a public synthetic phrase, never meeting text."""
        if self.live_session is not None or (self.worker and self.worker.is_alive()):
            self.status_var.set("Stop the current task before checking the engine.")
            return
        engine = self.engine_selected
        if not key_available(engine):
            self.status_var.set(f"Configure the {engine} key in .env, then restart the app.")
            return
        def task():
            try:
                model = create_translator(engine)
                response = model.translate(TranslationRequest(
                    text="Hola", source="es", target="en", context_enabled=False))
                from .providers import translation_text
                self.publish("engine_check", engine, translation_text(response))
            except Exception as exc:
                self.publish("engine_check", engine, f"ERROR: {exc}")
        self.status_var.set(f"Checking {engine} with a public example...")
        self.worker = threading.Thread(target=task, daemon=True)
        self.worker.start()

    def check_consent(self):
        engine = self.engine_selected
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
        if self.source_lang() != self.target_lang() and not key_available(self.engine_selected):
            self.status_var.set(f"Configure the key for {self.engine_selected} in .env.")
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
            self._dismiss_full_caption()
            self.source_text.config(text="Audio capture stopping.")
            self.translated_text.config(text="Previous captions hidden.")
            self.start_button.config(text="Stopping...", state="disabled")
            self.status_var.set("Stopping. Waiting for current processing to finish.")
            return
        if not self.devices:
            self.status_var.set("No audio sources found. Try REFRESH AUDIO.")
            return
        if (self.source_lang() != self.target_lang()
                and not key_available(self.engine_selected)):
            self.status_var.set(f"Configure the API key for {self.engine_selected} in .env.")
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
                    translator=create_translator(self.engine_selected)),
                chunk_seconds=duration,
                on_event=lambda kind, *data: self.publish(
                    "live", generation, kind, *data),
            )
        except (ValueError, TypeError, OSError) as exc:
            self.status_var.set(f"Cannot start audio session: {exc}")
            return
        self.live_session = session
        self.capture_started_at = time.monotonic()
        self._update_overlay_summary()
        self.start_button.config(text="Stop listening", state="normal")
        for selector in (self.source_box, self.source_picker,
                         self.target_picker, self.topic_box):
            selector.config(state="disabled")
        session.start()

    def _scheduled_update_check(self):
        if not self.closed:
            self.check_updates()
            self.root.after(24 * 60 * 60 * 1000, self._scheduled_update_check)

    def check_updates(self, manual=False):
        """Check off the UI thread; private repositories may need a public update feed."""
        if self.closed or self.update_checking or self.update_downloading or self.update_preparing:
            return
        self.update_checking = True
        self.update_button.config(state="disabled")
        if manual:
            self.status_var.set("Checking for updates...")
        def task():
            try:
                self.publish("update_result", manual, find_update(), None)
            except Exception as exc:
                self.publish("update_result", manual, None, str(exc))
        threading.Thread(target=task, daemon=True).start()

    def _show_update(self, update):
        if self.update_popup is not None and self.update_popup.winfo_exists():
            return
        popup = tk.Toplevel(self.root)
        self.update_popup = popup
        popup.title("BridgeCaption update")
        popup.configure(bg=BG)
        popup.geometry("390x180")
        popup.resizable(False, False)
        popup.transient(self.root)
        self.label(popup, f"Version {update.version} is available", 13, TEXT, True).pack(
            padx=20, pady=(22, 14))
        self.label(popup, "Install update?", 11, MUTED).pack(padx=20)
        actions = tk.Frame(popup, bg=BG)
        actions.pack(pady=18)
        self.button(actions, "No", popup.destroy).pack(side="left", padx=8)
        self.button(actions, "Yes", lambda: self._install_update(update)).pack(side="left", padx=8)

    def _dismiss_update(self):
        with self.update_lock:
            self.update_cancel.set()
            if self.update_package is not None:
                discard_update(self.update_package)
                self.update_package = None
        if self.update_popup is not None and self.update_popup.winfo_exists():
            self.update_popup.destroy()
        self.update_popup = None

    def _install_update(self, update):
        if self.update_downloading or self.update_preparing:
            return
        try:
            installed_app()
        except (OSError, RuntimeError) as exc:
            messagebox.showerror("Installer unavailable", str(exc), parent=self.update_popup or self.root)
            return
        self.update_downloading = True
        self.update_cancel.clear()
        popup = self.update_popup
        popup.protocol("WM_DELETE_WINDOW", self._dismiss_update)
        for widget in popup.winfo_children():
            widget.destroy()
        self.label(popup, f"Installing v{update.version}", 13, TEXT, True).pack(padx=20, pady=15)
        self.update_progress = tk.DoubleVar(value=0)
        ttk.Progressbar(popup, maximum=100, variable=self.update_progress).pack(fill="x", padx=20)
        self.update_progress_label = self.label(popup, "Downloading 0%", 10, MUTED)
        self.update_progress_label.pack(pady=10)
        self.button(popup, "Cancel", self._dismiss_update).pack()

        def task():
            try:
                package = download_update(update, lambda done, total:
                    self.publish("update_progress", done, total), self.update_cancel.is_set)
                with self.update_lock:
                    if self.update_cancel.is_set():
                        discard_update(package)
                    else:
                        self.update_package = package
                self.publish("update_downloaded")
            except Exception as exc:
                self.publish("update_failed", str(exc))
        threading.Thread(target=task, daemon=True).start()

    def _finish_update(self):
        if self.update_package is None or self.update_preparing:
            return
        if self.live_session is not None or (self.worker and self.worker.is_alive()):
            messagebox.showinfo("Update waiting", "Stop captioning before installing.", parent=self.root)
            return
        self.update_preparing = True
        self.start_button.config(state="disabled")
        self.update_popup.protocol("WM_DELETE_WINDOW", lambda: None)
        self.update_progress_label.config(text="Preparing update. BridgeCaption will restart.")
        for widget in self.update_popup.winfo_children():
            if isinstance(widget, (tk.Button, ttk.Button)):
                widget.config(state="disabled")
        package = self.update_package

        def task():
            try:
                install_update_on_exit(package)
                self.publish("update_ready")
            except Exception as exc:
                self.publish("update_failed", str(exc))
        threading.Thread(target=task, daemon=True).start()

    def poll(self):
        try:
            while True:
                kind, values = self.events.get_nowait()
                if kind == "update_progress":
                    if self.update_popup is not None:
                        received, total = values
                        percentage = int(100 * received / total)
                        self.update_progress.set(percentage)
                        self.update_progress_label.config(text=f"Downloading {percentage}%")
                elif kind == "update_downloaded":
                    self.update_downloading = False
                    if self.update_package is None or self.update_popup is None:
                        continue
                    if self.live_session is not None or (self.worker and self.worker.is_alive()):
                        self.update_progress_label.config(text="Verified. Stop captioning to install.")
                        self.button(self.update_popup, "Install when idle", self._finish_update).pack(pady=5)
                    else:
                        self._finish_update()
                elif kind == "update_ready":
                    self.update_preparing = False
                    self.update_package = None  # The installer now owns the verified file.
                    self.close()
                    return
                elif kind == "update_failed":
                    self.update_downloading = False
                    if self.update_preparing:
                        self.update_preparing = False
                        self.start_button.config(state="normal")
                    if not self.update_cancel.is_set():
                        self._dismiss_update()
                        messagebox.showerror("Update failed", values[0], parent=self.root)
                elif kind == "update_result":
                    manual, update, error = values
                    self.update_checking = False
                    self.update_button.config(state="normal")
                    if error:
                        if manual:
                            self.status_var.set("Update check unavailable: " + error)
                    elif update:
                        self._show_update(update)
                    elif manual:
                        self.status_var.set("BridgeCaption is up to date.")
                elif kind == "status":
                    self.status_var.set(values[0])
                elif kind == "heard":
                    self.last_caption = None
                    self._dismiss_full_caption()
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
                    engine, result = values
                    if result.startswith("ERROR:"):
                        self.status_var.set(f"ERROR · {engine.capitalize()} connection check failed: {result[6:]}")
                    else:
                        self.status_var.set(
                            f"{engine.capitalize()} is working · Test translation: {result[:90]}")
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
                        self._dismiss_full_caption()
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
                        self._dismiss_full_caption()
                        self.live_session = None
                        self.capture_started_at = None
                        self._update_overlay_summary()
                        self.pipeline.reset()
                        for selector in (self.source_box, self.source_picker,
                                         self.target_picker, self.topic_box):
                            selector.config(state="readonly")
                        self.start_button.config(text="Start listening", state="normal")
                        self.source_text.config(text="Listening stopped.")
                        self.translated_text.config(text="Start another authorized session to see captions.")
                        self.status_var.set(
                            f"Stopped. Session memory cleared. Skipped chunks: {data[0]}.")
        except queue.Empty:
            pass
        if not self.closed:
            if self.capture_started_at is not None:
                elapsed = int(time.monotonic() - self.capture_started_at)
                self.live_indicator.config(text=f"●  {elapsed // 60}:{elapsed % 60:02d}")
            else:
                self.live_indicator.config(text="")
            self.root.after(100, self.poll)

    def close(self):
        if self.update_preparing:
            return
        self.closed = True
        self._dismiss_full_caption()
        self._dismiss_update()
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
