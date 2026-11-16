"""Small-window preferences plus an optional real Tk interaction smoke test.

Tk smoke test skips on headless environments. Run it with Xvfb on CI or on
Ubuntu's desktop to exercise controls without microphone/API credentials.
"""
import os

import pytest

from bridgecaption.overlay import (
    COMPACT_SIZE, EXPANDED_SIZE, apply_opacity, opacity_percent, window_size,
)


class FakeWindow:
    def __init__(self, supports_opacity=True):
        self.alpha = 1.0
        self.supports_opacity = supports_opacity

    def attributes(self, name, value=None):
        assert name == "-alpha"
        if value is not None and self.supports_opacity:
            self.alpha = value
        return self.alpha


def test_compact_is_smaller_than_expanded():
    assert window_size(True) == COMPACT_SIZE
    assert window_size(False) == EXPANDED_SIZE
    assert COMPACT_SIZE[0] < EXPANDED_SIZE[0]
    assert COMPACT_SIZE[1] < EXPANDED_SIZE[1]


@pytest.mark.parametrize("raw,expected", [
    ("92", 92), ("75", 82), (130, 100), ("bad", 92),
    (float("nan"), 92), (float("inf"), 92),
])
def test_readable_opacity_settings(raw, expected):
    assert opacity_percent(raw) == expected


def test_unsupported_transparency_does_not_break_window():
    assert apply_opacity(FakeWindow(), 92)
    assert not apply_opacity(FakeWindow(supports_opacity=False), 92)


def test_real_tk_expands_settings_without_hiding_start_stop(monkeypatch):
    if not os.environ.get("DISPLAY"):
        pytest.skip("requires a graphical desktop or xvfb-run")
    import tkinter as tk
    import bridgecaption.app as application
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk display unavailable")
    try:
        monkeypatch.setattr(application, "sources", lambda: [])
        app = application.Application(root)
        root.update_idletasks()
        assert app.compact is True
        assert app.controls_frame.winfo_manager() == ""
        assert app.bottom_frame.winfo_manager() == ""
        assert app.caption_panel.winfo_manager() == "pack"
        assert app.start_button.winfo_manager() == "pack"
        # Compact preview is bounded; full sentence remains available on expansion.
        app._render_caption("x" * 400, "y" * 650, "es")
        assert len(app.translated_text.cget("text")) < 260
        app.toggle_settings()
        root.update_idletasks()
        assert app.compact is False
        assert app.translated_text.cget("text") == "y" * 650
        assert app.source_box.winfo_manager() == "pack"
        assert app.source_picker.winfo_manager() == "pack"
        assert app.target_picker.winfo_manager() == "pack"
        assert app.engine_box.winfo_manager() == "pack"
        assert app.topic_box.winfo_manager() == "pack"
        assert app.test_text.winfo_manager() == "pack"
        assert app.source_text.cget("wraplength") > 500
        assert app.settings_button.cget("text") == "Hide settings"
        app.status_var.set("ERROR · provider unavailable")
        assert app.status_label.cget("fg") == application.ALERT
        assert app.controls_frame.winfo_manager() == "pack"
        assert app.engine_controls_frame.winfo_manager() == "pack"
        assert app.bottom_frame.winfo_manager() == "pack"
        assert app.start_button.winfo_manager() == "pack"
        app.toggle_settings()
        root.update_idletasks()
        assert app.compact is True
        assert app.source_text.winfo_manager() == "pack"
        assert app.start_button.winfo_manager() == "pack"
        app.reset_history()
        assert app.last_caption is None
        app.toggle_settings()
        assert "y" * 20 not in app.translated_text.cget("text")
    finally:
        root.destroy()
