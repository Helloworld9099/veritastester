"""Offline UI-state checks; widget doubles do not verify Tk rendering."""
import queue
import threading
from unittest.mock import MagicMock

import pytest

from bridgecaption.app import Application
from bridgecaption import app as module


@pytest.fixture
def app():
    instance = Application.__new__(Application)
    instance.root = MagicMock()
    instance.pipeline = MagicMock()
    instance.events = queue.Queue()
    instance.live_session = None
    instance.session_generation = 1
    instance.text_generation = 0
    instance.last_caption = None
    instance.caption_popup = None
    instance.update_popup = None
    instance.update_package = None
    instance.update_preparing = False
    instance.update_cancel = threading.Event()
    instance.update_lock = threading.Lock()
    instance.closed = True
    instance.compact = True
    for name in ("source_text", "translated_text", "status_var", "summary_var",
                 "engine_note", "start_button", "source_box",
                 "source_picker", "target_picker", "topic_box"):
        setattr(instance, name, MagicMock())
    instance.engine_selected = "ollama"
    instance.source_picker.get.return_value = "Spanish [es]"
    instance.target_picker.get.return_value = "English [en]"
    instance.source_box.get.return_value = "System audio (current output)"
    return instance


@pytest.mark.parametrize("engine", ["ollama", "groq", "azure", "gemini"])
@pytest.mark.parametrize("listening", [False, True])
def test_summary_distinguishes_capture_state_and_text_destination(app, engine, listening):
    app.engine_selected = engine
    app.live_session = object() if listening else None
    app._update_overlay_summary()
    summary = app.summary_var.set.call_args.args[0]
    assert ("Listening" if listening else "Ready") in summary
    assert "Spanish → English" in summary
    assert engine.casefold() not in summary.casefold()
    note = app.engine_note.config.call_args.kwargs["text"]
    assert ("Local translation" if engine == "ollama" else "Cloud translation") in note
    app.pipeline.translate.assert_not_called()


def test_empty_full_caption_viewer_does_not_create_a_window(app, monkeypatch):
    window = MagicMock()
    monkeypatch.setattr(module.tk, "Toplevel", window)
    app.show_full_caption()
    window.assert_not_called()
    assert "completed translation" in app.status_var.set.call_args.args[0]


def test_full_caption_retains_unicode_and_closes_on_context_reset(app, monkeypatch):
    window, text = MagicMock(), MagicMock()
    monkeypatch.setattr(module.tk, "Toplevel", MagicMock(return_value=window))
    monkeypatch.setattr(module.tk, "Frame", MagicMock())
    monkeypatch.setattr(module.tk, "Text", MagicMock(return_value=text))
    monkeypatch.setattr(module.ttk, "Scrollbar", MagicMock())
    app.label = MagicMock()
    app.button = MagicMock()
    original, translated = "¿Cómo estás? " * 100, "私は元気です。 " * 200
    app._render_caption(original, translated, "es")
    assert len(app.translated_text.config.call_args.kwargs["text"]) <= 320
    app.show_full_caption()
    displayed = "".join(call.args[1] for call in text.insert.call_args_list)
    assert original in displayed and translated in displayed
    assert app.last_caption == (original, translated, "es")
    text.config.assert_called_with(state="disabled")
    app.pipeline.translate.assert_not_called()
    app.reset_history()
    window.destroy.assert_called_once()
    assert app.last_caption is None and app.caption_popup is None
    app.pipeline.reset.assert_called_once()


def test_stopped_session_dismisses_private_caption_snapshot(app):
    app.live_session = MagicMock()
    popup = MagicMock()
    app.caption_popup = popup
    app.last_caption = ("old speech", "old translation", "es")
    app.publish("live", 1, "stopped", 0)
    app.poll()
    popup.destroy.assert_called_once()
    assert app.caption_popup is None and app.last_caption is None
    assert app.live_session is None
    app.start_button.config.assert_called_with(text="Start listening", state="normal")
    assert "Ready" in app.summary_var.set.call_args.args[0]


def test_repeated_dismiss_is_safe(app):
    popup = MagicMock()
    app.caption_popup = popup
    app._dismiss_full_caption()
    app._dismiss_full_caption()
    popup.destroy.assert_called_once()
    assert app.caption_popup is None


def test_new_heard_event_dismisses_stale_snapshot(app):
    popup = MagicMock()
    app.caption_popup = popup
    app.publish("live", 1, "heard", "new speech", "es")
    app.poll()
    popup.destroy.assert_called_once()
    assert app.caption_popup is None
    assert app.last_caption is None


def test_old_generation_does_not_clear_current_snapshot(app):
    popup = MagicMock()
    app.caption_popup = popup
    app.last_caption = ("current", "current", "en")
    app.publish("live", 0, "heard", "stale speech", "es")
    app.poll()
    popup.destroy.assert_not_called()
    assert app.last_caption == ("current", "current", "en")


def test_close_dismisses_snapshot_without_translation(app):
    popup = MagicMock()
    app.caption_popup = popup
    app.close()
    popup.destroy.assert_called_once()
    app.root.destroy.assert_called_once()
    app.pipeline.translate.assert_not_called()


def test_update_controls_remain_in_expanded_settings(app):
    panels = ("controls_frame", "engine_controls_frame", "settings_frame", "updates_frame",
              "bottom_frame", "settings_workspace", "settings_canvas", "caption_panel",
              "top_frame", "title_row", "action_row", "original_heading", "translation_heading",
              "caption_divider", "summary_label", "status_label", "full_caption_button",
              "settings_button")
    for panel in panels:
        setattr(app, panel, MagicMock())
    app.root.winfo_screenwidth.return_value = 1200
    app.root.winfo_screenheight.return_value = 800
    app.root.winfo_ismapped.return_value = False
    app._resize_caption_wrap = MagicMock()

    app._apply_overlay_mode()
    app.toggle_settings()

    app.settings_workspace.pack.assert_called_once()
    app.updates_frame.pack_forget.assert_not_called()
