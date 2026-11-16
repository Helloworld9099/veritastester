"""Window-only preferences for a lightweight live-caption overlay.

Keep geometry and opacity decisions outside audio/model code. Tk applies alpha to
the entire window (including text); it is not a click-through or secure overlay.
"""

COMPACT_SIZE = (760, 312)
EXPANDED_SIZE = (980, 690)
DEFAULT_OPACITY_PERCENT = 92
MIN_OPACITY_PERCENT = 82
MAX_OPACITY_PERCENT = 100


def opacity_percent(value):
    """Parse user settings defensively; always retain legible captions."""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return DEFAULT_OPACITY_PERCENT
    if number != number or number in (float("inf"), float("-inf")):
        return DEFAULT_OPACITY_PERCENT
    return round(max(MIN_OPACITY_PERCENT, min(MAX_OPACITY_PERCENT, number)))


def window_size(compact):
    return COMPACT_SIZE if compact else EXPANDED_SIZE


def apply_opacity(window, value):
    """Request opacity from the desktop compositor; report unavailable effects.

    X11/Wayland compositor policies vary. Unsupported transparency must not
    prevent the user from opening or operating the application.
    """
    requested = opacity_percent(value) / 100.0
    try:
        window.attributes("-alpha", requested)
        actual = float(window.attributes("-alpha"))
    except Exception:  # Tk may raise TclError or lack the window-manager feature.
        return False
    return abs(actual - requested) <= 0.03
