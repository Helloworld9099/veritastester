# Caption overlay UI review

## Integration on November 16, 2026

PR #6 now incorporates main `736980e26705516d364f2881c72648517041ffa1`
(including merged PRs #7 and #8). The newer main interface is preserved:
separate action row, bounded scrollable settings, screen-position clamping,
provider-output guards and readable diagnostics. Overlay dimensions and
translation/audio pipelines are unchanged from that main revision.

The remaining UI additions are deliberately narrow:
- Full caption opens a scrollable, read-only Unicode snapshot of the entire
  last result without another provider request or saved file.
- Reopening replaces the previous viewer. Close/Escape dismiss it.
- Context clearing, engine change, capture stop, new heard speech, and app
  close dismiss private snapshots. Stale-generation events are ignored.
- The engine section identifies local versus cloud transcript destination.
  Existing consent remains required; this note does not authorize sharing.

No optional Marian integration, research models, activation, deployment or
paid calls are included.

## Verification

At this revision, `python -m pytest -q` reports **156 passed, 1 skipped**.
Compilation and `git diff --check` pass. Tests cover all four provider labels,
Ready/Listening summary, complete Unicode snapshots, no extra translation,
context/session clearing, repeated dismissal, stale-generation protection,
and app close. Existing main diagnostic/provider tests are retained.

The skipped test requires a graphical desktop. No display or Xvfb is available
in this container; its package index cannot locate Xvfb. Thus native Tk layout,
compositor behavior, live audio and provider requests remain **unverified**.
Tests use local fixtures and synthetic text, not private audio or cloud calls.

## Desktop review before merge

```bash
python -m pytest -q tests/test_overlay.py tests/test_overlay_interface.py
python -m bridgecaption
```

1. Toggle settings repeatedly and resize compact/expanded windows. Ensure Start,
   Settings, Full caption, captions and status remain visible. Settings should
   scroll independently rather than hide the caption panel.
2. Check keyboard focus, disabled controls, multilingual font fallback and
   opacity on the actual compositor.
3. Use synthetic typed text; open a long result, scroll to its end, reopen it,
   and close through Escape and the window close control.
4. Open a snapshot then clear context/change engines. Verify it closes and the
   destination note matches the chosen engine.
5. With consent, start/stop/restart audio. Verify selectors lock during capture,
   Ready/Listening state changes and old snapshots disappear.

Offline test success does not establish real-speech/model quality or readiness.
