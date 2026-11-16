# Caption overlay UI review

The UI branch starts from main `4c45b8d734e52da9d10c567e58b32648b563589f`.
It changes the Tk interface and window dimensions only; translation providers,
audio capture, pipelines, trained artifacts, and model defaults are unchanged.
The optional Marian integration and parallel research branch are not included.

## Interface changes

- Consistent dark fields, mint accents, keyboard focus outlines, and caption hierarchy.
- Labeled audio source, spoken language, target language, and topic selectors.
- Context/glossary and opacity controls grouped beside translation-engine settings.
- Separate NOT LISTENING / LISTENING badge and local/cloud text destination note.
- Compact view: 760 × 320; settings view: 940 × 720. Start, settings, and pin
  stay at the top. Native window controls handle minimizing and closing.
- Typed-text guidance explains manual source-language selection and Enter.
- Full caption opens a scrollable, read-only snapshot of the entire last
  translation, including text beyond the overlay preview. It sends no request
  and saves no file. A new viewer replaces the previous one. Context clearing,
  engine changes, and capture-stop paths dismiss the snapshot; Escape closes it.

## Verified in this environment

`python -m compileall -q src tests` passed.

`python -m pytest -q`: **149 passed, 2 skipped**. One dependency deprecation
warning comes from the installed AnyIO pytest plugin, not the application.
The two skipped tests require a graphical desktop. Headless UI-state tests
cover capture-state and destination messaging for all four existing engines,
untruncated Unicode caption viewing without translation calls, empty-state
handling, and snapshot dismissal on context reset and session stop.

All existing non-layout application methods were compared with main using
Python ASTs: they are unchanged except for dismissing the UI snapshot when
an existing path clears its caption. `git diff --check` passed.

**No GUI rendering or visual inspection is claimed.** No display was available.
A temporary Xvfb binary was unpacked outside the repository, but this execution
environment disallows the local display sockets it requires. No audio was
captured, cloud provider called, model downloaded, model activated, or
application deployed during this UI verification.

## Desktop review before merge

On an Ubuntu desktop with the documented package dependencies installed:

```bash
python -m pytest -q tests/test_overlay.py tests/test_overlay_interface.py
python -m bridgecaption
```

Or use `xvfb-run -a python -m pytest -q tests/test_overlay.py` on a machine
where Xvfb is installed and display sockets are permitted. The optional layout
test checks control bounds at the compact/expanded defaults and minimum sizes.

Check the following with synthetic text and authorized audio only:

1. Expand/hide settings with the button and Ctrl+comma. Resize both views;
   ensure Start, Settings, Pin, caption viewer, and status remain visible.
2. Tab through fields and buttons; check focused and disabled controls, menu
   contrast, opacity on the actual compositor, and multilingual font fallback.
3. Select a source language for typed text. Review a long completed caption,
   open Full caption, scroll through it, then close it with Escape.
4. Reopen the viewer and clear context or change engines; the snapshot closes.
5. With permission, start/stop/restart listening; selectors lock while listening,
   the badge follows session state, and stopping dismisses the old snapshot.

Real speech accuracy, provider availability, compositor behavior, and native
window layout still require workstation validation. Passing offline tests does
not establish model quality or production readiness.
