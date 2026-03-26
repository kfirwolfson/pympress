# Pympress — Copilot Instructions

Pympress is a dual-screen PDF presentation tool (like PowerPoint Presenter View) built with **GTK 3** (PyGObject) and **Poppler**. Licensed GPLv2+.

## Quick Reference

| Action | Command |
|---|---|
| Install (dev) | `pip install -e .` |
| Install deps only | `pip install -r requirements.txt` |
| Lint | `flake8 . --count --show-source --statistics --select=E,F,W,C` |
| Lint docstrings | `flake8 . --select=D` |
| Build docs | `sphinx-build -b html docs build/sphinx/html` |
| Extract i18n | `pybabel extract -F pympress/share/locale/babel_mapping.cfg -o pympress/share/locale/pympress.pot .` |
| Compile translations | `pybabel compile -d pympress/share/locale/ -D pympress` |
| Run | `python -m pympress [file.pdf]` |

There is **no automated test suite** — linting (flake8) is the primary CI check.

## Architecture

Component-based, event-driven design around a central `UI` object. Not strict MVC, but clear separation:

- **Model**: `Document` / `Page` in [pympress/document.py](../pympress/document.py) — pure PDF data, no GTK imports except rendering
- **View**: Glade XML files in `pympress/share/xml/` define widget trees, loaded by `Builder`
- **Controller**: `UI` (in [pympress/ui.py](../pympress/ui.py)) owns all sub-components and handles events

### Key modules

| Module | Role |
|---|---|
| `ui.py` | Main `UI(Builder)` class — creates Content + Presenter windows, wires everything |
| `document.py` | `Document`, `Page`, `PdfPage` — GUI-independent PDF model via Poppler |
| `builder.py` | `Builder(Gtk.Builder)` — loads Glade XML, auto-translates strings, introspectively binds widgets to object attributes by matching IDs to `None`-valued class attributes |
| `config.py` | `Config(ConfigParser)` — `defaults.conf` → user config → CLI overrides. Layouts stored as JSON within INI |
| `extras.py` | Auxiliary features: `Media`, `TimingReport`, `Annotations`, `Zoom`, `Cursor`, `FileWatcher` |
| `scribble.py` | Freehand drawing/annotation overlay on slides |
| `pointer.py` | Software laser pointer |
| `talk_time.py` | Elapsed/remaining time with color-coded warnings |
| `editable_label.py` | Click-to-edit labels (page number, estimated talk time) |
| `surfacecache.py` | Thread-safe LRU cache of rendered `cairo.ImageSurface` pages |
| `util.py` | Platform detection (`IS_WINDOWS`/`IS_MAC_OS`/`IS_POSIX`), resource paths, icon/CSS loading |
| `__main__.py` | Entry point — CLI parsing (`getopt`), locale/gettext setup, `Gtk.main()` |

### Media overlays (plugin system)

`pympress/media_overlays/` provides a backend abstraction for video/animation playback:

- **Base**: `VideoOverlay(Builder)` in `base.py` — abstract interface (`do_play`, `do_stop`, etc.), overlay positioning, progress bar
- **GIF**: `gif_backend.py` — `GdkPixbuf.PixbufAnimation` frame-by-frame rendering
- **GStreamer**: `gst_backend.py` — `GstPlayer.Player` for full video
- **VLC**: `vlc_backend.py` — `vlc.MediaPlayer` for full video
- Backends are lazily registered by mime type in `Media._setup_backends()` (`extras.py`). Priority: GIF → GStreamer → VLC (last becomes default catch-all)
- Each media element creates **two** overlay instances (content window muted + presenter window with audio)

### Wiring pattern

`UI.__init__()` creates sub-components → calls `Builder.load_ui(name)` for each Glade file → `connect_signals(self)` resolves signal handlers via dot-path notation (e.g. `doc.goto_page` resolves `self.doc.goto_page`).

## Code Conventions

- **Style**: PEP 8, 120-char line limit, 4-space indent — see `[style]` and `[flake8]` in [setup.cfg](../setup.cfg)
- **Docstrings**: Google style with types in backticks (e.g. `` `str` ``), no type annotations in signatures
- **Naming**: `snake_case` functions/variables, `PascalCase` classes, `_`/`__` prefix for private
- **Imports**: standard lib → `gi.repository` → `pympress` modules. `from __future__ import print_function, unicode_literals` in every file (Python 2/3 compat)
- **gettext**: `_()` builtin installed by `gettext.install()`. All user-facing strings wrapped in `_()`
- **Widget binding**: Declare `attribute = None` on the class, give the Glade widget the same `id` — `Builder` auto-assigns it
- **Flake8 config**: Some rules are deliberately ignored (alignment flexibility) — see `[flake8]` in setup.cfg before "fixing" style warnings

## Internationalization

- Translations managed via [POEditor](https://poeditor.com/join/project/nKfRxeN8pS) — do not edit `.po` files by hand for existing languages
- Extraction: `babel` + `babelgladeextractor` from both Python and Glade sources
- `.mo` compiled at install time. Supported locales: cs, de, es, fr, pl
- `Builder.__translate_widget_strings()` applies `_()` to all string properties of loaded widgets

## Glade UI files (`pympress/share/xml/`)

| File | Purpose |
|---|---|
| `presenter.glade` | Presenter window (current/next slide, notes, annotations, timer, menus) |
| `content.glade` | Audience window (fullscreen slide) |
| `media_overlay.glade` | Video overlay widget with progress bar and controls |
| `shortcuts.glade` | Keyboard shortcuts help window |
| `time_report_dialog.glade` | Per-section timing breakdown dialog |

## Dependencies (non-pip, system-level)

PyGObject, PyCairo, GTK 3, Cairo, Poppler (with GObject introspection bindings). Optionally VLC. See [README.md](../README.md#dependencies) for platform-specific install commands.

## Common Pitfalls

- GTK operations must happen on the main thread; `GLib.idle_add()` for cross-thread UI updates
- `Builder` widget auto-binding only works if the class attribute is set to `None` before `load_ui()`
- Config values may come as strings from INI — use `getboolean()`, `getint()`, etc.
- Platform-specific code paths (especially Windows) — check `util.IS_WINDOWS` and test accordingly
- No type hints in the codebase — don't add them unless specifically asked
