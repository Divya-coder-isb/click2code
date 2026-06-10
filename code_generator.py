"""Core agent logic for the MockScreen-to-Code Generator.

This module is the brain of the project. It accepts 1–3 PIL mockup images and
a target tech stack ("Streamlit", "React", or "Flask"), sends them to
Google Gemini 1.5 Flash (vision-capable), and returns a structured dict with
all the generated files plus a short explanation.

Public API
----------
- ``generate_code_from_mockscreen(images, tech_stack, ...)`` → dict
- ``create_zip_download(files)`` → bytes

Both functions are designed to be called directly from a Streamlit app.
"""

from __future__ import annotations

import io
import os
import re
import zipfile
from typing import Optional

import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image

load_dotenv()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "gemini-1.5-flash"
"""Gemini model used for vision + code generation."""

SUPPORTED_TECH_STACKS: tuple[str, ...] = ("Streamlit", "React", "Flask")
"""Tech stacks the agent knows how to scaffold."""

_API_KEY_PLACEHOLDER = "your_gemini_api_key_here"


def _configure_gemini(api_key: Optional[str] = None) -> None:
    """Configure the ``google-generativeai`` client.

    The key is resolved in this order:

    1. Explicit ``api_key`` argument (used when a user pastes a key into the
       Streamlit sidebar).
    2. ``GEMINI_API_KEY`` from the environment (loaded from ``.env`` at module
       import via ``python-dotenv``).

    Raises:
        ValueError: If no usable API key can be found.
    """
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key or key.strip() in ("", _API_KEY_PLACEHOLDER):
        raise ValueError(
            "GEMINI_API_KEY is not set. Add it to your .env file or pass it "
            "explicitly via the sidebar."
        )
    genai.configure(api_key=key)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_BASE = (
    "You are an expert UI/UX engineer and senior full-stack developer. "
    "You will be shown one or more screenshots or mockups of a user "
    "interface. Your job is to recreate that UI as production-quality "
    "code that runs end-to-end after a single install step. "
    "Pay close attention to exact layout, hierarchy, spacing, alignment, "
    "typography, color palette, icons, and interactive states. "
    "Write idiomatic, well-organized, accessible code — never skeletons "
    "or TODO stubs."
)

STREAMLIT_PROMPT = """\
TARGET STACK: Streamlit (Python)

Files to emit (at minimum):
  - app.py            full Streamlit application; the only required Python file
  - requirements.txt  streamlit plus every other library you actually import
  - README.md         one-paragraph install + run instructions

Streamlit implementation rules:
- Start with `st.set_page_config(...)` (title, icon, layout).
- Use `import streamlit as st` and any of: pandas, numpy, plotly, altair, etc.
  Add every import to `requirements.txt`.
- Map the mockup to Streamlit widgets faithfully:
    * Layout: `st.columns`, `st.container`, `st.tabs`, `st.expander`,
      `st.sidebar`.
    * Typography: `st.title`, `st.header`, `st.subheader`, `st.markdown`,
      `st.caption`.
    * Inputs: `st.button`, `st.text_input`, `st.text_area`, `st.selectbox`,
      `st.multiselect`, `st.checkbox`, `st.radio`, `st.slider`,
      `st.date_input`, `st.file_uploader`, `st.form`.
    * Data / visuals: `st.metric`, `st.dataframe`, `st.table`, `st.image`,
      `st.line_chart`, `st.bar_chart`, `st.area_chart`, `st.progress`.
- Include realistic placeholder data so the rendered page looks like the
  mockup (lists of dicts, hand-built pandas DataFrames, JSON literals).
  Never leave widgets dangling without data.
- Wire up genuine interactivity using `st.session_state` (button clicks,
  form submissions, tab/screen switches).
- If the default theme cannot match the mockup's palette, inject a small
  CSS block with `st.markdown(..., unsafe_allow_html=True)` near the top
  of `app.py`.
- If multiple mockup images are provided, model each as a separate
  `st.tabs` tab OR a separate sidebar-navigated page (use `st.session_state`
  to remember the active page).
"""

REACT_PROMPT = """\
TARGET STACK: React (JavaScript, functional components + hooks)

Project layout to emit (Vite-style; runs with `npm install && npm run dev`):
  - package.json
  - vite.config.js
  - index.html                       includes the Tailwind CDN script
  - src/main.jsx                     mounts <App /> via ReactDOM.createRoot
  - src/App.jsx                      top-level component / lightweight router
  - src/components/<Screen>.jsx      one per mockup screen when >1 image
  - src/index.css                    minimal global resets (optional)
  - README.md

React implementation rules:
- React 18 only. Functional components and hooks (`useState`, `useEffect`,
  `useMemo`, `useReducer`) — no class components.
- Style with **Tailwind CSS via CDN**: include
  `<script src="https://cdn.tailwindcss.com"></script>` in `index.html` so
  the project runs without a PostCSS build step. Use Tailwind utility
  classes to match the mockup's spacing, colors, radii, shadows, and
  typography precisely. `style={{ ... }}` inline styles are fine for
  one-off values Tailwind cannot express.
- Use semantic elements (`<header>`, `<nav>`, `<main>`, `<section>`,
  `<button>`) with appropriate `aria-*` / `alt` attributes.
- Populate components with realistic placeholder data declared inline
  (arrays of objects). Wire up the interactions visible in the mockup —
  buttons, inputs, tabs, modals, toggles.
- Multi-image input → break the UI into per-screen components under
  `src/components/`, then switch between them from `App.jsx` using
  `useState` (no need for react-router unless the mockup clearly demands
  URL-driven routing).
- Do NOT depend on external image hosts or remote fonts. Use inline SVG,
  emoji, and Tailwind-only visuals.
- `package.json` must specify:
    "dependencies":   { "react": "^18.3.1", "react-dom": "^18.3.1" }
    "devDependencies":{ "vite": "^5.4.0", "@vitejs/plugin-react": "^4.3.1" }
    "scripts":        { "dev": "vite", "build": "vite build",
                        "preview": "vite preview" }
"""

FLASK_PROMPT = """\
TARGET STACK: Flask (Python + HTML + CSS, with Bootstrap 5 via CDN)

Project layout to emit:
  - app.py                       Flask app: imports, routes, __main__ runner
  - templates/base.html          shared layout w/ Bootstrap 5 CDN + nav
  - templates/index.html         extends base.html, recreates main mockup
  - templates/<screen>.html      one per extra mockup image
  - static/styles.css            custom CSS layered on Bootstrap
  - requirements.txt             flask + anything else imported
  - README.md

Flask implementation rules:
- Use Flask 3.x:
    from flask import Flask, render_template, request, url_for
    app = Flask(__name__)
    @app.route("/")
    def index(): ...
    if __name__ == "__main__":
        app.run(debug=True)
- Load Bootstrap 5 from the CDN inside `base.html` (link the CSS in <head>
  and the bundle script before </body>). Use Bootstrap utility classes and
  components (`navbar`, `card`, `btn`, `container`, `row`, `col`, `form-*`,
  `modal`, `alert`, ...) as the primary styling layer.
- Use `static/styles.css` ONLY for things Bootstrap cannot express
  (custom palette, gradients, exact spacing, Google Font import). Load it
  with `<link href="{{ url_for('static', filename='styles.css') }}" rel="stylesheet">`.
- Use Jinja2 template inheritance: `base.html` defines `{% block title %}`,
  `{% block content %}`, `{% block extra_css %}`, `{% block extra_js %}`,
  and every page template `{% extends "base.html" %}`s it.
- Each route should pass realistic placeholder data into the template via
  `render_template("...", items=[...], user={...})`. Render that data with
  Jinja `{% for %}` loops so the page looks alive.
- Multi-image input → one route + one template per screen, plus matching
  nav links in `base.html`.
- Use semantic HTML5 and `alt` / `aria` attributes throughout.
- `requirements.txt` must include at least `flask>=3.0`.
"""

_STACK_PROMPTS: dict[str, str] = {
    "Streamlit": STREAMLIT_PROMPT,
    "React": REACT_PROMPT,
    "Flask": FLASK_PROMPT,
}

_PRIMARY_LANG: dict[str, str] = {
    "Streamlit": "python",
    "React": "jsx",
    "Flask": "python",
}

_FALLBACK_EXT: dict[str, str] = {"python": "py", "jsx": "jsx"}

# Matches a single   ===FILE: path===\n<body>\n===END===   block.
_FILE_BLOCK_RE = re.compile(
    r"=== *FILE: *(?P<path>[^=\n]+?) *===\s*\n"
    r"(?P<body>.*?)\n"
    r"=== *END *===",
    re.DOTALL,
)

# Matches the trailing   ===EXPLANATION===\n<body>\n===END===   block.
_EXPLANATION_RE = re.compile(
    r"=== *EXPLANATION *===\s*\n(?P<body>.*?)\n=== *END *===",
    re.DOTALL,
)


def _build_prompt(
    tech_stack: str,
    num_images: int,
    extra_instructions: str = "",
) -> str:
    """Assemble the final prompt that gets sent to Gemini.

    Combines the global quality bar, image-count framing, the stack-specific
    rules, and the strict ``===FILE:===`` output format.
    """
    if tech_stack not in _STACK_PROMPTS:
        raise ValueError(
            f"Unsupported tech_stack: {tech_stack!r}. "
            f"Choose one of {SUPPORTED_TECH_STACKS}."
        )

    labels = [
        "primary screen / main landing view",
        "secondary screen / alternate page or state",
        "third screen / alternate page or state",
    ]
    image_lines = "\n".join(
        f"  - Image {i + 1}: {labels[i]}" for i in range(num_images)
    )

    if num_images == 1:
        multi_screen_directive = (
            "There is ONE mockup image. Generate a complete, polished UI for "
            "that single screen — every visible element should be implemented, "
            "wired up, and populated with realistic placeholder data."
        )
    else:
        multi_screen_directive = (
            f"There are {num_images} mockup images. Treat each one as a "
            "SEPARATE screen / page / state of the same app. Generate "
            "components/routes/templates so every screen is reachable from a "
            "single running app, and wire up navigation between them "
            "(tabs, sidebar links, router state, or a nav bar — whichever "
            "best matches the mockups)."
        )

    prompt = f"""{SYSTEM_PROMPT_BASE}

You were given {num_images} mockup image(s):
{image_lines}

{multi_screen_directive}

{_STACK_PROMPTS[tech_stack]}

GLOBAL QUALITY BAR (applies to every file you emit):
- The generated project MUST run end-to-end after the documented install
  command. No missing imports, no undefined references.
- No `TODO`, `FIXME`, `pass`, `// TODO`, or "implement this later" stubs.
  Fill in real code.
- Realistic placeholder data (not lorem ipsum stubs) so the rendered UI
  resembles the mockup.
- Accessible markup: semantic elements, labels for inputs, alt text for
  images, sensible `aria-*` attributes.
- Modern, idiomatic style (Python 3.10+, ES2022).
- README.md must show the exact install + run commands.

==========================================================================
OUTPUT FORMAT — follow this EXACTLY. No markdown fences. No extra prose.
==========================================================================

For every file in the project, emit one block of this form:

===FILE: <relative/path/to/file.ext>===
<full file contents — raw, no surrounding ``` fences>
===END===

After ALL the file blocks, emit ONE explanation block:

===EXPLANATION===
<2–4 sentence summary: which screens map to which files, and how to run
the project>
===END===

The very first character of your response MUST be `=` (the start of the
first ``===FILE:`` block). Do not write any prose before, between, or
after the blocks.
"""

    if extra_instructions.strip():
        prompt += (
            "\nAdditional user requirements — apply these on top of the rules "
            f"above:\n{extra_instructions.strip()}\n"
        )

    return prompt


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_code_from_mockscreen(
    images: list,
    tech_stack: str,
    extra_instructions: str = "",
    api_key: Optional[str] = None,
) -> dict:
    """Generate a runnable project from one or more UI mockup images.

    Args:
        images: List of 1 to 3 ``PIL.Image.Image`` instances representing the
            uploaded mockup screens. Image 1 is required; images 2 and 3 are
            optional and, when present, are treated as additional screens of
            the same app.
        tech_stack: Target tech stack — one of ``"Streamlit"``, ``"React"``,
            or ``"Flask"`` (see :data:`SUPPORTED_TECH_STACKS`).
        extra_instructions: Optional free-form text appended to the prompt
            (e.g. "use a dark palette", "make it responsive"). Defaults to
            an empty string.
        api_key: Optional override for ``GEMINI_API_KEY``. When omitted, the
            key is read from the environment (loaded from ``.env``).

    Returns:
        A dict with four keys, always present:

        - ``"code"`` *(str)*: All generated files concatenated into a single
          string with ``# ===== <filename> =====`` separators — handy for a
          quick preview.
        - ``"explanation"`` *(str)*: 2–4 sentence summary the model wrote
          about what it produced. Empty string if the model omitted it.
        - ``"files"`` *(list[dict])*: One entry per emitted file, each shaped
          ``{"filename": <relative/path>, "content": <text>}``. Suitable for
          rendering in Streamlit tabs and for :func:`create_zip_download`.
        - ``"error"`` *(str | None)*: ``None`` on success, otherwise a
          human-readable error message describing what went wrong (invalid
          input, rate limit, empty response, etc.).

    The function never raises — every failure is captured in
    ``result["error"]`` so callers can render it directly in the UI.
    """
    result: dict = {
        "code": "",
        "explanation": "",
        "files": [],
        "error": None,
    }

    try:
        if not images:
            raise ValueError("At least one mockup image is required.")
        if len(images) > 3:
            raise ValueError("At most 3 mockup images are supported.")
        for idx, img in enumerate(images, start=1):
            if not isinstance(img, Image.Image):
                raise ValueError(
                    f"Image {idx} is not a valid PIL Image instance "
                    f"(got {type(img).__name__})."
                )

        _configure_gemini(api_key)

        prompt = _build_prompt(
            tech_stack=tech_stack,
            num_images=len(images),
            extra_instructions=extra_instructions,
        )

        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(
            [prompt, *images],
            generation_config={
                "temperature": 0.2,
                "max_output_tokens": 8192,
            },
        )

        raw_text = _extract_response_text(response)
        if not raw_text:
            raise RuntimeError(
                "Gemini returned an empty response. Try uploading a clearer "
                "mockup or rerunning the request."
            )

        files = _parse_files(raw_text)
        explanation = _parse_explanation(raw_text)

        if not files:
            ext = _FALLBACK_EXT.get(_PRIMARY_LANG[tech_stack], "txt")
            files = [
                {
                    "filename": f"generated_output.{ext}",
                    "content": raw_text,
                }
            ]

        result["files"] = files
        result["code"] = _concatenate_for_preview(files)
        result["explanation"] = explanation or (
            f"Generated a {tech_stack} project with {len(files)} file(s) "
            f"from {len(images)} mockup image(s)."
        )

    except ValueError as exc:
        result["error"] = f"Invalid input: {exc}"
    except Exception as exc:  # noqa: BLE001 - capture SDK / network errors
        message = str(exc) or exc.__class__.__name__
        lowered = message.lower()
        if "rate" in lowered or "429" in lowered or "quota" in lowered:
            result["error"] = (
                "Gemini API rate limit / quota exceeded. Wait a few seconds "
                "and try again, or upgrade your quota. "
                f"(details: {message})"
            )
        elif "image" in lowered and ("format" in lowered or "decode" in lowered):
            result["error"] = (
                "Invalid image format. Please upload a PNG or JPG screenshot. "
                f"(details: {message})"
            )
        else:
            result["error"] = f"Code generation failed: {message}"

    return result


def create_zip_download(
    files: list,
    project_name: str = "mockscreen_output",
) -> bytes:
    """Pack a list of ``{filename, content}`` dicts into a ZIP archive.

    Args:
        files: List of dicts shaped ``{"filename": str, "content": str}`` —
            typically the ``"files"`` field returned by
            :func:`generate_code_from_mockscreen`.
        project_name: Name of the top-level folder inside the ZIP. Defaults
            to ``"mockscreen_output"`` so unzipping produces a tidy
            ``mockscreen_output/...`` directory.

    Returns:
        The ZIP archive as raw ``bytes``, ready to be handed to
        ``st.download_button(data=...)``.

    Raises:
        ValueError: If ``files`` is empty or any entry is not a dict.
    """
    if not files:
        raise ValueError("Cannot create a ZIP archive from an empty file list.")

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for entry in files:
            if not isinstance(entry, dict):
                raise ValueError(
                    "Each file entry must be a dict with "
                    f"'filename' and 'content'; got {type(entry).__name__}."
                )
            filename = (entry.get("filename") or "").strip().lstrip("/")
            content = entry.get("content", "")
            if not filename:
                continue
            zf.writestr(f"{project_name}/{filename}", content)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_response_text(response) -> str:
    """Pull text out of a Gemini response and surface useful errors.

    ``response.text`` raises if the candidate was blocked by safety filters
    or otherwise unusable; we re-raise with the ``prompt_feedback`` payload
    included so the caller can show something actionable in the UI.
    """
    try:
        text = response.text
    except Exception as exc:  # noqa: BLE001 - re-raised with extra context
        feedback = getattr(response, "prompt_feedback", None)
        raise RuntimeError(
            "Gemini did not return a usable text response "
            f"(prompt_feedback={feedback!r}). Underlying error: {exc}"
        ) from exc
    return (text or "").strip()


def _parse_files(raw_text: str) -> list[dict]:
    """Extract every ``===FILE: ... ===END===`` block from the model output."""
    files: list[dict] = []
    seen: set[str] = set()
    for match in _FILE_BLOCK_RE.finditer(raw_text):
        path = match.group("path").strip().lstrip("/")
        if not path or path in seen:
            continue
        body = _strip_optional_code_fence(match.group("body"))
        files.append({"filename": path, "content": body})
        seen.add(path)
    return files


def _parse_explanation(raw_text: str) -> str:
    """Extract the trailing ``===EXPLANATION===`` block, if present."""
    match = _EXPLANATION_RE.search(raw_text)
    return match.group("body").strip() if match else ""


def _strip_optional_code_fence(text: str) -> str:
    """Remove a leading ```lang fence and trailing ``` if the model added one."""
    body = text.strip("\n")
    stripped = body.strip()
    if stripped.startswith("```"):
        first_nl = stripped.find("\n")
        if first_nl != -1:
            inner = stripped[first_nl + 1 :]
            if inner.rstrip().endswith("```"):
                inner = inner.rstrip()[:-3].rstrip("\n")
            return inner
    return body


def _concatenate_for_preview(files: list[dict]) -> str:
    """Glue every file's content into one string for the ``"code"`` field."""
    parts: list[str] = []
    for entry in files:
        parts.append(f"# ===== {entry['filename']} =====")
        parts.append(entry["content"])
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"
