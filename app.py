"""Mockscreen to Code Agent

Streamlit interface that converts UI mock screen images into a runnable
starter project using Google Gemini. Pick a target stack, upload 1-3 mock
screens, optionally add extra instructions, and download the generated code
as a ZIP.

This module only orchestrates the UI; it never executes generated code.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import streamlit as st

from code_generator import (
    DEFAULT_MODEL,
    GeminiAPIError,
    generate_code_from_images,
    get_api_key,
)
from file_builder import (
    create_zip_from_files,
    iter_files_for_preview,
    suggested_zip_filename,
)
from validators import (
    ALLOWED_IMAGE_EXTENSIONS,
    MAX_IMAGES,
    MIN_IMAGES,
    SUPPORTED_STACKS,
    validate_uploaded_images,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mockscreen-to-code-agent")


# ---------------------------------------------------------------------------
# Static config
# ---------------------------------------------------------------------------

PAGE_TITLE = "Mockscreen to Code Agent"
PAGE_SUBTITLE = (
    "Upload mock screens and generate UI code in Streamlit, React, or Flask"
)
FOOTER_TEXT = "Built for Agathon No-Code Series"

STACK_LABELS: Dict[str, str] = {
    "streamlit": "Streamlit (Python)",
    "react": "React (Vite)",
    "flask": "Flask (Python)",
}

KNOWN_MODELS: List[str] = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
    "Custom...",
]

SESSION_KEY_RESULT = "last_result"
SESSION_KEY_ERROR = "last_error"

MISSING_API_KEY_MESSAGE = (
    "Gemini API key is missing. Add GEMINI_API_KEY to Streamlit secrets "
    "or environment variables."
)


# ---------------------------------------------------------------------------
# CSS — minimal, scoped, modern. Injected once at the top of `main()`.
# ---------------------------------------------------------------------------

_CUSTOM_CSS = """
<style>
.hero {
    background: linear-gradient(135deg, #1F6FEB 0%, #6F39EB 100%);
    padding: 2.25rem 2rem;
    border-radius: 14px;
    color: #ffffff;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 24px rgba(31, 111, 235, 0.18);
}
.hero h1 {
    color: #ffffff;
    margin: 0;
    font-size: 2.2rem;
    font-weight: 700;
    letter-spacing: -0.01em;
}
.hero p {
    color: rgba(255, 255, 255, 0.92);
    margin: 0.5rem 0 0;
    font-size: 1.05rem;
}

.workflow {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
    align-items: center;
    justify-content: center;
    margin: 0.5rem 0 1.25rem;
}
.workflow .step {
    background: #f1f3f5;
    color: #212529;
    padding: 0.4rem 0.85rem;
    border-radius: 8px;
    font-weight: 500;
    font-size: 0.95rem;
}
.workflow .arrow {
    color: #adb5bd;
    font-size: 1.05rem;
}

.status-pill {
    display: inline-block;
    padding: 0.32rem 0.85rem;
    border-radius: 999px;
    font-size: 0.88rem;
    font-weight: 600;
    letter-spacing: 0.01em;
    margin: 0.25rem 0 0.5rem;
}
.status-waiting    { background: #f1f3f5; color: #495057; }
.status-ready      { background: #e7f5ff; color: #1971c2; }
.status-generating { background: #fff9db; color: #9c7900; }
.status-done       { background: #ebfbee; color: #2b8a3e; }
.status-error      { background: #ffe3e3; color: #c92a2a; }

.footer {
    text-align: center;
    padding: 2rem 0 1rem;
    color: #868e96;
    font-size: 0.9rem;
}
.footer strong { color: #495057; }
</style>
"""


def _inject_css() -> None:
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


def _init_session_state() -> None:
    if SESSION_KEY_RESULT not in st.session_state:
        st.session_state[SESSION_KEY_RESULT] = None
    if SESSION_KEY_ERROR not in st.session_state:
        st.session_state[SESSION_KEY_ERROR] = None


# ---------------------------------------------------------------------------
# Sidebar (unchanged behavior, kept compact and clean)
# ---------------------------------------------------------------------------


def _render_sidebar(api_key_present: bool) -> Dict[str, str]:
    """Render the sidebar controls and return the user's selections."""
    with st.sidebar:
        st.markdown("### Setup")
        if api_key_present:
            st.success("Gemini API key detected", icon="✅")
        else:
            st.error(MISSING_API_KEY_MESSAGE, icon="🔑")
            st.caption(
                "For local development, copy "
                "`.streamlit/secrets.toml.example` to "
                "`.streamlit/secrets.toml` and paste your key. "
                "For Streamlit Community Cloud, paste the key under "
                "**Settings → Secrets**."
            )

        st.divider()
        st.markdown("### Configuration")

        stack = st.radio(
            "Target tech stack",
            options=list(SUPPORTED_STACKS),
            format_func=lambda key: STACK_LABELS.get(key, key.title()),
            index=0,
            help="Which stack should Gemini generate code for?",
        )

        model_choice = st.selectbox(
            "Gemini model",
            options=KNOWN_MODELS,
            index=0,
            help="Vision-capable Gemini model. Defaults to gemini-2.5-flash.",
        )
        if model_choice == "Custom...":
            custom = st.text_input(
                "Custom model id",
                value=DEFAULT_MODEL,
                placeholder="e.g. gemini-2.5-flash",
            ).strip()
            model = custom or DEFAULT_MODEL
        else:
            model = model_choice

        st.divider()
        st.markdown("### Extra instructions")
        extra_instructions = st.text_area(
            "Optional notes for Gemini",
            placeholder=(
                "e.g. Use a dark theme, keep brand color #1F6FEB, place the "
                "form on the right, keep copy concise."
            ),
            height=140,
            label_visibility="collapsed",
        )

        st.divider()
        with st.expander("Tips for best results"):
            st.markdown(
                "- Upload **1-3** mock screen images (png, jpg, jpeg, webp).\n"
                "- Clear, high-contrast mocks produce better code.\n"
                "- Use the **Extra instructions** field for colors, copy, "
                "naming, or behavior you want preserved.\n"
                "- Generated code is for review only — this app does **not** "
                "execute it."
            )

    return {
        "stack": stack,
        "model": model,
        "extra_instructions": extra_instructions,
    }


# ---------------------------------------------------------------------------
# Hero, feature cards, workflow
# ---------------------------------------------------------------------------


def _render_hero() -> None:
    st.markdown(
        f"""
        <div class="hero">
            <h1>🧩 {PAGE_TITLE}</h1>
            <p>{PAGE_SUBTITLE}.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_feature_cards() -> None:
    cards = [
        (
            "📤",
            "Upload mock screens",
            "Drop 1-3 PNG / JPG / JPEG / WEBP images. Multi-screen mocks "
            "are treated as states of the same product.",
        ),
        (
            "🎯",
            "Choose target stack",
            "Streamlit, React (Vite), or Flask. The prompt and file "
            "layout adapt to whatever stack you pick.",
        ),
        (
            "📦",
            "Download generated code",
            "Gemini returns strict JSON. We parse, validate, package "
            "everything into a clean ZIP, and preview every file inline.",
        ),
    ]
    cols = st.columns(3, gap="medium")
    for col, (icon, title, body) in zip(cols, cards):
        with col:
            with st.container(border=True):
                st.markdown(f"#### {icon} {title}")
                st.caption(body)


def _render_workflow() -> None:
    st.markdown(
        """
        <div class="workflow">
            <span class="step">📤 Upload mocks</span>
            <span class="arrow">→</span>
            <span class="step">🎯 Configure stack</span>
            <span class="arrow">→</span>
            <span class="step">✨ Generate</span>
            <span class="arrow">→</span>
            <span class="step">📦 Download ZIP</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Status badge
# ---------------------------------------------------------------------------

_STATUS_LABELS = {
    "waiting": ("status-waiting", "⏳ Waiting for upload"),
    "ready": ("status-ready", "🟢 Ready to generate"),
    "generating": ("status-generating", "✨ Generating code with Gemini..."),
    "done": ("status-done", "✅ Code generated"),
    "error": ("status-error", "⚠️ Generation failed"),
}


def _status_html(state: str) -> str:
    css_class, label = _STATUS_LABELS.get(state, _STATUS_LABELS["waiting"])
    return f'<span class="status-pill {css_class}">{label}</span>'


def _set_status(placeholder: Any, state: str) -> None:
    """Render the status badge into a stable placeholder slot."""
    placeholder.markdown(_status_html(state), unsafe_allow_html=True)


def _determine_status(
    *, uploaded_files: List[object], has_result: bool, has_error: bool
) -> str:
    if has_error:
        return "error"
    if has_result:
        return "done"
    if len(uploaded_files) >= MIN_IMAGES:
        return "ready"
    return "waiting"


# ---------------------------------------------------------------------------
# Uploader + image preview grid
# ---------------------------------------------------------------------------


def _render_uploader() -> List[object]:
    st.markdown("#### 1. Upload mock screens")
    st.caption(
        f"Drop {MIN_IMAGES} to {MAX_IMAGES} images "
        f"({', '.join(ALLOWED_IMAGE_EXTENSIONS)}). "
        "Generated code is never executed."
    )
    uploaded = st.file_uploader(
        "Mock screen images",
        type=list(ALLOWED_IMAGE_EXTENSIONS),
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    return list(uploaded or [])


def _render_image_preview_grid(uploaded_files: List[object]) -> None:
    if not uploaded_files:
        return
    files = uploaded_files[:MAX_IMAGES]
    cols = st.columns(len(files), gap="small")
    for col, f in zip(cols, files):
        with col:
            with st.container(border=True):
                st.image(
                    f,
                    caption=getattr(f, "name", "image"),
                    use_container_width=True,
                )


# ---------------------------------------------------------------------------
# Generation pipeline
# ---------------------------------------------------------------------------


def _do_generation(
    *,
    stack: str,
    model: str,
    uploaded_files: List[object],
    extra_instructions: str,
) -> None:
    """Call the generator and store the result/error in session state."""
    try:
        with st.spinner("Generating code with Gemini..."):
            project = generate_code_from_images(
                uploaded_files,
                target_stack=stack,
                extra_instructions=extra_instructions,
                model_name=model,
            )
        st.session_state[SESSION_KEY_RESULT] = project
        st.session_state[SESSION_KEY_ERROR] = None
    except ValueError as exc:
        st.session_state[SESSION_KEY_RESULT] = None
        st.session_state[SESSION_KEY_ERROR] = str(exc)
    except GeminiAPIError as exc:
        st.session_state[SESSION_KEY_RESULT] = None
        st.session_state[SESSION_KEY_ERROR] = f"Gemini API error: {exc}"
    except Exception as exc:  # last-resort safety net
        logger.exception("Unexpected error during generation")
        st.session_state[SESSION_KEY_RESULT] = None
        st.session_state[SESSION_KEY_ERROR] = f"Unexpected error: {exc}"


# ---------------------------------------------------------------------------
# Results: metrics + 4 tabs (Summary / Generated Files / Run / Safety)
# ---------------------------------------------------------------------------


_SAFETY_REMINDERS = [
    "This app does **not** execute generated code. It is for review only.",
    "Review the generated code carefully before running it in production.",
    "Audit `requirements.txt` / `package.json` for unfamiliar dependencies.",
    "Treat Gemini's output like a contractor's first draft — read, lint, test.",
    "Never upload customer or production data — images are sent to Google's API.",
]


def _render_results(project: Dict[str, Any]) -> None:
    stack = project.get("stack", "")
    stack_label = STACK_LABELS.get(stack, stack.title() if stack else "Unknown")
    project_name = project.get("project_name", "generated-project")
    summary = project.get("summary", "")
    files = project.get("files", []) or []
    run_instructions = project.get("run_instructions", []) or []
    notes = project.get("notes", []) or []

    st.success(
        f"Generated **{project_name}** for **{stack_label}**.",
        icon="✨",
    )

    # Metrics row + download button
    m1, m2, m3, m4 = st.columns([1, 1, 1, 1.4])
    m1.metric("Files generated", len(files))
    m2.metric("Target stack", stack_label)
    m3.metric("Run steps", len(run_instructions))
    with m4:
        st.write("")  # vertical breathing room
        zip_bytes = create_zip_from_files(project)
        st.download_button(
            label="📦 Download ZIP",
            data=zip_bytes,
            file_name=suggested_zip_filename(project),
            mime="application/zip",
            type="primary",
            use_container_width=True,
        )

    tab_summary, tab_files, tab_run, tab_safety = st.tabs(
        ["Summary", f"Generated Files ({len(files)})", "Run Instructions", "Safety Notes"]
    )

    with tab_summary:
        st.markdown(f"### {project_name}")
        st.caption(f"Target stack: **{stack_label}**")
        if summary:
            st.markdown(summary)
        else:
            st.info("Gemini did not return a summary for this project.")

    with tab_files:
        if not files:
            st.info("No files in the generated project.")
        else:
            st.caption("Click any file to view its contents.")
            for path, content, language in iter_files_for_preview(project):
                with st.expander(path, expanded=False):
                    if language:
                        st.code(content, language=language)
                    else:
                        st.code(content)

    with tab_run:
        if run_instructions:
            for i, step in enumerate(run_instructions, 1):
                st.markdown(f"**{i}.** {step}")
        else:
            st.info("Gemini did not return any run instructions.")

    with tab_safety:
        if notes:
            st.markdown("**Notes from Gemini**")
            for note in notes:
                st.markdown(f"- {note}")
            st.divider()
        st.markdown("**General safety reminders**")
        for reminder in _SAFETY_REMINDERS:
            st.markdown(f"- {reminder}")


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------


def _render_footer() -> None:
    st.markdown(
        '<div class="footer">Built for '
        '<strong>Agathon No-Code Series</strong></div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon="🧩",
        layout="wide",
    )
    _inject_css()
    _init_session_state()

    # We only check *presence* of the key here. The actual value is never
    # passed through the UI layer; the generator resolves it itself.
    api_key_present = get_api_key() is not None
    sidebar_state = _render_sidebar(api_key_present=api_key_present)

    # --- Hero + feature cards + workflow ---
    _render_hero()
    _render_feature_cards()
    _render_workflow()

    st.divider()

    # --- Uploader + preview grid ---
    uploaded_files = _render_uploader()
    _render_image_preview_grid(uploaded_files)

    if uploaded_files and len(uploaded_files) > MAX_IMAGES:
        st.warning(
            f"You uploaded {len(uploaded_files)} images. Only the first "
            f"{MAX_IMAGES} will be sent to Gemini."
        )

    # --- Status badge (gets updated live during generation) ---
    status_placeholder = st.empty()
    _set_status(
        status_placeholder,
        _determine_status(
            uploaded_files=uploaded_files,
            has_result=st.session_state.get(SESSION_KEY_RESULT) is not None,
            has_error=bool(st.session_state.get(SESSION_KEY_ERROR)),
        ),
    )

    # --- Generate button ---
    can_generate = api_key_present and MIN_IMAGES <= len(uploaded_files) <= MAX_IMAGES
    generate_clicked = st.button(
        "✨ Generate Code",
        type="primary",
        use_container_width=True,
        disabled=not can_generate,
        help=(
            None
            if can_generate
            else "Add your API key and upload 1-3 mock screen images first."
        ),
    )

    if generate_clicked:
        if not api_key_present:
            _set_status(status_placeholder, "error")
            st.error(MISSING_API_KEY_MESSAGE)
            return

        stack = sidebar_state["stack"]

        ok, message = validate_uploaded_images(uploaded_files)
        if not ok:
            _set_status(status_placeholder, "error")
            st.error(message)
            return

        # Live status update while the spinner is active.
        _set_status(status_placeholder, "generating")

        _do_generation(
            stack=stack,
            model=sidebar_state["model"],
            uploaded_files=uploaded_files[:MAX_IMAGES],
            extra_instructions=sidebar_state["extra_instructions"],
        )

        # Reflect final state in the badge.
        _set_status(
            status_placeholder,
            "done" if st.session_state.get(SESSION_KEY_RESULT) is not None else "error",
        )

    # --- Error surface ---
    if st.session_state.get(SESSION_KEY_ERROR):
        st.error(st.session_state[SESSION_KEY_ERROR])

    # --- Results (metric row + 4 tabs) ---
    result: Optional[Dict[str, Any]] = st.session_state.get(SESSION_KEY_RESULT)
    if result is not None:
        st.divider()
        _render_results(result)
        st.divider()
        if st.button("Clear result", use_container_width=False):
            st.session_state[SESSION_KEY_RESULT] = None
            st.session_state[SESSION_KEY_ERROR] = None
            st.rerun()

    # --- Footer ---
    _render_footer()


if __name__ == "__main__":
    main()
