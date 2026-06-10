"""Convert a Gemini-generated project dict into a downloadable ZIP.

The "generated_result" exchanged between modules is the plain `dict`
returned by `code_generator.generate_code_from_images`. Its schema:

    {
        "project_name": str,
        "stack": "streamlit" | "react" | "flask",
        "summary": str,
        "files": [{"path": str, "content": str}, ...],
        "run_instructions": [str, ...],
        "notes": [str, ...],
    }

This module is intentionally dependency-free — it uses only the Python
standard library (`io`, `re`, `zipfile`, `pathlib`, `typing`).
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# Characters NOT allowed in a project slug. Replaced with single dashes.
_UNSAFE_NAME_PATTERN = re.compile(r"[^a-zA-Z0-9._-]+")

# Common extensions → Streamlit `st.code` language hints, used by the UI.
_EXTENSION_TO_LANGUAGE: Dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "jsx",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "scss",
    ".json": "json",
    ".md": "markdown",
    ".toml": "toml",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".sh": "bash",
    ".txt": "text",
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def sanitize_project_name(name: str) -> str:
    """Turn an arbitrary project name into a filesystem-safe slug.

    - Lowercased.
    - Whitespace and unsafe characters are collapsed into a single `-`.
    - Leading / trailing `-`, `_`, `.` are stripped.
    - Empty results fall back to `"generated-project"`.

    Examples:
        sanitize_project_name("My Cool App!")   -> "my-cool-app"
        sanitize_project_name("  ../bad/name") -> "bad-name"
        sanitize_project_name("")               -> "generated-project"
        sanitize_project_name(None)             -> "generated-project"
    """
    if not isinstance(name, str):
        return "generated-project"
    slug = _UNSAFE_NAME_PATTERN.sub("-", name.strip().lower()).strip("-._")
    return slug or "generated-project"


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------


def _normalize_path(path: str) -> str:
    """Normalize separators, strip leading slashes, collapse `./` segments.

    Does **not** decide whether the path is safe — that's `_is_safe_relative_path`.
    """
    cleaned = path.strip().replace("\\", "/").lstrip("/")
    parts = [p for p in cleaned.split("/") if p and p != "."]
    return "/".join(parts)


def _is_safe_relative_path(path: str) -> bool:
    """Return True only if `path` is a relative path with no traversal.

    Rejected:
    - Empty strings.
    - Unix absolute paths (`/foo`).
    - Windows absolute paths (`C:\\foo`, `C:/foo`) and UNC paths (`\\\\server`).
    - Home-expansions (`~/foo`).
    - Any segment equal to `..`.
    - Paths containing NUL bytes.
    """
    if not isinstance(path, str) or not path:
        return False
    # Reject Unix absolute / UNC / Windows absolute prefixes on the raw input.
    if path.startswith(("/", "\\", "~")):
        return False
    if len(path) >= 2 and path[1] == ":" and path[0].isalpha():
        return False
    if "\x00" in path:
        return False

    parts = path.replace("\\", "/").split("/")
    for segment in parts:
        if segment == "..":
            return False
    return True


# ---------------------------------------------------------------------------
# README_GENERATED.md
# ---------------------------------------------------------------------------


def _format_readme(generated_result: Dict[str, Any]) -> str:
    """Build the contents of the in-ZIP `README_GENERATED.md`."""
    project_name = (generated_result.get("project_name") or "Generated Project").strip()
    stack = (generated_result.get("stack") or "unknown").strip()
    summary = (generated_result.get("summary") or "").strip()

    run_instructions = generated_result.get("run_instructions") or []
    notes = generated_result.get("notes") or []

    lines: List[str] = [f"# {project_name}", "", f"**Stack:** `{stack}`", ""]

    lines.append("## Summary")
    lines.append("")
    lines.append(summary if summary else "_No summary provided._")
    lines.append("")

    lines.append("## Run instructions")
    lines.append("")
    if isinstance(run_instructions, list) and run_instructions:
        for i, step in enumerate(run_instructions, 1):
            text = str(step).strip()
            if text:
                lines.append(f"{i}. {text}")
    else:
        lines.append("_No run instructions provided._")
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    if isinstance(notes, list) and notes:
        for note in notes:
            text = str(note).strip()
            if text:
                lines.append(f"- {text}")
    else:
        lines.append("_No additional notes._")
    lines.append("")

    lines.append("---")
    lines.append("_Generated by mockscreen-to-code-agent._")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def create_zip_from_files(generated_result: Dict[str, Any]) -> bytes:
    """Convert a Gemini-generated project dict into an in-memory ZIP.

    Args:
        generated_result: The dict produced by
            `code_generator.generate_code_from_images`. Must contain a
            `files` key whose value is a list of `{"path", "content"}`
            dicts.

    Behavior:
        - All files are nested under a top-level directory named after a
          sanitized project slug, so unzipping produces a clean folder.
        - Paths are normalized (forward slashes, no leading `/`, no
          `./` segments) before being added to the archive.
        - Absolute paths, paths containing `..` segments, Windows drive
          paths, UNC paths, home-expanded paths, and paths with NUL
          bytes are **rejected** (silently skipped).
        - Duplicate arcnames are de-duplicated (first occurrence wins).
        - A `README_GENERATED.md` file is added at the project root,
          containing the project name, stack, summary, run instructions,
          and notes.

    Returns:
        The raw bytes of the ZIP archive.

    Raises:
        ValueError: when `generated_result` is not a dict or its `files`
            value is not a list.
    """
    if not isinstance(generated_result, dict):
        raise ValueError("generated_result must be a dict.")

    files = generated_result.get("files")
    if not isinstance(files, list):
        raise ValueError("generated_result['files'] must be a list.")

    slug = sanitize_project_name(generated_result.get("project_name", ""))

    buffer = io.BytesIO()
    seen_arcnames: set = set()

    with zipfile.ZipFile(
        buffer, mode="w", compression=zipfile.ZIP_DEFLATED
    ) as zf:
        for entry in files:
            if not isinstance(entry, dict):
                continue

            raw_path = entry.get("path", "")
            content = entry.get("content", "")
            if not isinstance(raw_path, str) or not raw_path.strip():
                continue
            if not isinstance(content, str):
                # Defensive: refuse to write non-string payloads to the ZIP.
                continue

            # Reject unsafe inputs *before* normalization to be strict.
            if not _is_safe_relative_path(raw_path):
                continue

            normalized = _normalize_path(raw_path)
            if not normalized:
                continue
            # Belt-and-suspenders: re-check after normalization.
            if not _is_safe_relative_path(normalized):
                continue

            arcname = f"{slug}/{normalized}"
            if arcname in seen_arcnames:
                continue
            seen_arcnames.add(arcname)
            zf.writestr(arcname, content)

        readme_arcname = f"{slug}/README_GENERATED.md"
        # If Gemini happened to emit a file at that exact path, the user's
        # version wins (it was added above). Otherwise, write ours.
        if readme_arcname not in seen_arcnames:
            zf.writestr(readme_arcname, _format_readme(generated_result))

    return buffer.getvalue()


# ---------------------------------------------------------------------------
# UI-facing convenience helpers (used by app.py for previews + filenames).
# These do not write to disk; everything is in-memory or string-based.
# ---------------------------------------------------------------------------


def suggested_zip_filename(generated_result: Dict[str, Any]) -> str:
    """Return a friendly download filename like `my-app-streamlit.zip`."""
    slug = sanitize_project_name(generated_result.get("project_name", ""))
    stack = (generated_result.get("stack") or "").strip().lower() or "project"
    return f"{slug}-{stack}.zip"


def iter_files_for_preview(
    generated_result: Dict[str, Any],
) -> Iterable[Tuple[str, str, Optional[str]]]:
    """Yield `(path, content, language_hint)` tuples for UI rendering.

    `language_hint` is suitable as the `language=` argument to `st.code`.
    """
    files: List[Dict[str, str]] = generated_result.get("files") or []
    for entry in files:
        if not isinstance(entry, dict):
            continue
        path = entry.get("path", "")
        content = entry.get("content", "")
        if not isinstance(path, str) or not isinstance(content, str):
            continue
        suffix = Path(path).suffix.lower()
        yield path, content, _EXTENSION_TO_LANGUAGE.get(suffix)
