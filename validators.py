"""Input and output validation for mockscreen-to-code-agent.

This module is intentionally **independent from Streamlit**. It uses
duck typing on uploaded-file objects (expecting `.name`, optionally
`.size`, and one of `.getvalue()` / `.read()`), so it works equally well
with `st.UploadedFile` objects, plain `io.BytesIO` blobs, or test fakes.

Each validator returns a `tuple[bool, str]`:

- `(True, "")`  when the input is valid.
- `(False, "helpful human-readable message")` when it is not.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Tuple


# ---------------------------------------------------------------------------
# Module constants (consumed by the UI layer as well)
# ---------------------------------------------------------------------------

ALLOWED_IMAGE_EXTENSIONS: Tuple[str, ...] = ("png", "jpg", "jpeg", "webp")
SUPPORTED_STACKS: Tuple[str, ...] = ("streamlit", "react", "flask")

MIN_IMAGES: int = 1
MAX_IMAGES: int = 3

# Top-level keys the generated-result dict MUST contain.
REQUIRED_RESULT_KEYS: Tuple[str, ...] = (
    "project_name",
    "stack",
    "summary",
    "files",
    "run_instructions",
    "notes",
)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def get_file_extension(filename: str) -> str:
    """Return the lowercase file extension of `filename` without the dot.

    Examples:
        get_file_extension("Login.PNG")     -> "png"
        get_file_extension("photo.tar.gz")  -> "gz"
        get_file_extension("README")        -> ""
        get_file_extension(None)            -> ""
    """
    if not isinstance(filename, str):
        return ""
    if "." not in filename:
        return ""
    return filename.rsplit(".", 1)[-1].strip().lower()


def _detect_size(file_obj: Any) -> Optional[int]:
    """Best-effort byte-size detection.

    Returns:
        The size in bytes if it can be determined, otherwise `None`.
        `None` means "unknown — do not reject on size".
    """
    size = getattr(file_obj, "size", None)
    if isinstance(size, int) and size >= 0:
        return size

    if hasattr(file_obj, "getvalue"):
        try:
            data = file_obj.getvalue()
        except Exception:
            return None
        return len(data) if data is not None else 0

    # We deliberately do NOT consume `.read()` here — that would mutate
    # the stream position and surprise the caller. Treat as unknown.
    return None


def _as_list(uploaded_files: Any) -> list:
    """Normalize input into a list, dropping None entries."""
    if uploaded_files is None:
        return []
    if isinstance(uploaded_files, (list, tuple)):
        return [f for f in uploaded_files if f is not None]
    return [uploaded_files]


# ---------------------------------------------------------------------------
# Uploaded image validation
# ---------------------------------------------------------------------------


def validate_uploaded_images(uploaded_files: Any) -> Tuple[bool, str]:
    """Validate the user's uploaded mock screen image(s).

    Checks:
    - At least `MIN_IMAGES` and at most `MAX_IMAGES` files are uploaded.
    - Each file has a supported extension (png/jpg/jpeg/webp).
    - Each file has non-zero size (when size is detectable).

    Args:
        uploaded_files: A single file-like object, a list/tuple of them,
            or `None`. File-like objects are expected to expose `.name`
            and (optionally) `.size` and/or `.getvalue()`.

    Returns:
        `(True, "")` on success.
        `(False, "<helpful message>")` otherwise. The first failing file
        determines the message so the user can fix issues one at a time.
    """
    files = _as_list(uploaded_files)

    if len(files) < MIN_IMAGES:
        plural = "image" if MIN_IMAGES == 1 else "images"
        return (
            False,
            f"Please upload at least {MIN_IMAGES} mock screen {plural}.",
        )

    if len(files) > MAX_IMAGES:
        return (
            False,
            f"Too many images: {len(files)}. Maximum allowed is {MAX_IMAGES}.",
        )

    allowed_label = ", ".join(f".{ext}" for ext in ALLOWED_IMAGE_EXTENSIONS)

    for f in files:
        name = getattr(f, "name", None)
        if not isinstance(name, str) or not name.strip():
            return False, "An uploaded file has no name; cannot validate it."

        ext = get_file_extension(name)
        if not ext:
            return (
                False,
                f"'{name}' has no file extension. Allowed: {allowed_label}.",
            )
        if ext not in ALLOWED_IMAGE_EXTENSIONS:
            return (
                False,
                f"'{name}' is not a supported image type (got '.{ext}'). "
                f"Allowed: {allowed_label}.",
            )

        size = _detect_size(f)
        if size is not None and size == 0:
            return False, f"'{name}' is empty (0 bytes)."

    return True, ""


# ---------------------------------------------------------------------------
# Generated-result validation
# ---------------------------------------------------------------------------


def _is_relative_path(path: str) -> bool:
    """True iff `path` is a relative path (no absolute / UNC / drive / home)."""
    if not isinstance(path, str) or not path:
        return False
    if path.startswith(("/", "\\", "~")):
        return False
    if len(path) >= 2 and path[1] == ":" and path[0].isalpha():
        return False  # e.g. C:\... or C:/...
    return True


def _path_segments(path: str) -> Iterable[str]:
    """Split a path on either separator."""
    return [seg for seg in path.replace("\\", "/").split("/") if seg]


def validate_generated_result(result: Any) -> Tuple[bool, str]:
    """Validate a Gemini-generated project dict.

    Checks:
    - `result` is a dict.
    - All required keys are present: `project_name`, `stack`, `summary`,
      `files`, `run_instructions`, `notes`.
    - `files` is a non-empty list.
    - Each entry in `files` has a non-empty string `path` and `content`.
    - All paths are relative (no leading `/`, `\\`, `~`, or drive letter).
    - No path contains a `..` segment.
    - No duplicate paths (compared after normalizing separators and
      stripping leading slashes).
    - `content` is not the empty string.

    Note: `validate_generated_result` does not enforce a particular value
    for `stack` here — schema-level fields are checked structurally.
    Higher-level callers (e.g. `code_generator`) check stack identity.

    Returns:
        `(True, "")` on success, otherwise `(False, "<helpful message>")`
        identifying the first failing rule.
    """
    if not isinstance(result, dict):
        return False, "Generated result must be a dict."

    missing = [k for k in REQUIRED_RESULT_KEYS if k not in result]
    if missing:
        return (
            False,
            f"Generated result is missing required key(s): {', '.join(missing)}.",
        )

    files = result.get("files")
    if not isinstance(files, list):
        return False, "'files' must be a list."
    if len(files) == 0:
        return False, "'files' must contain at least one entry."

    seen_paths: set = set()
    for idx, entry in enumerate(files):
        if not isinstance(entry, dict):
            return False, f"files[{idx}] is not an object."

        path = entry.get("path")
        content = entry.get("content")

        if not isinstance(path, str) or not path.strip():
            return False, f"files[{idx}].path is missing or empty."
        if not isinstance(content, str):
            return False, f"files[{idx}].content must be a string."
        if len(content) == 0:
            return False, f"files[{idx}].content is empty."

        if not _is_relative_path(path):
            return (
                False,
                f"files[{idx}].path must be a relative path: {path!r}",
            )

        segments = list(_path_segments(path))
        if ".." in segments:
            return False, f"files[{idx}].path must not contain '..': {path!r}"

        # De-duplicate using a normalized form so 'src/App.jsx' and
        # 'src\\App.jsx' are treated as the same file.
        normalized = "/".join(segments)
        if normalized in seen_paths:
            return False, f"Duplicate file path: {normalized!r}"
        seen_paths.add(normalized)

    return True, ""
