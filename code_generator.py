"""Gemini integration for mockscreen-to-code-agent.

This module is responsible for:

- Resolving the Gemini API key from `st.secrets` or `os.environ`.
- Validating uploaded mock screen images (1-3 png/jpg/jpeg/webp files).
- Building a multimodal request that combines the system instruction, the
  target stack, optional extra instructions, and the uploaded images as
  inline image parts.
- Calling Google Gemini through the official `google-genai` SDK with strict
  JSON output enabled.
- Robustly extracting a JSON object from the model's response (raw JSON,
  ```json fences, generic ``` fences, or JSON embedded inside prose).
- Validating that the parsed payload matches the documented schema and
  returning a plain `dict` to the caller.

The module **never** executes generated code and **never** writes generated
files to disk. Those responsibilities live in `app.py` and `file_builder.py`.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from prompt_templates import SYSTEM_INSTRUCTION, build_generation_prompt
from validators import validate_generated_result

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "gemini-2.5-flash"

MIN_IMAGES = 1
MAX_IMAGES = 3

ALLOWED_EXTENSIONS = ("png", "jpg", "jpeg", "webp")
ALLOWED_MIME_TYPES = ("image/png", "image/jpeg", "image/jpg", "image/webp")
ALLOWED_STACKS = ("streamlit", "react", "flask")

_EXTENSION_TO_MIME: Dict[str, str] = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
}


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class GeminiAPIError(RuntimeError):
    """Raised when the Gemini API call fails (auth, network, SDK)."""


class RateLimitError(ValueError):
    """Raised when Gemini responds with 429 / RESOURCE_EXHAUSTED / quota.

    Subclasses `ValueError` so existing handlers that catch `ValueError`
    continue to display the message without changes, but lets callers
    branch on a rate-limit failure specifically (e.g. to show a softer
    `st.warning` instead of a red `st.error`).

    Rate-limit errors are **never** auto-retried by this module — the
    free tier requires a full ~60s wait. Surfacing the error and letting
    the user retry manually is the intended UX.
    """


RATE_LIMIT_MESSAGE = (
    "Rate limit exceeded. You are making requests too quickly for the "
    "free tier. Please wait 60 seconds and try again. Consider uploading "
    "fewer images to save tokens."
)


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Heuristic check for a Gemini 429 / RESOURCE_EXHAUSTED / quota error.

    Different parts of the `google-genai` stack expose the status code in
    different attributes (`code`, `status_code`, `http_status`, `status`)
    and sometimes only via the stringified message. We check both.
    """
    for attr in ("code", "status_code", "http_status", "status"):
        value = getattr(exc, attr, None)
        if value == 429:
            return True
        if isinstance(value, str) and "429" in value:
            return True

    message = str(exc)
    if "429" in message:
        return True
    if "RESOURCE_EXHAUSTED" in message:
        return True
    if "quota" in message.lower():
        return True
    return False


# Note: other invalid-response / shape issues are intentionally raised as
# plain `ValueError` so callers can rely on a stable, well-known exception
# type for "the model gave us something we couldn't use".


# ---------------------------------------------------------------------------
# API key resolution
# ---------------------------------------------------------------------------


def get_api_key() -> Optional[str]:
    """Resolve the Gemini API key.

    Lookup order:
    1. `st.secrets["GEMINI_API_KEY"]` when Streamlit is importable and the
       secret is set (i.e. the user has a `.streamlit/secrets.toml`).
    2. `os.environ["GEMINI_API_KEY"]` as a fallback.

    Returns:
        The key string, or `None` if not found in either location.

    The key value itself is never logged or echoed.
    """
    # Try Streamlit secrets first. We import Streamlit lazily so this module
    # remains useful in non-Streamlit contexts (CLI, tests, notebooks).
    try:
        import streamlit as st  # type: ignore

        try:
            if "GEMINI_API_KEY" in st.secrets:
                value = st.secrets["GEMINI_API_KEY"]
                if isinstance(value, str) and value.strip():
                    return value.strip()
        except Exception:
            # st.secrets raises if no secrets.toml is configured.
            pass
    except ImportError:
        pass

    env_value = os.environ.get("GEMINI_API_KEY")
    if env_value and env_value.strip():
        return env_value.strip()
    return None


# ---------------------------------------------------------------------------
# Image normalization
# ---------------------------------------------------------------------------


@dataclass
class _ImagePart:
    """Internal representation of a single uploaded image."""

    name: str
    data: bytes
    mime_type: str


def _guess_mime_type(filename: str, declared: Optional[str] = None) -> str:
    """Infer a MIME type from a declared metadata value or the file extension.

    `image/jpg` (non-standard but common) is normalized to `image/jpeg`.
    Falls back to `image/png` when no signal is available.
    """
    if isinstance(declared, str):
        normalized = declared.strip().lower()
        if normalized == "image/jpg":
            return "image/jpeg"
        if normalized in {"image/png", "image/jpeg", "image/webp"}:
            return normalized

    if "." in filename:
        ext = filename.rsplit(".", 1)[-1].lower()
        if ext in _EXTENSION_TO_MIME:
            return _EXTENSION_TO_MIME[ext]
    return "image/png"


def _read_bytes(uploaded_file: Any) -> bytes:
    """Extract raw bytes from a Streamlit UploadedFile or any file-like object."""
    if hasattr(uploaded_file, "getvalue"):
        return uploaded_file.getvalue()
    if hasattr(uploaded_file, "read"):
        data = uploaded_file.read()
        if hasattr(uploaded_file, "seek"):
            try:
                uploaded_file.seek(0)
            except Exception:
                pass
        return data
    raise ValueError(
        f"Uploaded file {getattr(uploaded_file, 'name', uploaded_file)!r} "
        "is not readable."
    )


def _normalize_uploaded_files(uploaded_files: Any) -> List[_ImagePart]:
    """Validate count and types, load bytes, and return image parts.

    Raises:
        ValueError: when the count is out of range or any file is not a
            supported image type / is empty.
    """
    if uploaded_files is None:
        raise ValueError(
            f"Please upload at least {MIN_IMAGES} mock screen image."
        )

    # Allow a single UploadedFile to be passed without a list wrapper.
    if not isinstance(uploaded_files, (list, tuple)):
        uploaded_files = [uploaded_files]

    files = [f for f in uploaded_files if f is not None]

    if len(files) < MIN_IMAGES:
        raise ValueError(
            f"Please upload at least {MIN_IMAGES} mock screen image."
        )
    if len(files) > MAX_IMAGES:
        raise ValueError(
            f"Too many images: {len(files)}. Maximum allowed is {MAX_IMAGES}."
        )

    parts: List[_ImagePart] = []
    for f in files:
        name = getattr(f, "name", "image")
        declared_mime = getattr(f, "type", None)

        ext_ok = (
            "." in name
            and name.rsplit(".", 1)[-1].lower() in ALLOWED_EXTENSIONS
        )
        mime_ok = (
            isinstance(declared_mime, str)
            and declared_mime.strip().lower() in ALLOWED_MIME_TYPES
        )
        if not (ext_ok or mime_ok):
            raise ValueError(
                f"'{name}' is not a supported image type. "
                f"Allowed extensions: {', '.join(ALLOWED_EXTENSIONS)}."
            )

        data = _read_bytes(f)
        if not data:
            raise ValueError(f"'{name}' is empty.")

        parts.append(
            _ImagePart(
                name=name,
                data=data,
                mime_type=_guess_mime_type(name, declared_mime),
            )
        )

    return parts


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(
    r"```(?:json|JSON)?\s*(?P<body>.*?)\s*```",
    re.DOTALL,
)


_PREVIEW_CHARS = 240


def _preview(text: str, limit: int = _PREVIEW_CHARS) -> str:
    """Return a short, single-line preview of `text` for error messages."""
    if not isinstance(text, str):
        return repr(text)[:limit]
    flattened = " ".join(text.split())
    if len(flattened) <= limit:
        return flattened
    return flattened[:limit] + "..."


def _strip_markdown_fences(text: str) -> Optional[str]:
    """Return the body inside ```` ```json … ``` ```` or ```` ``` … ``` ```` fences, or None."""
    match = _FENCE_RE.search(text)
    if not match:
        return None
    body = match.group("body").strip()
    return body or None


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """Robustly extract a JSON object from arbitrary model output.

    Strategy, attempted in order:

    1. `json.loads(text)` directly.
    2. If that fails, remove markdown code fences (```` ```json … ``` ```` or
       generic ```` ``` … ``` ````) and parse the contents.
    3. If that fails, take the substring from the first `{` to the last
       `}` (inclusive) and parse it.
    4. If that fails, raise `ValueError` with a short preview of the
       response.

    Args:
        text: The raw response text from the model.

    Returns:
        The parsed JSON object as a Python dict.

    Raises:
        ValueError: when no strategy yields a JSON object. The message
            includes a short preview of the response so the caller can
            surface it to the user / logs.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Cannot extract JSON: response text is empty.")

    last_error: Optional[Exception] = None

    # Strategy 1: parse the whole response as-is.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        last_error = ValueError("Top-level JSON value is not an object.")
    except json.JSONDecodeError as exc:
        last_error = exc

    # Strategy 2: strip markdown fences if present, parse what's inside.
    fenced = _strip_markdown_fences(text)
    if fenced:
        try:
            parsed = json.loads(fenced)
            if isinstance(parsed, dict):
                return parsed
            last_error = ValueError("Top-level JSON value is not an object.")
        except json.JSONDecodeError as exc:
            last_error = exc

    # Strategy 3: substring from the first '{' to the last '}'.
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidate = text[first : last + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
            last_error = ValueError("Top-level JSON value is not an object.")
        except json.JSONDecodeError as exc:
            last_error = exc

    # Strategy 4: give up with a clear, previewable error.
    raise ValueError(
        "Could not parse JSON from Gemini response. "
        f"Last parser error: {last_error}. "
        f"Response preview: {_preview(text)!r}"
    )


# ---------------------------------------------------------------------------
# Payload validation
# ---------------------------------------------------------------------------


def _coerce_str_list(value: Any) -> List[str]:
    """Coerce arbitrary model output into a clean list[str]."""
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, list):
        return [
            str(item).strip()
            for item in value
            if item is not None and str(item).strip()
        ]
    return []


def _normalize_payload(
    payload: Dict[str, Any], *, expected_stack: str
) -> Dict[str, Any]:
    """Light normalization applied *after* strict schema validation.

    Assumes `payload` already passed `validators.validate_generated_result`
    (i.e. required keys exist, files have non-empty path + content, paths
    are relative with no `..`, no duplicates). This function only:

    - Strips whitespace from `project_name` and `summary`.
    - Lowercases `stack` and overrides it to `expected_stack` if the
      model drifted (with a warning log).
    - Coerces `run_instructions` and `notes` to clean `list[str]`.
    - Normalizes file paths to forward slashes (no traversal cleanup —
      that's the validator's job to reject upstream).
    """
    out: Dict[str, Any] = {}

    out["project_name"] = str(payload.get("project_name", "")).strip()

    raw_stack = str(payload.get("stack", "")).strip().lower()
    if raw_stack != expected_stack:
        logger.warning(
            "Gemini returned stack=%r but %r was requested; overriding.",
            raw_stack,
            expected_stack,
        )
    out["stack"] = expected_stack

    summary = payload.get("summary", "")
    out["summary"] = summary.strip() if isinstance(summary, str) else ""

    normalized_files: List[Dict[str, str]] = []
    for entry in payload.get("files", []):
        path = str(entry.get("path", "")).strip().replace("\\", "/")
        content = entry.get("content", "")
        normalized_files.append({"path": path, "content": content})
    out["files"] = normalized_files

    out["run_instructions"] = _coerce_str_list(payload.get("run_instructions"))
    out["notes"] = _coerce_str_list(payload.get("notes"))

    return out


# ---------------------------------------------------------------------------
# Gemini client + request
# ---------------------------------------------------------------------------


def _build_client(api_key: str):
    """Construct a `google-genai` Client. Imported lazily."""
    try:
        from google import genai  # type: ignore
    except ImportError as exc:  # pragma: no cover - import guard
        raise GeminiAPIError(
            "The 'google-genai' package is not installed. "
            "Run `pip install -r requirements.txt`."
        ) from exc
    try:
        return genai.Client(api_key=api_key)
    except Exception as exc:
        raise GeminiAPIError(f"Failed to initialize Gemini client: {exc}") from exc


def _build_contents(prompt: str, images: Sequence[_ImagePart]) -> list:
    """Build the multimodal `contents` list: prompt + inline image parts."""
    from google.genai import types  # type: ignore

    contents: list = [prompt]
    for img in images:
        contents.append(
            types.Part.from_bytes(data=img.data, mime_type=img.mime_type)
        )
    return contents


def _call_gemini(
    *,
    api_key: str,
    prompt: str,
    images: Sequence[_ImagePart],
    model_name: str,
) -> str:
    """Invoke `client.models.generate_content` and return the response text.

    Raises:
        RateLimitError: on Gemini 429 / RESOURCE_EXHAUSTED / quota errors.
            **Not** auto-retried — the caller (UI) is responsible for
            asking the user to wait.
        GeminiAPIError: on other network/SDK failures.
        ValueError: when Gemini returns no usable text.
    """
    from google.genai import types  # type: ignore

    client = _build_client(api_key)
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        response_mime_type="application/json",
        temperature=0.2,
    )

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=_build_contents(prompt, images),
            config=config,
        )
    except Exception as exc:  # SDK exceptions vary; surface them uniformly.
        # 429 / quota errors get their own clean message and never retry.
        if _is_rate_limit_error(exc):
            logger.warning("Gemini rate limit hit: %s", exc)
            raise RateLimitError(RATE_LIMIT_MESSAGE) from exc
        raise GeminiAPIError(f"Gemini API call failed: {exc}") from exc

    text = getattr(response, "text", None) or ""
    if not text.strip():
        # Some SDK builds expose text only via candidates[].content.parts[].text
        try:
            candidates = getattr(response, "candidates", []) or []
            if candidates:
                parts = getattr(candidates[0].content, "parts", []) or []
                text = "".join(getattr(p, "text", "") or "" for p in parts)
        except Exception:  # pragma: no cover - defensive
            text = ""

    if not text.strip():
        raise ValueError("Invalid response format: Gemini returned no text.")
    return text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_code_from_images(
    uploaded_files,
    target_stack: str,
    extra_instructions: str = "",
    model_name: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """Generate a runnable starter project from UI mock screen images.

    Args:
        uploaded_files: An iterable of Streamlit `UploadedFile` objects (or
            any objects exposing `name`, `type`, and either `getvalue()` or
            `read()`). Must contain 1 to 3 image files (png/jpg/jpeg/webp).
        target_stack: One of `"streamlit"`, `"react"`, `"flask"`.
        extra_instructions: Optional free-form notes from the user that are
            passed verbatim to the model as soft guidance.
        model_name: Gemini model identifier. Defaults to `"gemini-2.5-flash"`.

    Returns:
        A `dict` matching the project schema documented at the top of this
        module / in the README. Keys: `project_name`, `stack`, `summary`,
        `files` (list of `{path, content}`), `run_instructions`, `notes`.

    Raises:
        ValueError: when inputs are invalid, when the response cannot be
            parsed as JSON, or when the parsed payload does not match the
            expected schema.
        GeminiAPIError: when the Gemini API call itself fails (missing key,
            network error, SDK error).
    """
    # 1. Validate the requested stack.
    if not isinstance(target_stack, str) or not target_stack.strip():
        raise ValueError("Target stack is required.")
    stack = target_stack.strip().lower()
    if stack not in ALLOWED_STACKS:
        raise ValueError(
            f"Unsupported target stack {target_stack!r}. "
            f"Choose one of: {list(ALLOWED_STACKS)}."
        )

    # 2. Resolve the API key. Raised as GeminiAPIError (not ValueError) because
    #    it is an auth/config problem rather than a response-format problem.
    api_key = get_api_key()
    if not api_key:
        raise GeminiAPIError(
            "Gemini API key is missing. Add GEMINI_API_KEY to Streamlit "
            "secrets or environment variables."
        )

    # 3. Validate and load uploaded images.
    image_parts = _normalize_uploaded_files(uploaded_files)

    # 4. Build the user prompt. The system instruction is also attached
    #    via GenerateContentConfig(system_instruction=...) inside
    #    _call_gemini for redundancy.
    prompt = build_generation_prompt(
        target_stack=stack,
        extra_instructions=extra_instructions or "",
    )

    # 5. Call Gemini and get raw text.
    text = _call_gemini(
        api_key=api_key,
        prompt=prompt,
        images=image_parts,
        model_name=(model_name or DEFAULT_MODEL).strip() or DEFAULT_MODEL,
    )

    # 6. Robustly extract JSON.
    raw_payload = extract_json_from_text(text)

    # 7. Strictly validate the schema. Any failure here surfaces as a
    #    ValueError so the UI can present a clear, actionable message.
    #    We never silently drop invalid file entries — bad input fails.
    ok, message = validate_generated_result(raw_payload)
    if not ok:
        raise ValueError(
            f"Invalid response format: {message} "
            f"Response preview: {_preview(text)!r}"
        )

    # 8. Light normalization (whitespace, stack-case, list coercion).
    return _normalize_payload(raw_payload, expected_stack=stack)
