"""Prompt templates for mockscreen-to-code-agent.

This module owns the natural-language contract we hand to Gemini. The
primary export is `build_generation_prompt`, which composes a single
self-contained prompt string that:

- Declares Gemini's role (senior UI engineer / frontend dev / UX reviewer /
  code generator).
- Tells it how to analyze the uploaded mock screen image(s).
- Lays out generation rules and stack-specific file expectations.
- Specifies the exact JSON output schema.
- Enumerates strict validation rules the response must satisfy.

`SYSTEM_INSTRUCTION` is also exported for callers that want to pass a
separate `system_instruction` to the model config (the role declaration is
intentionally included in both places so the model stays in character even
if a particular SDK call ignores `system_instruction`).
"""

from __future__ import annotations

from typing import Dict, List


# ---------------------------------------------------------------------------
# Allowed stacks
# ---------------------------------------------------------------------------

SUPPORTED_STACKS: List[str] = ["streamlit", "react", "flask"]


# ---------------------------------------------------------------------------
# System instruction (also embedded into the user prompt for redundancy)
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTION = (
    "You are simultaneously a senior UI engineer, a frontend developer, a "
    "UX reviewer, and a code generator. You convert UI mock screen images "
    "into clean, runnable starter projects. You always return STRICT JSON "
    "matching the schema you are given. You never invent API keys, never "
    "include binary blobs, and never reference files you did not also emit."
)


# ---------------------------------------------------------------------------
# Stack-specific file requirements
# ---------------------------------------------------------------------------

STACK_REQUIREMENTS: Dict[str, str] = {
    "streamlit": (
        "For **Streamlit** (Python):\n"
        "- Emit `app.py` (entry point) and `requirements.txt`.\n"
        "- Use Streamlit components, columns, containers, markdown, and "
        "widgets (st.columns, st.container, st.tabs, st.sidebar, st.markdown, "
        "st.button, st.text_input, st.selectbox, st.dataframe, st.image, etc.) "
        "to recreate the layout faithfully.\n"
        "- If standard widgets cannot express a visual detail, inject CSS "
        "through `st.markdown(\"<style>...</style>\", unsafe_allow_html=True)`. "
        "Keep injected CSS small and well-scoped.\n"
        "- `requirements.txt` must list every pip dependency with concrete "
        "version constraints (e.g. `streamlit>=1.36`).\n"
        "- Keep the app runnable with `streamlit run app.py`."
    ),
    "react": (
        "For **React** (Vite + JavaScript):\n"
        "- Emit `package.json`, `index.html`, `src/main.jsx`, `src/App.jsx`, "
        "and `src/App.css`. You may add a `vite.config.js` if needed.\n"
        "- Use Vite + React with functional components and hooks. "
        "**Avoid TypeScript unless the user explicitly asked for it.**\n"
        "- `package.json` must list `react`, `react-dom`, `vite`, and "
        "`@vitejs/plugin-react` with concrete versions, plus `scripts` for "
        "`dev`, `build`, and `preview`.\n"
        "- Style with plain CSS in `src/App.css`. Do not pull in Tailwind, "
        "Bootstrap, Material UI, or any framework unless the mock clearly "
        "demands it.\n"
        "- Keep the app runnable with `npm install` then `npm run dev`."
    ),
    "flask": (
        "For **Flask** (Python + Jinja templates):\n"
        "- Emit `app.py` (entry point exposing a Flask instance), "
        "`requirements.txt`, `templates/index.html`, and `static/styles.css`. "
        "Add additional templates/partials as needed.\n"
        "- `app.py` should use `render_template` and define at least one "
        "route. Use a `base.html` layout when multiple pages are needed.\n"
        "- Put HTML in `templates/` and CSS/JS in `static/`.\n"
        "- `requirements.txt` must pin `flask` to a concrete version.\n"
        "- Keep the app runnable with `python app.py`."
    ),
}


# ---------------------------------------------------------------------------
# JSON schema (verbatim, embedded into the prompt)
# ---------------------------------------------------------------------------

JSON_SCHEMA = """\
{
  "project_name": "string",
  "stack": "streamlit | react | flask",
  "summary": "string",
  "files": [
    {
      "path": "relative/path/to/file",
      "content": "complete file content"
    }
  ],
  "run_instructions": ["step 1", "step 2"],
  "notes": ["note 1", "note 2"]
}"""


# ---------------------------------------------------------------------------
# Strict output-format rules (shown as a stand-alone section in the prompt)
# ---------------------------------------------------------------------------

OUTPUT_FORMAT_RULES = """\
Output format — these rules are absolute. Violating any of them makes
your response useless to the caller, so treat them as hard constraints:

- **Return ONLY valid JSON. No markdown. No prose. No comments.**
- **Do NOT wrap the JSON in code fences** (no ```` ``` ````, no ```` ```json ````).
- **Do NOT add any text before or after the JSON object.** The very first
  character of your response MUST be `{` and the very last character
  MUST be `}`.
- **All newline characters inside file `content` MUST be escaped
  correctly by JSON serialization** (use `\\n`, never a literal newline
  inside a JSON string).
- **All double-quotes inside file `content` MUST be escaped as `\\"`.**
- **The response MUST be parseable by Python `json.loads` on the very
  first attempt** with no preprocessing on the caller's side.
- **No trailing commas. No JSON5. No comments inside the JSON.**
"""


# ---------------------------------------------------------------------------
# Strict validation rules (content-level constraints)
# ---------------------------------------------------------------------------

VALIDATION_RULES = """\
Content rules. Your response MUST satisfy ALL of these:

1. Every file MUST have a non-empty `path` AND non-empty `content`.
2. Do NOT return duplicate file paths.
3. Use **relative paths only**. No absolute paths (no leading `/`, no
   `C:\\`, no `~/`). Use forward slashes.
4. No `..` segments anywhere in any path.
5. Do NOT emit a `.env` file or any file containing real secrets, API
   keys, tokens, passwords, or credentials. If a config value is needed,
   use a placeholder like `your-api-key-here` inside a `.env.example`.
6. Do NOT include shell scripts that delete files or perform destructive
   operations (no `rm -rf`, no `del /F`, no `format`, no `dd`).
7. Do NOT include code that sends uploaded images, user data, or
   generated code to unknown third-party services. Any outbound network
   call must target a clearly-identified, free, reputable endpoint AND
   must be optional / disabled by default.
8. Every file referenced from another file (imports, `src`, `href`,
   template names, route targets) MUST also exist in the `files` array."""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_generation_prompt(
    target_stack: str,
    extra_instructions: str = "",
) -> str:
    """Compose the full user prompt sent alongside the mock screen image(s).

    Args:
        target_stack: One of `"streamlit"`, `"react"`, `"flask"`.
        extra_instructions: Optional free-form notes from the end user.
            Passed as soft guidance; hard rules above always win.

    Returns:
        A single self-contained prompt string. The caller is expected to
        send this string as the first content part of the multimodal
        request, followed by the inline image parts.

    Raises:
        ValueError: when `target_stack` is missing or unsupported.
    """
    if not isinstance(target_stack, str) or not target_stack.strip():
        raise ValueError("target_stack is required.")
    stack = target_stack.strip().lower()
    if stack not in STACK_REQUIREMENTS:
        raise ValueError(
            f"Unsupported target_stack {target_stack!r}. "
            f"Choose one of: {SUPPORTED_STACKS}."
        )

    stack_block = STACK_REQUIREMENTS[stack]

    extras_block = ""
    if extra_instructions and extra_instructions.strip():
        extras_block = (
            "\n## Extra instructions from the user\n"
            "Treat these as soft guidance. They never override the hard "
            "rules above.\n\n"
            f"{extra_instructions.strip()}\n"
        )

    return f"""\
# Role
You are simultaneously playing four roles for this task:
- **Senior UI engineer**
- **Frontend developer**
- **UX reviewer**
- **Code generator**

Behave like all four at once: design-aware, detail-oriented, and able to
ship runnable code.

# Task
Analyze the uploaded mock screen image(s) **carefully** and generate a
runnable starter project that recreates the UI for the target stack:
**{stack}**.

# What to infer from the mock(s)
For every uploaded image, extract:
- **Overall layout**: page hierarchy, grid, header / sidebar / footer,
  number of columns, content vs. chrome.
- **Typography**: heading vs. body weight contrast, sizing scale,
  alignment, font family hints (sans-serif / serif / mono).
- **Color palette**: background, surface, primary, secondary, accent,
  text, muted, success/danger if visible.
- **Spacing**: paddings, margins, gutters, vertical rhythm, density.
- **Components**: sections, cards, buttons (primary/secondary/ghost),
  forms, inputs, dropdowns, lists, tables, modals, badges, avatars.
- **Navigation**: top nav, sidebar nav, tabs, breadcrumbs, pagination.
- **Responsive behavior**: assume mobile-friendly defaults; stack
  columns on narrow widths; collapse navigation where appropriate.

If multiple mock images are uploaded, treat them as different screens or
states of the same product and produce a single coherent project.

# Generation rules
- Generate **clean, working code** for the **{stack}** stack only.
- Use only **simple, free dependencies** (Streamlit, Flask, React, Vite,
  plain CSS). **Do NOT use external paid assets** of any kind.
- Where exact images, icons, or fonts are not available, use
  **placeholders**: text labels, emoji glyphs, simple inline SVG, or
  `https://placehold.co/<W>x<H>` image URLs.
- Make the result **visually close to the mock screen**: colors,
  spacing, hierarchy, and primary controls should match.
- **Return only strict JSON** matching the schema below.
- **Do NOT wrap your output in markdown code fences.**
- **Do NOT include any explanation, preamble, or trailing text outside
  the JSON object.**
- **Use relative file paths only.**
- **Never include real secrets, API keys, tokens, or credentials.**
- Include actionable `run_instructions` so a developer can run the
  project end-to-end on a clean machine.
- Populate `notes` with anything the developer should know (assumptions
  you made, parts of the mock you could not infer, follow-up ideas).

# Stack-specific requirements
{stack_block}

# Output schema (the ONLY allowed shape)
Respond with exactly one JSON object matching this schema. No extra
top-level keys, no missing top-level keys.

{JSON_SCHEMA}

# {OUTPUT_FORMAT_RULES}

# {VALIDATION_RULES}
{extras_block}
# Final reminder (read carefully)
- Return **only valid JSON**. **No markdown. No prose. No comments.**
- The first character of your response MUST be `{{` and the last
  character MUST be `}}`.
- All newlines inside file `content` MUST be escaped as `\\n`, and all
  double-quotes inside file `content` MUST be escaped as `\\"`.
- The response MUST be parseable by **Python `json.loads`** as-is."""
