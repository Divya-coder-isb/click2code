# Mockscreen to Code Agent

Turn UI mock screen images into a runnable starter project in one click.
Upload 1-3 mock screens, pick a target stack (Streamlit / React / Flask),
and Google Gemini writes the code for you — returned as strict JSON,
packaged into a downloadable ZIP, and previewed in-browser.

> Built for hackathon-speed demos. Streamlit on the outside, Gemini on the
> inside, plain stdlib for everything in between.

---

## Use case

You have a Figma export, a hand-drawn wireframe, or a screenshot from a
competitor and you want a working starter project in seconds. Instead of
opening a blank editor and typing boilerplate, you:

1. Drop the image(s) into the app.
2. Pick the framework you want.
3. Click **Generate Code**.
4. Download a clean, runnable ZIP.

Designers can prototype in any tool they like; engineers get a
ready-to-extend codebase to build on.

---

## Features

- **Multimodal generation** — sends the mock images plus a strict-JSON
  prompt to Gemini and parses the response.
- **Three target stacks** — Streamlit, React (Vite + JavaScript), Flask.
- **Up to 3 images** — combine multiple screens / states into one project.
- **Extra instructions** — colors, copy, naming hints get passed verbatim.
- **Robust JSON extraction** — handles raw JSON, ```` ```json ``` ```` fences,
  generic fences, and JSON embedded in prose.
- **Path safety** — absolute paths, `..` traversal, drive letters, and
  duplicate paths are rejected.
- **In-browser preview** — every generated file is shown in an expandable
  `st.code` block with syntax highlighting.
- **ZIP download** — files are nested under a sanitized project folder
  and an auto-generated `README_GENERATED.md` is included.
- **No code execution** — generated code is only displayed and zipped.
  Nothing runs in the app.
- **Session memory** — the most recent result persists across reruns.

---

## Tech stack

| Layer | Tech |
|---|---|
| UI | [Streamlit](https://streamlit.io) |
| Model | [Google Gemini](https://ai.google.dev) via the `google-genai` Python SDK |
| Image rendering | Pillow (via Streamlit) |
| Everything else | Python standard library (`io`, `re`, `json`, `zipfile`, `pathlib`, `dataclasses`, `typing`) |

Only three third-party packages. No databases, no message queues, no
background workers.

---

## Architecture

```
┌─────────────────┐    images + stack + notes    ┌──────────────────┐
│   Streamlit UI  │ ──────────────────────────▶  │  code_generator  │
│     (app.py)    │                              │     .py          │
└─────────────────┘                              └────────┬─────────┘
        ▲                                                 │
        │                                                 │ prompt + image parts
        │                                                 ▼
        │                                       ┌──────────────────┐
        │  dict {project_name, stack, files,    │ google-genai SDK │
        │        summary, run_instructions,     │   (Gemini API)   │
        │        notes}                          └────────┬─────────┘
        │                                                 │
        │                                                 │ strict JSON
        │                                                 ▼
        │                                       ┌──────────────────┐
        │                                       │  extract_json    │
        │                                       │  + validators    │
        │                                       └────────┬─────────┘
        │                                                │
        │                                                ▼
┌──────────────────┐    project dict    ┌────────────────────────┐
│  file_builder.py │ ◀────────────────  │   validated payload    │
│  → ZIP + README  │                    └────────────────────────┘
└──────────────────┘
```

Pipeline in plain English:

1. The user uploads 1-3 mock images and picks a stack in `app.py`.
2. `validators.py` checks count, extension, size, and JSON shape.
3. `prompt_templates.build_generation_prompt(...)` assembles the prompt
   (role + analysis tasks + stack-specific rules + JSON schema + strict
   validation rules).
4. `code_generator.generate_code_from_images(...)` resolves the API key
   from `st.secrets` or `os.environ`, sends prompt + inline image parts
   to Gemini with `response_mime_type="application/json"`, then uses
   `extract_json(...)` to recover the JSON payload.
5. `file_builder.create_zip_from_files(...)` writes every file to an
   in-memory ZIP with path-traversal protection and appends a
   `README_GENERATED.md`.
6. `app.py` renders the summary, file previews, and a ZIP download
   button. Nothing is ever executed.

---

## Folder structure

```
.
├── app.py                         # Streamlit UI
├── code_generator.py              # Gemini client + extract_json
├── prompt_templates.py            # Prompt builder + JSON schema
├── file_builder.py                # ZIP builder + path safety
├── validators.py                  # Upload + result validation
├── requirements.txt               # streamlit, google-genai, Pillow
├── README.md                      # You are here
├── .gitignore
├── .streamlit/
│   └── secrets.toml.example       # Copy to secrets.toml and fill in key
├── generated_outputs/             # Optional cache (git-ignored)
│   └── .gitkeep
└── assets/                        # Sample mocks, screenshots, etc.
    └── .gitkeep
```

---

## Setup steps

### 1. Clone the repo

```bash
git clone <your-fork-url> mockscreen-to-code-agent
cd mockscreen-to-code-agent
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate     # Windows PowerShell: .venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

That's it. Three packages, no system libraries.

---

## How to get a Gemini API key

1. Go to <https://aistudio.google.com/app/apikey>.
2. Sign in with your Google account.
3. Click **Create API key**.
4. Copy the key (it looks like `AIza...`).

The free tier is plenty for hackathon use. Treat the key like a password.

---

## How to configure local Streamlit secrets

You have two equivalent options. The app tries Streamlit secrets first,
then falls back to the environment variable.

### Option A — Streamlit secrets file (recommended for local dev)

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Open `.streamlit/secrets.toml` and replace the placeholder:

```toml
GEMINI_API_KEY = "AIza...your-real-key..."
```

`.streamlit/secrets.toml` is git-ignored and will not be committed.

### Option B — Environment variable

```bash
export GEMINI_API_KEY="AIza...your-real-key..."          # macOS / Linux
$env:GEMINI_API_KEY = "AIza...your-real-key..."          # Windows PowerShell
setx GEMINI_API_KEY "AIza...your-real-key..."            # Windows persistent
```

---

## How to run locally

```bash
streamlit run app.py
```

Streamlit prints a URL (usually <http://localhost:8501>). Open it in
your browser.

If the sidebar shows **Gemini API key not configured**, double-check
your secrets file or environment variable and reload the page.

---

## How to deploy on Streamlit Community Cloud

1. Push this repo to GitHub (public or private).
2. Go to <https://streamlit.io/cloud> and sign in with GitHub.
3. Click **New app** and pick your repo / branch / `app.py`.
4. Click **Advanced settings** → **Secrets** and paste:

   ```toml
   GEMINI_API_KEY = "AIza...your-real-key..."
   ```

5. Click **Deploy**. The first build takes 1-2 minutes.

Other GitHub-based hosts (Render, Railway, Fly.io, Hugging Face Spaces)
work the same way: set `GEMINI_API_KEY` as a secret env var and run
`streamlit run app.py`.

---

## How to use the app

1. **Check the sidebar.** A green ✅ next to "Gemini API key detected"
   means you're good to go.
2. **Pick the target stack** — Streamlit, React, or Flask.
3. **Pick the model** — defaults to `gemini-2.5-flash`. Pick `Custom...`
   to type any model id you want.
4. *(Optional)* **Type extra instructions** — e.g. *"Use a dark theme,
   brand color #1F6FEB, form on the right, keep copy concise."*
5. **Upload 1-3 mock screens** — png, jpg, jpeg, or webp. Thumbnails
   appear immediately.
6. Click **Generate Code**. A spinner shows while Gemini works.
7. **Review the result** — summary, file list, expandable file
   contents, run instructions, and notes.
8. Click **Download ZIP** to get the project. Unzip and run.

---

## Supported target stacks

| Stack | Entry file | Files generated | Run command |
|---|---|---|---|
| **Streamlit** | `app.py` | `app.py`, `requirements.txt` | `streamlit run app.py` |
| **React** (Vite) | `src/App.jsx` | `package.json`, `index.html`, `src/main.jsx`, `src/App.jsx`, `src/App.css` | `npm install && npm run dev` |
| **Flask** | `app.py` | `app.py`, `requirements.txt`, `templates/index.html`, `static/styles.css` | `python app.py` |

React generation defaults to plain JavaScript (no TypeScript) and plain
CSS (no Tailwind / Bootstrap), unless you ask for them in the extra
instructions.

---

## Troubleshooting

### "Gemini API key is missing. Add GEMINI_API_KEY to Streamlit secrets or environment variables."

- **Locally:** confirm `.streamlit/secrets.toml` exists (not just the
  `.example` file) and contains `GEMINI_API_KEY = "..."`. Or set the
  `GEMINI_API_KEY` environment variable in the shell where you run
  `streamlit run app.py`.
- **Streamlit Community Cloud:** open your app → **Manage app** →
  **Settings** → **Secrets** and paste:

  ```toml
  GEMINI_API_KEY = "your-key-here"
  ```

  Restart / redeploy after saving. The sidebar should flip to "Gemini
  API key detected".

### Gemini quota / rate-limit errors

If you see messages like `429 RESOURCE_EXHAUSTED`, `quota exceeded`, or
`Too many requests`:

- Wait 30-60 seconds and try again. The free tier of `gemini-2.5-flash`
  has per-minute and per-day caps.
- Switch to `gemini-2.0-flash` in the sidebar — it's lighter and often
  rate-limited separately.
- Reduce image size and count to lower your token usage per call.
- Add billing to your Google AI Studio project if you need higher
  sustained limits.

### "Could not parse JSON from Gemini response" / "Invalid response format"

The model returned text we couldn't validate. The error message includes
a short preview of the response. Mitigations:

- Click **Generate Code** again — Gemini is non-deterministic and the
  second attempt usually succeeds.
- Simplify the mock(s): one screen at a time, fewer fine-grained
  components, higher contrast.
- Switch model to `gemini-2.5-pro` — slower, but more reliable for
  complex JSON schemas.
- Trim **Extra instructions** down to one or two short bullets so the
  model is not pulled in too many directions at once.

### "Image too large" / upload rejected

- Each image is capped at the file uploader's limit (defaults to ~200 MB
  on Streamlit but we recommend keeping individual images under ~8 MB).
- Re-export the mock as JPG or WEBP at 1920×1080 or smaller.
- Crop to the relevant region — a tightly framed mock generates better
  code than a full-screen 4K capture.

### Generated React app doesn't run

After unzipping and running `npm install && npm run dev`:

- **`vite: command not found`** — run `npm install` again (sometimes
  the first run aborts if you're behind a proxy). Verify `vite` is in
  `package.json` under `devDependencies`.
- **`Module not found: 'react-dom/client'`** — Gemini may have pinned
  an old React. Open `package.json` and ensure `react` and `react-dom`
  are `^18.0.0` or newer.
- **Blank white page in the browser** — open DevTools → Console. If
  you see `process is not defined` or similar, the model emitted
  Node-only globals. Remove those references or regenerate the project.
- **CSS not applying** — verify `src/main.jsx` imports `./App.css` (or
  whatever stylesheet was emitted).

### Generated Flask app doesn't run

After unzipping and running `python app.py`:

- **`ModuleNotFoundError: No module named 'flask'`** — install
  dependencies with `pip install -r requirements.txt` from the
  unzipped project's folder.
- **`TemplateNotFound: index.html`** — Gemini referenced a template
  that wasn't emitted. Open the ZIP and confirm `templates/index.html`
  exists. If not, regenerate.
- **`Address already in use`** — another process is on port 5000. Run
  `python app.py --port 5001` or change `app.run(port=...)` in
  `app.py`.
- **CSS not applying** — confirm `static/styles.css` exists and that
  the HTML uses
  `<link rel="stylesheet" href="{{ url_for('static', filename='styles.css') }}">`.

---

## Security notes

- **Never commit secrets.** `.streamlit/secrets.toml` and `.env` are in
  `.gitignore`. The Gemini API key is read from `st.secrets` or the
  `GEMINI_API_KEY` environment variable — never hardcoded.
- **The app does not execute generated code.** It only displays it and
  zips it. The generation pipeline also rejects unsafe file paths
  (absolute, `..` traversal, drive letters, duplicates) before they
  reach the ZIP.
- **Review generated code before running it in production.** Gemini is
  an LLM. It can hallucinate imports, mis-license snippets, or pull in
  dependencies you don't want. Treat its output like any contractor's
  first draft — read it, run it locally, audit dependencies, and run
  your usual static-analysis / lint / test suite before shipping.
- **Don't paste customer data into mock screens.** The images are sent
  to Google's Gemini API. Stick to non-sensitive UI mocks for the demo.

---

## Future enhancements

Ideas we'd ship next if we had more than a hackathon weekend:

- **Multi-page generation** — chain multiple Gemini calls so each screen
  becomes its own route / page, sharing a common layout.
- **Design-system selection** — let users pick a starting design system
  (shadcn/ui, MUI, Chakra, Streamlit-native, plain CSS) and tailor the
  prompt accordingly.
- **Figma import** — pull frames straight from the Figma API instead of
  asking the user to export PNGs.
- **Tailwind support** — first-class Tailwind output for the React stack
  (and Tailwind-via-CDN for Flask templates).
- **Live preview sandbox** — render the generated app inside a sandboxed
  iframe (StackBlitz / WebContainer / Pyodide for Streamlit) so the
  user can see it run before downloading.
- **GitHub PR creation** — one click to push the generated project to a
  new branch and open a pull request, with the mock images attached.

---

## Hackathon demo script

Roughly 90 seconds. Adjust to your time slot.

> **0:00 — Hook.**
> *"Designers send mocks. Engineers stare at a blank file. We bridge
> that gap in one click with Google Gemini."*

> **0:10 — Show the UI.**
> *"Sidebar: API key detected, pick a stack — Streamlit, React, or
> Flask. Pick a model — `gemini-2.5-flash` by default. Drop in extra
> instructions: dark theme, brand color #1F6FEB."*

> **0:25 — Upload mocks.**
> *Drag two mock screens into the uploader. Thumbnails appear.*
> *"Up to three screens — Gemini treats them as different states of
> the same product."*

> **0:35 — Generate.**
> *Click* **Generate Code**. *Spinner. ~10-15 seconds.*

> **0:50 — Walk the result.**
> *"Project name. Summary. Files tab — `app.py`, `requirements.txt`,
> expand to see real code with syntax highlighting. Run instructions
> tab. Notes tab — assumptions Gemini made."*

> **1:10 — Download and run.**
> *Click* **Download ZIP**. *Unzip. `streamlit run app.py` in a
> terminal. App opens. Looks like the mock.*

> **1:25 — Close.**
> *"Strict JSON contract, path-traversal protection, no code execution
> inside the app, three deps total. Future: Figma import, design
> systems, one-click PRs. That's it."*

---

## License

MIT — do whatever you want, no warranty.
