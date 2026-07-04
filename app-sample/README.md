# dgml-app-sample

A **sample** end-to-end app for DGML — not a production tool, but a demo meant to make the format's four layers tangible and to inspire what can be built on top of it.

Two files, no build step:

- [`dgml-app-sample.html`](dgml-app-sample.html) — the frontend: markup, CSS, and JS in one file.
- [`dgml-app-sample-server.py`](dgml-app-sample-server.py) — an optional stdlib-only Python HTTP server (no pip dependencies) that adds the full pipeline and real chain anchoring on top.

## Start with the HTML

Open [`dgml-app-sample.html`](dgml-app-sample.html) directly in a browser — `file://`, no server needed — and point it at an existing DGML workspace to see the four layers of the format in action:

1. **Semantic** — explore the semantic elements discovered in each document, and how files were grouped into docsets.
2. **Spatial** — see how every element shows exactly where it came from on the source page (`dg:origin`).
3. **Attestation** — click an element to see how it could be anchored on-chain — a simple, concrete illustration of element-level Proof of Origin.
4. **Readable** — for the plainest illustration of this layer, try [`tools/dgml2html`](../tools/dgml2html) instead: a tiny standalone script that renders a `.dgml.xml` file as a styled HTML page.

## Run the server for the full pipeline

`dgml-app-sample-server.py` turns the same page into a driver for the whole toolchain:

```bash
python app-sample/dgml-app-sample-server.py [--workspace /path/to/workspace] [--port 5173]
```

Then open `http://localhost:5173`. Workspace root resolution: `--workspace` flag → `DGML_HOME` env var → `./dgml-workspace`.

From the UI you can:

- Pick a source folder of PDFs and a workspace folder (native OS directory picker)
- Set an LLM model and API key for classification/generation
- Run the pipeline end to end — `init` → `file add` → `cluster` → assign any unclustered files to their own docset → `docset generate` for each docset — and watch live step/log events
- Exercise chain anchoring for real: stake a document or a single element on an actual chain (e.g. the NVNM testnet, see [`get-started`](../get-started)) and verify the resulting proof, not just simulate the click

The server shells out to the `dgml` CLI (`uv run dgml` by default; override with the `DGML_CMD` env var), so it expects to be run from an environment where the workspace is set up per the main [`README`](../README.md) / [`get-started`](../get-started) instructions.

## Not for production

This is a demo, not a hardened server: it binds to a local port with `Access-Control-Allow-Origin: *`, has no auth, and forwards whatever API key you type in the browser to whichever provider URL you configure. Run it locally against your own workspace only.
