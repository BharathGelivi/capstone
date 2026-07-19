# X-RAG Diagnostic Framework

## Project Overview

The X-RAG Diagnostic Framework is a comprehensive evaluation tool designed to diagnose, trace, and recommend fixes for Retrieval-Augmented Generation (RAG) pipelines. It explicitly isolates execution from reasoning, allowing deterministic analysis of where exactly a pipeline failed (e.g., retrieval miss vs. hallucination vs. unsupported claim).

## Architecture Diagram

```
Client
      ↓
FastAPI (API Layer)
      ↓
X-RAG Diagnostic Framework
      ↓
Artifacts
      ↓
Reports
```

## Repository Layout

- `src/`: Core framework and API source code.
- `tests/`: Unit test suite.
- `configs/`: Modular configuration properties.
- `artifacts/`: Generated traces, verification results, and reports.
- `data/`: Sample input files and source datasets.
- `eval/`: Labeled evaluation dataset used for baseline benchmark comparisons.
- `scripts/`: Standalone tools (threshold calibration, baseline comparison benchmark).
- `docs/`: Research logs and architectural decisions.

## Installation

1. Clone the repository.
2. Ensure you have Python 3.9+ installed.
3. Install dependencies from requirements:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and fill in necessary API keys.

### Optional: Baseline Comparison Benchmark

Comparing X-RAG against RAGAS/RAGChecker/ARES (see `docs/RESEARCH_LOG.md` for
the full methodology and investigation) needs extra, heavier dependencies not
required for the core pipeline, split across **three separate virtual
environments** -- `ragchecker` and `ares-ai` have pinned dependencies that
conflict with each other and with this project's own Python 3.13 venv, so each
gets its own **Python 3.10** venv. No Rust or C/C++ compiler install is needed
for any of them -- read `docs/RESEARCH_LOG.md` for how each blocker that
initially looked like it needed one was actually resolved (short version: old
pinned dependencies like `numpy<2.0`/`litellm==1.91.x` have prebuilt wheels for
Python 3.10, just not for 3.13).

1. **ragas** installs into this project's *main* venv:
   ```bash
   pip install -r requirements-eval.txt
   ```

   Also read `requirements-eval.txt` first -- it documents a required local
   compatibility shim for `ragas` (an upstream packaging issue, not something
   this project can fix).
2. **ragchecker** needs its own Python 3.10 venv:
   ```bash
   py -3.10 -m venv venv_eval_ragchecker
   venv_eval_ragchecker\Scripts\pip install -r requirements-eval-ragchecker.txt
   ```
3. **ares-ai** needs its own, separate Python 3.10 venv:
   ```bash
   py -3.10 -m venv venv_eval_ares
   venv_eval_ares\Scripts\pip install -r requirements-eval-ares.txt
   ```

   This one reliably fails on its *first* run with a Windows long-path
   `OSError` on a jupyterlab asset -- just run the same install command again
   and it completes using the cached downloads.

The comparison (and the core pipeline's own generation/claim decomposition)
uses Groq's free tier as the default LLM provider (`configs/models.py`
`LLM_PROVIDER = "groq"`) -- get a free key at https://console.groq.com/keys
and put it in `.env` as `GROQ_API_KEY=...`. No paid OpenAI/Bedrock key is
required. Switch `LLM_PROVIDER` back to `"huggingface"` in
`configs/models.py` (and set `HF_TOKEN`) once HF Inference Providers credits
are purchased/renewed -- no other code changes needed.

**Running the comparison** (after the environments above are set up):

```bash
# 1. Run all examples in eval/eval_dataset.csv through X-RAG + all three baselines.
#    Resumable: safe to re-run after an interruption or an HF credit/rate-limit error --
#    it skips eval rows already recorded in the manifest.
python -m scripts.run_baseline_comparison

# Preview what would run without calling any model or LLM API:
python -m scripts.run_baseline_comparison --dry-run

# 2. Compute cross-framework correlation, agreement (Cohen's kappa), and disagreements.
python -m scripts.analyze_agreement

# 3. Render the self-contained paper-ready report.
python -m scripts.generate_comparison_report
```

Outputs land in `artifacts/benchmark_comparison/`: `results.json`/`results.csv`
(per-example scores), `correlations.json`, `agreement.json`,
`disagreements.csv`, `failures.log`, and finally `comparison_report.md` -- the
one file to read for the full methodology and findings.

**Per-example detailed report** -- a single self-contained HTML page for one
trace (full RAG trace, claim decomposition, chunk-level provenance, root
cause reasoning, and a side-by-side comparison against real RAGAS/RAGChecker/
ARES scores for that same trace, each heavy section tucked into a collapsible
`<details>` disclosure so the page opens compact):

```bash
python -m scripts.generate_diagnostic_report --trace-id <trace_id>
```

Writes to `artifacts/diagnostic_reports/<trace_id>.html`. Requires the trace
to have already been run once through `run_baseline_comparison.py` (or
`query.py`) so `artifacts/reports/<trace_id>.json` and the trace file exist;
the RAGAS/RAGChecker/ARES comparison section is included automatically if a
matching row exists in `artifacts/benchmark_comparison/results.json`.

## Configuration

Configuration variables are located in the `configs/` directory:

- `models.py`: Specify embedding and language models.
- `pipeline.py`: Configure chunk size, overlap, and retrieval thresholds.
- `thresholds.py`: Adjust NLI evaluation thresholds.
- `api.py`: Configure API ports and hosts.

## Running the Pipeline (CLI)

1. Ingest documents from `data/` into the chunk registry and vector store (run this first, and again any time the source PDFs change):
   ```bash
   python run_pipeline.py
   ```
2. Ask a question against the ingested corpus and run the full diagnostic pipeline (retrieval, generation, claim decomposition, verification, root cause analysis, corrective actions, and a timestamped PDF report):
   ```bash
   python query.py "your question here"
   ```

   If `HF_TOKEN` isn't set in `.env` or the environment, you'll be prompted for it interactively.

## Running Unit Tests

To verify the installation and the diagnostic framework integrity:

```bash
python -m unittest discover tests
```

## Running FastAPI

To launch the API server locally:

```bash
python run_api.py
```

The server will start on `http://127.0.0.1:8000`.

## Swagger Documentation

Once the API is running, access the interactive auto-generated Swagger UI at:
`http://127.0.0.1:8000/docs`

## Example API Calls

**Healthcheck:**

```bash
curl http://127.0.0.1:8000/health
```

**Analyze a RAGTrace:**

```bash
curl -X POST http://127.0.0.1:8000/analyze \
     -H 'Content-Type: application/json' \
     -d @artifacts/rag_traces/TRACE_123.json
```

## Example Diagnostic Report

The `DiagnosticEvaluationReport` represents the final analysis output. Rendered formats are available via:

- `GET /report/{trace_id}/markdown`
- `GET /report/{trace_id}/html`

## Future Work

- Integration with major RAG deployment frameworks (LlamaIndex, LangChain).
- Enhanced feedback loops directly returning Corrective Action Plans to the generation model.
- Dashboard UI for real-time trace viewing.

## Open Source Contribution Guide

We welcome community contributions! Please review `CONTRIBUTING.md` for guidelines on coding standards, folder structure, adding new diagnostic modules, and submitting Pull Requests.


`python -m scripts.run_baseline_comparison`
