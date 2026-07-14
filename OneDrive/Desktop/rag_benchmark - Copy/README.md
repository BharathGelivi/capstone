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
- `docs/`: Research logs and architectural decisions.

## Installation
1. Clone the repository.
2. Ensure you have Python 3.9+ installed.
3. Install dependencies from requirements:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and fill in necessary API keys.

## Configuration
Configuration variables are located in the `configs/` directory:
- `models.py`: Specify embedding and language models.
- `pipeline.py`: Configure chunk size, overlap, and retrieval thresholds.
- `thresholds.py`: Adjust NLI evaluation thresholds.
- `api.py`: Configure API ports and hosts.

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
