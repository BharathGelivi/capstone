# X-RAG Testing and Project Structure Guide

This guide details how to run evaluations/tests in the X-RAG framework and maps out the project structure to clarify what each component does. We will not delete any files without explicit approval, but this guide should help distinguish critical core files from optional research scripts.

---

## 1. Commands to Run Tests and Evaluations

### Ingest Data (Chunking and Embedding)
Before running any evaluations, you must ingest the source PDFs (located in the `data/` directory) into the chunk registry and vector database. Run this first, and run it again anytime the source documents change:
```powershell
venv\Scripts\python.exe run_pipeline.py
```

### Run a Single Example through X-RAG
To run a specific evaluation example (e.g., example 11) using the core diagnostic pipeline:
```powershell
venv\Scripts\python.exe -m scripts.run_baseline_comparison --limit 11 --skip-ragchecker --skip-ares
```
*Note: This generates `artifacts/diagnostic_reports/11.html` and saves the trace to `artifacts/rag_traces/`.*

### Re-run a Single Example with a Different Model
If the primary generator hits a rate limit (e.g., Groq's daily token limit), you can surgically re-run a specific example with a different model without affecting other results:
```powershell
venv\Scripts\python.exe -m scripts.rerun_example_with_model --eval-id 11 --model llama-3.1-8b-instant
```

### Run the Post-Analysis Metric Scripts
After running examples, execute the post-analysis scripts to compute aggregate metrics (diagnostic accuracy, calibration, cost/latency):
```powershell
venv\Scripts\python.exe -m scripts.evaluate_diagnostic_accuracy
venv\Scripts\python.exe -m scripts.evaluate_confidence_calibration
venv\Scripts\python.exe -m scripts.evaluate_reasoning_consistency
venv\Scripts\python.exe -m scripts.evaluate_cost_latency
```

### Run the Aggregation Ablation Test
To test whether the choice of evidence-aggregation strategy (`top1` / `max_pool_top3` / `concat_top3`) affects accuracy:
```powershell
venv\Scripts\python.exe -m scripts.ablate_aggregation_strategy
```

### Regenerate the Final Comparison Report
After running the post-analysis scripts, regenerate the aggregate Markdown report:
```powershell
venv\Scripts\python.exe -m scripts.generate_comparison_report
```
*The output will be available at `artifacts/benchmark_comparison/comparison_report.md`.*

---

## 2. Project File Structure & Purpose

Here is a map of the repository to help you understand what is critical and what is supplemental research tooling.

### 🧠 Core X-RAG Pipeline (`src/`)
These are the most critical files. They implement the RAG components and the X-RAG diagnostic capabilities.

* **Standard RAG Components:**
  * `retriever.py`, `generator.py`: Executes retrieval and LLM generation.
  * `vector_store.py`, `embedding_engine.py`, `chunk_engine.py`: Data ingestion, embedding, and storage.
* **X-RAG Diagnostic Engine:**
  * `claim_decomposer.py`: Breaks the generated answer into verifiable atomic claims using an LLM.
  * `claim_verifier.py`: Uses a local NLI model (DeBERTa) to verify each claim against retrieved chunks.
  * `root_cause_reasoner.py`: Analyzes the verification results and attributes failure to a specific pipeline stage (retrieval, chunking, etc.).
  * `pipeline_state_analyzer.py`, `corrective_action_engine.py`: Analyzes the trace state and suggests fixes.
* **Orchestration & Reporting:**
  * `runner.py`: Connects all components together into a single pipeline execution.
  * `rag_trace.py`: Data model for the execution trace.
  * `report.py`, `report_builder.py`, `report_presenter.py`: Formats the final diagnostic output (powers the HTML reports).

### 📊 Evaluation & Benchmarking (`scripts/`)
These scripts are used for evaluating X-RAG against baselines and running experiments.

* **Core Evaluation Runner:**
  * `run_baseline_comparison.py`: The main loop that runs X-RAG alongside RAGAS, RAGChecker, and ARES. Uses a `manifest.json` to track state.
* **Reporting:**
  * `generate_comparison_report.py`: Aggregates all JSON outputs into the final Markdown report.
  * `generate_diagnostic_report.py`: Helper script to render individual `trace.json` files into `.html` reports.
* **Research / Metric Scripts (Improvements 1-5):**
  * `evaluate_diagnostic_accuracy.py`: Validates if X-RAG's root-cause matches human labels.
  * `evaluate_confidence_calibration.py`: Checks if X-RAG's confidence score reflects empirical accuracy.
  * `evaluate_reasoning_consistency.py`: Automatically checks if X-RAG's reasoning chain contradicts itself.
  * `evaluate_cost_latency.py`: Analyzes wall-clock time and token usage.
  * `ablate_aggregation_strategy.py`: Experiment altering the claim verification threshold logic.
* **Utilities:**
  * `rerun_example_with_model.py`: Fast utility for rescuing rate-limited runs.
  * `baseline_adapters/`: Wrappers for RAGAS, RAGChecker, and ARES.

### ⚙️ Configuration (`configs/`)
* `models.py`: Central hub for defining which LLMs are used for generation vs decomposition, and which NLI model is used.
* `prompts.py`: System prompts for generation and decomposition.
* `thresholds.py`: Tuning values for when something is considered "Supported" or "Contradicted".

### 📂 Data & Artifacts
* `eval/eval_dataset.csv`: The "ground truth" 40-question dataset. *Do not modify or delete unless expanding the benchmark.*
* `artifacts/`: Automatically generated directory containing vector DBs (`db/chroma`), generated RAG traces (`rag_traces/`), and benchmarking outputs (`benchmark_comparison/`). This is safe to clear if you want a fully fresh run.

---

## What to Clean Up?
Everything currently serves a clear purpose for the research benchmark. The only things you might consider cleaning out in the future are the `artifacts/` subdirectories if you want to wipe all past runs and start a fresh benchmark. The `evaluate_*.py` scripts are specific to the research paper/report, but they are highly valuable for proving X-RAG's validity.
