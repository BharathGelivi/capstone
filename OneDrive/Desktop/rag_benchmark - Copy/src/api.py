import os
import glob
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse, PlainTextResponse
from pydantic import BaseModel

from src.logger import get_logger
from src.rag_trace import RAGTrace
from src.runner import PipelineRunner
from src.report import DiagnosticEvaluationReport
from src.report_presenter import DiagnosticReportPresenter
from configs.api import API_VERSION, SUPPORTED_PIPELINE_VERSION

logger = get_logger(__name__)

app = FastAPI(
    title="X-RAG Diagnostic Framework API",
    version=API_VERSION,
    description="API for running diagnostic pipelines on RAG execution traces."
)

runner = PipelineRunner()

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal Server Error",
            "details": str(exc)
        }
    )

@app.get("/")
def get_root():
    return {
        "framework_name": "X-RAG Diagnostic Framework",
        "framework_version": API_VERSION,
        "status": "Online"
    }

@app.get("/health")
def get_health():
    return {
        "status": "healthy",
        "framework_version": API_VERSION
    }

@app.get("/version")
def get_version():
    return {
        "framework_version": API_VERSION,
        "artifact_versions": "1.0",
        "supported_pipeline_version": SUPPORTED_PIPELINE_VERSION
    }

@app.post("/analyze")
def analyze_trace(trace_data: dict):
    """
    Takes a RAGTrace JSON payload, runs the complete diagnostic pipeline,
    and returns a DiagnosticEvaluationReport JSON.
    """
    try:
        # Reconstruct trace from JSON payload
        trace = RAGTrace.from_json(trace_data)
    except Exception as e:
        logger.error(f"Failed to parse RAGTrace: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid RAGTrace format")
        
    try:
        report = runner.run(trace)
        
        # Save artifacts
        report.save()
        
        # We can also save the original trace since it was passed here
        os.makedirs("artifacts/rag_traces", exist_ok=True)
        with open(f"artifacts/rag_traces/{trace.trace_id}.json", "w") as f:
            f.write(trace.to_json())
            
        return report.__dict__
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Pipeline Analysis Failed",
                "details": str(e)
            }
        )

@app.get("/report/{trace_id}")
def get_report(trace_id: str):
    path = f"artifacts/reports/{trace_id}.json"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Report not found")
    report = DiagnosticEvaluationReport.load(path)
    return report.__dict__

@app.get("/report/{trace_id}/markdown")
def get_report_markdown(trace_id: str):
    path = f"artifacts/reports/{trace_id}.json"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Report not found")
    report = DiagnosticEvaluationReport.load(path)
    presenter = DiagnosticReportPresenter(report)
    return PlainTextResponse(presenter.render_markdown())

@app.get("/report/{trace_id}/html")
def get_report_html(trace_id: str):
    path = f"artifacts/reports/{trace_id}.json"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Report not found")
    report = DiagnosticEvaluationReport.load(path)
    presenter = DiagnosticReportPresenter(report)
    return HTMLResponse(presenter.render_html())

@app.get("/artifacts/{trace_id}")
def get_artifacts(trace_id: str):
    artifacts = {}
    base_dirs = {
        "RAGTrace": "artifacts/rag_traces",
        "ClaimSet": "artifacts/claim_sets",
        "Verification": "artifacts/verification",
        "PipelineStateMatrix": "artifacts/pipeline_state_matrix",
        "RootCauseAnalysis": "artifacts/root_cause_analysis",
        "CorrectiveActionPlan": "artifacts/corrective_action_plan",
        "DiagnosticEvaluationReport": "artifacts/reports"
    }
    
    for name, directory in base_dirs.items():
        path = f"{directory}/{trace_id}.json"
        if os.path.exists(path):
            artifacts[name] = path
            
    return {"trace_id": trace_id, "artifacts": artifacts}
