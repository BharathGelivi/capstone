import os
import sys
import glob
import json
import logging

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.runner import PipelineRunner
from src.rag_trace import RAGTrace
from src.report_presenter import DiagnosticReportPresenter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_latest_trace():
    trace_dir = os.path.join(project_root, "artifacts", "rag_traces")
    date_dirs = glob.glob(os.path.join(trace_dir, "*"))
    if not date_dirs:
        return None
        
    date_dirs.sort(reverse=True)
    for d in date_dirs:
        if os.path.isdir(d):
            files = glob.glob(os.path.join(d, "*.json"))
            if files:
                files.sort(key=os.path.getmtime, reverse=True)
                return files[0]
    return None

def run_test():
    print("-" * 50)
    print("TEST: RUNNING EVALUATION FRAMEWORK ON TRACE")
    print("-" * 50)
    
    trace_file = get_latest_trace()
    if not trace_file:
        print("No RAG Trace found. Please generate one first.")
        return
        
    print(f"Loading trace from: {trace_file}")
    with open(trace_file, 'r', encoding='utf-8') as f:
        trace_data = f.read()
    
    trace = RAGTrace.from_json(trace_data)
    
    runner = PipelineRunner()
    
    try:
        report = runner.run(trace)
        
        presenter = DiagnosticReportPresenter(report)
        output = "\n" + presenter.render_console()
        
        try:
            print(output)
        except UnicodeEncodeError:
            # Fallback for Windows consoles that don't support UTF-8 characters like checkmarks
            print(output.encode('ascii', 'replace').decode('ascii'))
        
        # Optionally save the report as well
        report_path = os.path.join(project_root, "artifacts", "diagnostic_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report.to_json())
        print(f"\nDiagnostic report saved to {report_path}")
        
    except Exception as e:
        logger.error(f"Failed to run the framework: {e}", exc_info=True)

if __name__ == "__main__":
    run_test()
