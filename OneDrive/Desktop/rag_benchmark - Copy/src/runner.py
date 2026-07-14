import os
from src.logger import get_logger
from src.rag_trace import RAGTrace
from src.claim_decomposer import ClaimDecomposer
from src.claim_verifier import ClaimVerifier
from src.pipeline_state_analyzer import PipelineStateAnalyzer
from src.root_cause_reasoner import RootCauseReasoner
from src.corrective_action_engine import CorrectiveActionEngine
from src.report_builder import ReportBuilder
from src.report import DiagnosticEvaluationReport

logger = get_logger(__name__)

class PipelineRunner:
    """
    Orchestrates the execution of the entire diagnostic pipeline.
    """
    def __init__(self):
        self.decomposer = ClaimDecomposer()
        self.verifier = ClaimVerifier()
        self.analyzer = PipelineStateAnalyzer()
        self.reasoner = RootCauseReasoner()
        self.cae = CorrectiveActionEngine()
        self.report_builder = ReportBuilder()

    def run(self, trace: RAGTrace) -> DiagnosticEvaluationReport:
        """
        Executes the diagnostic pipeline on a given RAGTrace.
        """
        logger.info(f"Starting pipeline run for Trace ID: {trace.trace_id}")
        
        # 1. Claim Decomposition
        logger.info("Running Claim Decomposer...")
        claim_set = self.decomposer.decompose(trace)
        
        # 2. Claim Verification
        logger.info("Running Claim Verifier...")
        verification = self.verifier.verify(trace, claim_set)
        
        # 3. Pipeline State Analyzer
        logger.info("Running Pipeline State Analyzer...")
        psm = self.analyzer.analyze(trace, claim_set, verification)
        
        # 4. Root Cause Reasoner
        logger.info("Running Root Cause Reasoner...")
        rca = self.reasoner.analyze(psm)
        
        # 5. Corrective Action Engine
        logger.info("Running Corrective Action Engine...")
        cap = self.cae.generate(rca)
        
        # 6. Report Builder
        logger.info("Running Report Builder...")
        report = self.report_builder.build(
            trace=trace,
            psm=psm,
            rca=rca,
            cap=cap,
            verification=verification
        )
        
        logger.info(f"Pipeline run completed for Trace ID: {trace.trace_id}")
        return report
