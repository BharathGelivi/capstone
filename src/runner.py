import os
from typing import Optional
from src.logger import get_logger
from src.rag_trace import RAGTrace
from src.claim_decomposer import ClaimDecomposer
from src.claim_verifier import ClaimVerifier
from src.pipeline_state_analyzer import PipelineStateAnalyzer
from src.root_cause_reasoner import RootCauseReasoner
from src.corrective_action_engine import CorrectiveActionEngine
from src.report_builder import ReportBuilder
from src.report import DiagnosticEvaluationReport
from src.answer_correctness_evaluator import AnswerCorrectnessEvaluator

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
        # Reuses this runner's own decomposer/verifier instances (same NLI
        # model, same LLM client) rather than loading a second copy of each.
        self.answer_correctness_evaluator = AnswerCorrectnessEvaluator(
            decomposer=self.decomposer, verifier=self.verifier
        )

    def run(self, trace: RAGTrace, gold_answer: Optional[str] = None) -> DiagnosticEvaluationReport:
        """
        Executes the diagnostic pipeline on a given RAGTrace.

        gold_answer: optional reference answer. When provided, also computes
        answer correctness (claim recall) -- see src/answer_correctness_evaluator.py.
        This is purely additive: it does not affect Overall Health, primary_cause,
        or any existing pipeline stage.
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
        cap = self.cae.generate(rca, psm=psm)

        # 5.5 Answer Correctness (Claim Recall) -- optional, only when a gold answer is available
        answer_correctness = None
        if gold_answer:
            logger.info("Running Answer Correctness Evaluator...")
            answer_correctness = self.answer_correctness_evaluator.evaluate(
                trace.generated_answer, gold_answer, trace.trace_id
            )
            answer_correctness.save()

        # 6. Report Builder
        logger.info("Running Report Builder...")
        report = self.report_builder.build(
            trace=trace,
            psm=psm,
            rca=rca,
            cap=cap,
            verification=verification,
            answer_correctness=answer_correctness
        )

        logger.info(f"Pipeline run completed for Trace ID: {trace.trace_id}")
        return report
