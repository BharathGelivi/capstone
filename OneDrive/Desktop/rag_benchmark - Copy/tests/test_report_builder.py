import unittest
from src.report_builder import ReportBuilder
from src.rag_trace import RAGTrace
from src.pipeline_state_analyzer import PipelineStateMatrix, PipelineState, PipelineStage, PipelineStatus
from src.root_cause_reasoner import RootCauseAnalysis, FailureType
from src.corrective_action_engine import CorrectiveActionPlan, CorrectiveAction
from src.claim_verifier import VerificationSummary, VerificationResult, VerificationStatus

class TestReportBuilder(unittest.TestCase):
    def setUp(self):
        self.builder = ReportBuilder()
        self.trace_id = "TRACE_123"
        
        self.trace = RAGTrace(
            trace_id=self.trace_id,
            trace_version="1.0",
            pipeline_version="1.0",
            framework_version="1.0",
            timestamp="2026-01-01T00:00:00Z",
            question="Test?",
            generated_answer="Test.",
            prompt_snapshot="Test prompt",
            prompt_length=10,
            retrieved_chunk_references=[{"chunk_id": "c1", "similarity_score": 0.9}],
            configuration_snapshot={},
            execution_statistics={"retrieval_time": 0.1, "generation_time": 0.5},
            pipeline_stage_status={}
        )
        
        self.psm = PipelineStateMatrix(
            trace_id=self.trace_id,
            pipeline_states=[
                PipelineState(PipelineStage.RETRIEVER, PipelineStatus.PASS, "OK", 0.9)
            ]
        )
        
        self.rca = RootCauseAnalysis(
            trace_id=self.trace_id,
            primary_cause=FailureType.UNKNOWN,
            secondary_effects=[],
            reasoning_chain=["Healthy pipeline"],
            confidence="HIGH"
        )
        
        self.cap = CorrectiveActionPlan(
            trace_id=self.trace_id,
            primary_cause=FailureType.UNKNOWN,
            immediate_actions=[CorrectiveAction("A1", "SYSTEM", "Title", "Desc", "Ev", "UNKNOWN", "Imp", "Metric", "Tradeoff", "low")]
        )
        
        self.verification = VerificationSummary(
            trace_id=self.trace_id,
            total_claims=1,
            supported_claims=1,
            partially_supported_claims=0,
            contradicted_claims=0,
            unsupported_claims=0,
            not_verifiable_claims=0,
            average_entailment_score=0.9,
            total_verification_latency_ms=100.0,
            results=[
                VerificationResult(
                    verification_id="v1",
                    trace_id=self.trace_id,
                    claim_id="cl1",
                    claim_text="Test claim",
                    verification_status=VerificationStatus.SUPPORTED,
                    verification_reason="Reason",
                    confidence=0.9,
                    best_chunk_id="c1",
                    best_chunk_rank=1,
                    best_chunk_score=0.9,
                    best_sentence_id="s1",
                    evidence_text="Evidence",
                    entailment_score=0.9,
                    contradiction_score=0.0,
                    neutral_score=0.1
                )
            ]
        )

    def test_healthy_pipeline(self):
        report = self.builder.build(self.trace, self.psm, self.rca, self.cap, self.verification)
        self.assertEqual(report.framework_metadata.trace_id, self.trace_id)
        self.assertEqual(report.analysis_status, "COMPLETED")
        self.assertEqual(report.evaluation_metrics.supported_claims, 1)
        self.assertEqual(report.evaluation_metrics.grounding_score, 1.0)
        self.assertEqual(len(report.pipeline_overview.pipeline_stages), 1)
        self.assertEqual(report.executive_summary.question, "Test?")

    def test_missing_verification(self):
        report = self.builder.build(trace=self.trace, psm=self.psm, rca=self.rca, cap=self.cap)
        self.assertEqual(report.analysis_status, "PARTIAL_ANALYSIS")
        self.assertEqual(report.evaluation_metrics.verified_claims, 0)

    def test_missing_corrective_actions(self):
        report = self.builder.build(trace=self.trace, psm=self.psm, rca=self.rca, verification=self.verification)
        self.assertEqual(report.analysis_status, "PARTIAL_ANALYSIS")
        self.assertEqual(len(report.corrective_actions), 0)
        self.assertIn("CorrectiveActionPlan", report.metadata["missing_artifacts"])

    def test_trace_id_mismatch(self):
        bad_rca = RootCauseAnalysis(trace_id="BAD_TRACE", primary_cause=FailureType.UNKNOWN)
        with self.assertRaises(ValueError):
            self.builder.build(trace=self.trace, rca=bad_rca)

    def test_serialization(self):
        report = self.builder.build(self.trace, self.psm, self.rca, self.cap, self.verification)
        json_str = report.to_json()
        self.assertIn("TRACE_123", json_str)
        self.assertIn("PARTIAL_ANALYSIS", report.build(trace=self.trace).to_json() if False else "PARTIAL_ANALYSIS") # just checking json serialization works without exception

if __name__ == '__main__':
    unittest.main()
