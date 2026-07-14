import unittest
import os
import tempfile
from src.rag_trace import RAGTrace
from src.claims import ClaimSet, Claim
from src.claim_verifier import VerificationSummary, VerificationResult, VerificationStatus
from src.pipeline_state_analyzer import PipelineStateAnalyzer, PipelineStage, PipelineStatus

class TestPipelineStateAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = PipelineStateAnalyzer(retrieval_score_threshold=0.5)

    def create_mock_trace(self, trace_id, max_score):
        return RAGTrace(
            trace_id=trace_id,
            trace_version="1.0",
            pipeline_version="1.0",
            framework_version="1.0",
            timestamp="2026-07-13T12:00:00Z",
            question="Test question",
            generated_answer="Test answer",
            prompt_snapshot="Prompt",
            prompt_length=100,
            retrieved_chunk_references=[{"chunk_id": "chunk1", "similarity_score": max_score}],
            configuration_snapshot={},
            execution_statistics={},
            pipeline_stage_status={}
        )

    def create_mock_claim_set(self, trace_id, num_claims):
        claim_set = ClaimSet(trace_id=trace_id)
        for i in range(num_claims):
            claim = Claim(
                claim_id=f"{trace_id}_C{i:03d}",
                trace_id=trace_id,
                claim_text=f"Claim {i}",
                source_sentence=f"Source {i}",
                sentence_id=f"S{i}",
                claim_index=i,
                character_start=0,
                character_end=10
            )
            claim_set.add_claim(claim)
        return claim_set

    def create_mock_verification(self, trace_id, statuses):
        results = []
        for i, status in enumerate(statuses):
            results.append(VerificationResult(
                verification_id=f"V{i}",
                trace_id=trace_id,
                claim_id=f"{trace_id}_C{i:03d}",
                claim_text=f"Claim {i}",
                verification_status=status,
                verification_reason="Reason",
                confidence=0.9,
                best_chunk_id="chunk1",
                best_chunk_rank=1,
                best_chunk_score=0.9,
                best_sentence_id="S1",
                evidence_text="Evidence",
                entailment_score=0.9 if status == VerificationStatus.SUPPORTED else 0.1,
                contradiction_score=0.1,
                neutral_score=0.1
            ))
        return VerificationSummary(
            trace_id=trace_id,
            total_claims=len(statuses),
            supported_claims=statuses.count(VerificationStatus.SUPPORTED),
            partially_supported_claims=statuses.count(VerificationStatus.PARTIALLY_SUPPORTED),
            contradicted_claims=statuses.count(VerificationStatus.CONTRADICTED),
            unsupported_claims=statuses.count(VerificationStatus.UNSUPPORTED),
            not_verifiable_claims=statuses.count(VerificationStatus.NOT_VERIFIABLE),
            average_entailment_score=0.5,
            total_verification_latency_ms=100.0,
            results=results
        )

    def test_healthy_pipeline(self):
        trace_id = "TRACE_HEALTHY"
        trace = self.create_mock_trace(trace_id, max_score=0.8)
        claim_set = self.create_mock_claim_set(trace_id, 2)
        verification = self.create_mock_verification(trace_id, [VerificationStatus.SUPPORTED, VerificationStatus.SUPPORTED])

        matrix = self.analyzer.analyze(trace, claim_set, verification)
        
        self.assertEqual(matrix.get(PipelineStage.RETRIEVER).status, PipelineStatus.PASS)
        self.assertEqual(matrix.get(PipelineStage.GENERATOR).status, PipelineStatus.PASS)
        self.assertEqual(matrix.get(PipelineStage.GROUNDING).status, PipelineStatus.PASS)
        
        for stage in PipelineStage:
            state = matrix.get(stage)
            self.assertIsNotNone(state.confidence)
            self.assertTrue(isinstance(state.confidence, float))
            # Test that language does not contain reasoning words
            obs_lower = state.observation.lower()
            for word in ["hallucination", "failure", "root cause"]:
                self.assertNotIn(word, obs_lower)
                
        self.assertEqual(matrix.artifact_version, "1.0")

    def test_retriever_failure(self):
        trace_id = "TRACE_RETRIEVER_FAIL"
        trace = self.create_mock_trace(trace_id, max_score=0.2)
        claim_set = self.create_mock_claim_set(trace_id, 2)
        verification = self.create_mock_verification(trace_id, [VerificationStatus.UNSUPPORTED, VerificationStatus.UNSUPPORTED])

        matrix = self.analyzer.analyze(trace, claim_set, verification)
        
        self.assertEqual(matrix.get(PipelineStage.RETRIEVER).status, PipelineStatus.FAIL)
        self.assertEqual(matrix.get(PipelineStage.GENERATOR).status, PipelineStatus.UNKNOWN)
        self.assertEqual(matrix.get(PipelineStage.GROUNDING).status, PipelineStatus.FAIL)

    def test_generator_failure(self):
        trace_id = "TRACE_GENERATOR_FAIL"
        trace = self.create_mock_trace(trace_id, max_score=0.9)
        claim_set = self.create_mock_claim_set(trace_id, 2)
        verification = self.create_mock_verification(trace_id, [VerificationStatus.SUPPORTED, VerificationStatus.UNSUPPORTED])

        matrix = self.analyzer.analyze(trace, claim_set, verification)
        
        self.assertEqual(matrix.get(PipelineStage.RETRIEVER).status, PipelineStatus.PASS)
        self.assertEqual(matrix.get(PipelineStage.GENERATOR).status, PipelineStatus.FAIL)
        self.assertEqual(matrix.get(PipelineStage.GROUNDING).status, PipelineStatus.FAIL)
        
        # Verify ids exist
        retriever_state = matrix.get(PipelineStage.RETRIEVER)
        self.assertTrue(len(retriever_state.supporting_claim_ids) > 0)
        self.assertTrue(len(retriever_state.supporting_chunk_ids) > 0)
        self.assertTrue(len(retriever_state.supporting_verification_ids) > 0)
        
    def test_grounding_failure(self):
        trace_id = "TRACE_GROUNDING_FAIL"
        trace = self.create_mock_trace(trace_id, max_score=0.8)
        claim_set = self.create_mock_claim_set(trace_id, 1)
        verification = self.create_mock_verification(trace_id, [VerificationStatus.UNSUPPORTED])
        
        matrix = self.analyzer.analyze(trace, claim_set, verification)
        self.assertEqual(matrix.get(PipelineStage.GROUNDING).status, PipelineStatus.FAIL)
        
    def test_save_and_load(self):
        trace_id = "TRACE_SAVE_LOAD"
        trace = self.create_mock_trace(trace_id, max_score=0.8)
        claim_set = self.create_mock_claim_set(trace_id, 1)
        verification = self.create_mock_verification(trace_id, [VerificationStatus.SUPPORTED])
        
        matrix = self.analyzer.analyze(trace, claim_set, verification)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = matrix.save(tmpdir)
            self.assertTrue(os.path.exists(filepath))
            
            loaded_matrix = matrix.__class__.load(filepath)
            self.assertEqual(loaded_matrix.trace_id, matrix.trace_id)
            self.assertEqual(loaded_matrix.artifact_version, matrix.artifact_version)
            self.assertEqual(len(loaded_matrix.pipeline_states), len(matrix.pipeline_states))
            
            loaded_retriever = loaded_matrix.get(PipelineStage.RETRIEVER)
            self.assertEqual(loaded_retriever.status, PipelineStatus.PASS)
            self.assertEqual(loaded_retriever.confidence, matrix.get(PipelineStage.RETRIEVER).confidence)

if __name__ == '__main__':
    unittest.main()
