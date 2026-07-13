import os
import unittest
from unittest.mock import patch, MagicMock

from src.claim_verifier import ClaimVerifier, VerificationStatus, EvidenceSentence
from src.rag_trace import RAGTrace
from src.retriever import RetrievedChunk
from src.claim_decomposer import CandidateClaim, CandidateClaimSet

class TestClaimVerifier(unittest.TestCase):
    @patch('src.claim_verifier.pipeline')
    def setUp(self, mock_pipeline):
        # Create a mock pipeline that returns predefined scores based on the hypothesis
        self.mock_nli = MagicMock()
        mock_pipeline.return_value = self.mock_nli
        
        def side_effect(kwargs):
            hypothesis = kwargs.get("text_pair", "")
            if "supported" in hypothesis.lower():
                return [{"label": "entailment", "score": 0.9}, {"label": "neutral", "score": 0.05}, {"label": "contradiction", "score": 0.05}]
            elif "contradict" in hypothesis.lower():
                return [{"label": "entailment", "score": 0.05}, {"label": "neutral", "score": 0.05}, {"label": "contradiction", "score": 0.9}]
            elif "partial" in hypothesis.lower():
                return [{"label": "entailment", "score": 0.5}, {"label": "neutral", "score": 0.4}, {"label": "contradiction", "score": 0.1}]
            else: # unsupported/neutral
                return [{"label": "entailment", "score": 0.1}, {"label": "neutral", "score": 0.85}, {"label": "contradiction", "score": 0.05}]
                
        self.mock_nli.side_effect = side_effect
        self.verifier = ClaimVerifier(model_name="mock_model")

    def test_sentence_splitting(self):
        text = "This is sentence one. This is sentence two! And three? Yes."
        sentences = self.verifier._split_into_sentences(text)
        self.assertEqual(len(sentences), 4)
        self.assertEqual(sentences[0], "This is sentence one.")

    def test_verification_status_supported(self):
        claim = CandidateClaim("c1", "t1", "This claim is supported.", "", "S1", 0, 0, 10, {})
        chunks = [RetrievedChunk(chunk_id="ch1", similarity_score=0.9, rank=1, page_number="1", source_file="doc.pdf", chunk_index=0, chunk_text="Some premise text.")]
        
        result = self.verifier.verify_claim(claim, chunks)
        
        self.assertEqual(result.verification_status, VerificationStatus.SUPPORTED)
        self.assertGreater(result.entailment_score, 0.7)
        self.assertIn("Evidence directly supports", result.verification_reason)
        
    def test_verification_status_contradicted(self):
        claim = CandidateClaim("c2", "t1", "This claim is contradicted.", "", "S1", 0, 0, 10, {})
        chunks = [RetrievedChunk(chunk_id="ch1", similarity_score=0.9, rank=1, page_number="1", source_file="doc.pdf", chunk_index=0, chunk_text="Some premise text.")]
        
        result = self.verifier.verify_claim(claim, chunks)
        self.assertEqual(result.verification_status, VerificationStatus.CONTRADICTED)
        self.assertGreater(result.contradiction_score, 0.7)

    def test_top_3_sentences(self):
        claim = CandidateClaim("c1", "t1", "This claim is supported.", "", "S1", 0, 0, 10, {})
        # Text with 4 sentences
        chunks = [RetrievedChunk(chunk_id="ch1", similarity_score=0.9, rank=1, page_number="1", source_file="doc.pdf", chunk_index=0, chunk_text="Sentence A. Sentence B. Sentence C. Sentence D.")]
        
        result = self.verifier.verify_claim(claim, chunks)
        self.assertEqual(len(result.top_evidence), 3)

    def test_verify_all_and_artifacts(self):
        claim_set = CandidateClaimSet("t1")
        claim_set.add_claim(CandidateClaim("c1", "t1", "Supported claim.", "", "S1", 0, 0, 10, {}))
        claim_set.add_claim(CandidateClaim("c2", "t1", "Contradicted claim.", "", "S2", 1, 0, 10, {}))
        
        trace_id = "t1"
        chunks = [RetrievedChunk(chunk_id="ch1", similarity_score=0.9, rank=1, page_number="1", source_file="doc.pdf", chunk_index=0, chunk_text="Premise.")]
        
        summary = self.verifier.verify_all(claim_set, trace_id, chunks)
        
        self.assertEqual(summary.total_claims, 2)
        self.assertEqual(summary.supported_claims, 1)
        self.assertEqual(summary.contradicted_claims, 1)
        
        # Test artifact saving
        self.verifier.save_artifacts(summary)
        
        filepath = os.path.join("artifacts", "verification", "TRACE_t1.json")
        self.assertTrue(os.path.exists(filepath))
        
        # Cleanup
        if os.path.exists(filepath):
            os.remove(filepath)

if __name__ == '__main__':
    unittest.main()
