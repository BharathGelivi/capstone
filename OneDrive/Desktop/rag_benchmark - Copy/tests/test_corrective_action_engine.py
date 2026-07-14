import unittest
import os
import tempfile
from src.root_cause_reasoner import RootCauseAnalysis, FailureType
from src.corrective_action_engine import CorrectiveActionEngine, ActionCategory

class TestCorrectiveActionEngine(unittest.TestCase):
    def setUp(self):
        self.engine = CorrectiveActionEngine()

    def test_healthy_pipeline(self):
        rca = RootCauseAnalysis(
            trace_id="TRACE_HEALTHY",
            primary_cause=FailureType.UNKNOWN,
            secondary_effects=[],
            reasoning_chain=[],
            recommendations_needed=False
        )
        plan = self.engine.generate(rca)
        
        self.assertEqual(plan.trace_id, "TRACE_HEALTHY")
        self.assertEqual(len(plan.immediate_actions), 0)
        self.assertEqual(len(plan.short_term_actions), 0)
        self.assertEqual(len(plan.experimental_actions), 0)

    def test_retriever_failure(self):
        rca = RootCauseAnalysis(
            trace_id="TRACE_RETRIEVER",
            primary_cause=FailureType.RETRIEVAL_MISS,
            secondary_effects=[],
            reasoning_chain=[],
            recommendations_needed=True
        )
        plan = self.engine.generate(rca)
        
        self.assertTrue(len(plan.immediate_actions) > 0)
        self.assertTrue(len(plan.short_term_actions) > 0)
        self.assertTrue(len(plan.experimental_actions) > 0)

        all_actions = plan.immediate_actions + plan.short_term_actions + plan.experimental_actions
        for act in all_actions:
            self.assertEqual(act.category, ActionCategory.RETRIEVAL)
            self.assertIsNotNone(act.observed_evidence)
            self.assertIsNotNone(act.root_cause)
            self.assertIsNotNone(act.expected_improvement)
            self.assertIsNotNone(act.success_metric)

    def test_mixed_failures(self):
        rca = RootCauseAnalysis(
            trace_id="TRACE_MIXED",
            primary_cause=FailureType.RETRIEVAL_MISS,
            secondary_effects=[FailureType.UNSUPPORTED_GENERATION],
            reasoning_chain=[],
            recommendations_needed=True
        )
        plan = self.engine.generate(rca)
        
        all_actions = plan.immediate_actions + plan.short_term_actions + plan.experimental_actions
        categories = [act.category for act in all_actions]
        self.assertIn(ActionCategory.RETRIEVAL, categories)
        self.assertIn(ActionCategory.GENERATION, categories)

    def test_save_and_load(self):
        rca = RootCauseAnalysis(
            trace_id="TRACE_SAVE",
            primary_cause=FailureType.CHUNK_BOUNDARY,
            secondary_effects=[],
            reasoning_chain=[],
            recommendations_needed=True
        )
        plan = self.engine.generate(rca)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = plan.save(tmpdir)
            self.assertTrue(os.path.exists(filepath))
            
            loaded_plan = plan.__class__.load(filepath)
            self.assertEqual(loaded_plan.trace_id, plan.trace_id)
            self.assertEqual(loaded_plan.primary_cause, plan.primary_cause)
            self.assertEqual(len(loaded_plan.immediate_actions), len(plan.immediate_actions))

if __name__ == '__main__':
    unittest.main()
