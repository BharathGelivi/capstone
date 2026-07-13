import unittest
from src.claim_decomposer import ClaimDecomposer, JSONRecoveryError

class TestJSONRecovery(unittest.TestCase):
    def setUp(self):
        self.decomposer = ClaimDecomposer(model_name="mock")

    def test_valid_json(self):
        text = '[{"claim_text": "Valid.", "sentence_id": "S1"}]'
        result = self.decomposer._robust_json_parse(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["claim_text"], "Valid.")

    def test_markdown_wrapped_json(self):
        text = '```json\n[{"claim_text": "Markdown.", "sentence_id": "S1"}]\n```'
        result = self.decomposer._robust_json_parse(text)
        self.assertEqual(len(result), 1)

    def test_trailing_text_json(self):
        text = 'Here is the output:\n[{"claim_text": "Trailing.", "sentence_id": "S1"}]\nHope this helps!'
        result = self.decomposer._robust_json_parse(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["claim_text"], "Trailing.")

    def test_unbalanced_brackets_json(self):
        # Missing closing bracket for array and brace for object
        text = '[{"claim_text": "Unbalanced.", "sentence_id": "S1"'
        result = self.decomposer._robust_json_parse(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["claim_text"], "Unbalanced.")

    def test_completely_invalid_json(self):
        text = 'This is not json at all.'
        with self.assertRaises(JSONRecoveryError):
            self.decomposer._robust_json_parse(text)

if __name__ == '__main__':
    unittest.main()
