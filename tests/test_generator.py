import unittest
from unittest.mock import patch, MagicMock

from src.retriever import RetrievalResult, RetrievedChunk
from src.generator import Generator, PromptBuilder
from configs.prompts import DEFAULT_SYSTEM_INSTRUCTIONS, LEGAL_DOMAIN_SYSTEM_INSTRUCTIONS


def make_retrieval_result(num_chunks=2):
    chunks = [
        RetrievedChunk(
            chunk_id=f"c{i}",
            similarity_score=0.9,
            rank=i,
            page_number=str(i),
            source_file="doc.pdf",
            chunk_index=i,
            chunk_text=f"Chunk text number {i}.",
        )
        for i in range(1, num_chunks + 1)
    ]
    return RetrievalResult(
        question="What does the document say?",
        question_embedding_dimension=3,
        retrieved_chunks=chunks,
        retrieved_chunk_ids=[c.chunk_id for c in chunks],
        similarity_scores=[c.similarity_score for c in chunks],
        retrieval_time=0.1,
        top_k=num_chunks,
        retrieval_metadata={},
    )


def make_chat_response(content):
    response = MagicMock()
    response.message.content = content
    return response


class TestGenerator(unittest.TestCase):
    @patch("src.generator.Groq")
    def test_generate_returns_generation_result_fields(self, mock_llm_cls):
        mock_llm = MagicMock()
        mock_llm.chat.return_value = make_chat_response("This is the generated answer.")
        mock_llm_cls.return_value = mock_llm

        retrieval_result = make_retrieval_result()
        generator = Generator()
        result = generator.generate(retrieval_result)

        self.assertEqual(result.generated_answer, "This is the generated answer.")
        self.assertEqual(result.question, retrieval_result.question)
        self.assertEqual(result.retrieved_chunk_ids, retrieval_result.retrieved_chunk_ids)
        self.assertGreater(result.prompt_length, 0)

    @patch("src.generator.Groq")
    def test_groq_uses_max_tokens_kwarg(self, mock_llm_cls):
        mock_llm_cls.return_value = MagicMock()
        Generator(max_tokens=2048)

        _, kwargs = mock_llm_cls.call_args
        self.assertEqual(kwargs.get("max_tokens"), 2048)

    @patch("src.generator.LLM_PROVIDER", "huggingface")
    @patch("src.generator.HuggingFaceInferenceAPI")
    def test_hf_uses_num_output_not_max_new_tokens(self, mock_llm_cls):
        # HuggingFaceInferenceAPI has no max_new_tokens field -- passing it is
        # silently ignored and the token limit falls back to the library's
        # 256-token default, truncating real answers. num_output is the real field.
        mock_llm_cls.return_value = MagicMock()
        Generator(max_tokens=2048)

        _, kwargs = mock_llm_cls.call_args
        self.assertEqual(kwargs.get("num_output"), 2048)
        self.assertNotIn("max_new_tokens", kwargs)

    @patch("src.generator.Groq")
    def test_prompt_length_and_chunk_ids_passthrough(self, mock_llm_cls):
        mock_llm = MagicMock()
        mock_llm.chat.return_value = make_chat_response("Answer.")
        mock_llm_cls.return_value = mock_llm

        retrieval_result = make_retrieval_result(num_chunks=1)
        generator = Generator()
        result = generator.generate(retrieval_result)

        self.assertEqual(result.prompt_length, len(result.prompt))
        self.assertEqual(result.retrieved_chunk_ids, ["c1"])

    @patch("src.generator.Groq")
    def test_llm_error_is_captured_in_generated_answer(self, mock_llm_cls):
        mock_llm = MagicMock()
        mock_llm.chat.side_effect = RuntimeError("boom")
        mock_llm_cls.return_value = mock_llm

        generator = Generator()
        result = generator.generate(make_retrieval_result())

        self.assertIn("Error generating answer", result.generated_answer)

    def test_default_instructions_are_domain_agnostic(self):
        self.assertNotIn("legal", DEFAULT_SYSTEM_INSTRUCTIONS.lower())

    def test_legal_preset_available(self):
        self.assertIn("legal assistant", LEGAL_DOMAIN_SYSTEM_INSTRUCTIONS.lower())

    def test_build_messages_uses_default_instructions_by_default(self):
        messages = PromptBuilder.build_messages("Q?", [])
        self.assertEqual(messages[0].content, DEFAULT_SYSTEM_INSTRUCTIONS)

    def test_build_messages_override(self):
        custom = "You are a custom assistant."
        messages = PromptBuilder.build_messages("Q?", [], system_instructions=custom)
        self.assertEqual(messages[0].content, custom)

    def test_build_messages_override_legal_preset(self):
        messages = PromptBuilder.build_messages("Q?", [], system_instructions=LEGAL_DOMAIN_SYSTEM_INSTRUCTIONS)
        self.assertEqual(messages[0].content, LEGAL_DOMAIN_SYSTEM_INSTRUCTIONS)

    def test_context_chunk_includes_chunk_id(self):
        chunk = RetrievedChunk(
            chunk_id="chunk-42", similarity_score=0.9, rank=1, page_number="1",
            source_file="doc.pdf", chunk_index=0, chunk_text="Some text."
        )
        messages = PromptBuilder.build_messages("Q?", [chunk])
        self.assertIn("[Chunk-ID: chunk-42]", messages[1].content)


if __name__ == "__main__":
    unittest.main()
