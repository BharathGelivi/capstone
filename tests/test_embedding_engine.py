import unittest
from unittest.mock import patch, MagicMock

from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
from src.chunk_registry import ChunkRegistry
from src.embedding_engine import generate_embeddings


def make_node(chunk_id, doc_id, text):
    node = TextNode(text=text, id_=chunk_id, metadata={"file_name": "doc.pdf"})
    node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=doc_id)
    return node


class TestEmbeddingEngine(unittest.TestCase):
    def setUp(self):
        self.registry = ChunkRegistry()
        self.registry.register([
            make_node("c1", "doc-1", "First chunk of text."),
            make_node("c2", "doc-1", "Second chunk of text."),
        ])

    @patch("src.embedding_engine.HuggingFaceEmbedding")
    def test_generate_embeddings_count_and_dimension(self, mock_embed_cls):
        mock_model = MagicMock()
        mock_model.get_text_embedding.return_value = [0.1, 0.2, 0.3]
        mock_embed_cls.return_value = mock_model

        records = generate_embeddings(self.registry)

        self.assertEqual(len(records), 2)
        for record in records:
            self.assertEqual(record.embedding_dimension, 3)
            self.assertEqual(record.embedding, [0.1, 0.2, 0.3])

    @patch("src.embedding_engine.HuggingFaceEmbedding")
    def test_chunk_id_and_parent_id_passthrough(self, mock_embed_cls):
        mock_model = MagicMock()
        mock_model.get_text_embedding.return_value = [0.1, 0.2]
        mock_embed_cls.return_value = mock_model

        records = generate_embeddings(self.registry)
        record_by_id = {r.chunk_id: r for r in records}

        self.assertEqual(record_by_id["c1"].parent_document_id, "doc-1")
        self.assertEqual(record_by_id["c2"].parent_document_id, "doc-1")


if __name__ == "__main__":
    unittest.main()
