import unittest
from fastapi.testclient import TestClient
from src.api import app
from src.rag_trace import RAGTrace

client = TestClient(app)

class TestAPI(unittest.TestCase):
    def test_get_root(self):
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("framework_name", response.json())

    def test_get_health(self):
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "healthy")

    def test_get_version(self):
        response = client.get("/version")
        self.assertEqual(response.status_code, 200)
        self.assertIn("framework_version", response.json())
        
    def test_get_report_not_found(self):
        response = client.get("/report/TRACE_INVALID")
        self.assertEqual(response.status_code, 404)
        
    def test_get_artifacts_not_found(self):
        response = client.get("/artifacts/TRACE_INVALID")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()["artifacts"]), 0)

    def test_analyze_invalid_payload(self):
        response = client.post("/analyze", json={"invalid": "payload"})
        self.assertEqual(response.status_code, 400)

if __name__ == '__main__':
    unittest.main()
