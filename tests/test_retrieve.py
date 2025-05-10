import unittest
from unittest.mock import patch, MagicMock
import os

# Assuming the rtx_classifier package is in the Python path
# or tests are run from the root directory (e.g., using python -m unittest discover)
from rtx_classifier.nodes.retrieve import RetrieveNode

class TestRetrieveNode(unittest.TestCase):

    def setUp(self):
        # Set up environment variables that might be used by the node
        # These are defaults that RetrieveNode might try to load
        self.patch_env = patch.dict(os.environ, {
            "COLLECTION_NAME": "test_default_collection",
            "PERSIST_DIRECTORY": "./test_default_vectorstore",
            "TOP_K_RETRIEVAL": "10", # Default in retrieve.py
            "EMBEDDING_MODEL": "text-embedding-3-small" # Default in retrieve.py
        })
        self.patch_env.start()

        # Mock data for vector store results
        self.mock_docs_texts = ["Sample document 1 text", "Sample document 2 text"]
        self.mock_metadatas = [
            {"standard": "FAS100", "paragraph": "1.2.3"},
            {"standard": "FAS200", "paragraph": "4.5.6"}
        ]
        self.mock_distances = [0.1, 0.2] # Lower distance = higher similarity

    def tearDown(self):
        self.patch_env.stop()

    @patch('rtx_classifier.nodes.retrieve.get_embedding_function')
    @patch('rtx_classifier.nodes.retrieve.OpenAIEmbeddings')
    @patch('rtx_classifier.nodes.retrieve.get_or_create_vectorstore')
    def test_retrieve_node_success(self, mock_get_or_create_vs, mock_openai_embed, mock_get_embed_func):
        # --- Configure Mocks ---
        # Mock for get_or_create_vectorstore
        mock_collection_instance = MagicMock()
        mock_collection_instance.query.return_value = {
            "documents": [self.mock_docs_texts],
            "metadatas": [self.mock_metadatas],
            "distances": [self.mock_distances]
        }
        mock_get_or_create_vs.return_value = (MagicMock(), mock_collection_instance) # (client, collection)

        # Mock for OpenAIEmbeddings
        mock_openai_embed_instance = MagicMock()
        dummy_embedding_vector = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_openai_embed_instance.embed_query.return_value = dummy_embedding_vector
        mock_openai_embed.return_value = mock_openai_embed_instance
        
        # Mock for get_embedding_function (fallback, not strictly needed if OpenAIEmbeddings works)
        mock_get_embed_func.return_value = MagicMock()


        # --- Initialize RetrieveNode ---
        # Override some defaults for clarity in test
        test_collection_name = "my_test_collection"
        test_persist_dir = "/tmp/my_test_persist"
        test_top_k = 2 # To match mock data length
        test_embedding_model = "test-model-from-env"
        
        with patch.dict(os.environ, {"EMBEDDING_MODEL": test_embedding_model, "TOP_K_RETRIEVAL": str(test_top_k)}):
            retrieve_node = RetrieveNode(
                docs_dir="dummy_docs_dir", # Not used due to mocking get_or_create_vs
                collection_name=test_collection_name,
                persist_directory=test_persist_dir,
                embedding_function_name="openai", # Matches default, explicit for test
                top_k=test_top_k
            )

        # --- Sample Input State ---
        sample_state = {
            "entries": [
                {"account": "Cash", "debit": 1000.0, "credit": 0.0},
                {"account": "Sales Revenue", "debit": 0.0, "credit": 1000.0}
            ],
            "context": "Cash sale of goods",
            "adjustments": "None",
            "accounting_treatment": "Recognize revenue",
            "valid": True
        }

        # --- Call the Node ---
        result_state = retrieve_node(sample_state)

        # --- Assertions ---
        self.assertTrue(result_state.get("valid", False))
        self.assertNotIn("error_message", result_state)

        self.assertIn("retrieved_texts", result_state)
        self.assertEqual(result_state["retrieved_texts"], self.mock_docs_texts)

        self.assertIn("retrieved_sources", result_state)
        expected_sources = [
            f"{self.mock_metadatas[0]['standard']} ¶{self.mock_metadatas[0]['paragraph']}",
            f"{self.mock_metadatas[1]['standard']} ¶{self.mock_metadatas[1]['paragraph']}"
        ]
        self.assertEqual(result_state["retrieved_sources"], expected_sources)

        # Verify mock calls
        mock_get_or_create_vs.assert_called_once_with(
            docs_dir="dummy_docs_dir",
            collection_name=test_collection_name,
            persist_directory=test_persist_dir,
            embedding_function_name="openai"
        )
        mock_openai_embed.assert_called_once_with(model=test_embedding_model)
        mock_openai_embed_instance.embed_query.assert_called_once()
        
        # Verify query to ChromaDB
        # Construct expected query text based on _format_entries_for_search logic
        expected_query_text = """Transaction entries:
Dr. Cash: 1000.0
Cr. Sales Revenue: 1000.0

Context: Cash sale of goods

Adjustments: None

Accounting treatment: Recognize revenue"""
        # Check that embed_query was called with the correctly formatted text
        self.assertEqual(mock_openai_embed_instance.embed_query.call_args[0][0], expected_query_text)

        mock_collection_instance.query.assert_called_once_with(
            query_embeddings=[dummy_embedding_vector],
            n_results=test_top_k,
            include=["documents", "metadatas", "distances"]
        )

    @patch('rtx_classifier.nodes.retrieve.get_or_create_vectorstore')
    def test_retrieve_node_invalid_input_state(self, mock_get_or_create_vs):
        retrieve_node = RetrieveNode() # Use default init
        
        sample_state = {"valid": False, "error_message": "Previous step failed"}
        
        result_state = retrieve_node(sample_state)
        
        self.assertFalse(result_state["valid"])
        self.assertEqual(result_state["error_message"], "Previous step failed")
        self.assertNotIn("retrieved_texts", result_state)
        self.assertNotIn("retrieved_sources", result_state)
        
        mock_get_or_create_vs.assert_not_called() # Should not attempt to get collection

    @patch('rtx_classifier.nodes.retrieve.get_embedding_function')
    @patch('rtx_classifier.nodes.retrieve.OpenAIEmbeddings')
    @patch('rtx_classifier.nodes.retrieve.get_or_create_vectorstore')
    def test_retrieve_node_embedding_failure(self, mock_get_or_create_vs, mock_openai_embed, mock_get_embed_func):
        # Mock vector store setup (still needed to get to embedding part)
        mock_collection_instance = MagicMock()
        mock_get_or_create_vs.return_value = (MagicMock(), mock_collection_instance)

        # Mock OpenAIEmbeddings to raise primary error
        mock_openai_embed_instance = MagicMock()
        primary_error_msg = "OpenAI API is down"
        mock_openai_embed_instance.embed_query.side_effect = Exception(primary_error_msg)
        mock_openai_embed.return_value = mock_openai_embed_instance
        
        # Mock the fallback embedding function (self._embeddings) to also fail
        mock_chroma_fallback_embed_obj = MagicMock()
        fallback_error_msg = "Fallback embedder failed"
        # Make it callable but raise error, and ensure embed_documents also fails if tried
        mock_chroma_fallback_embed_obj.__call__.side_effect = Exception(fallback_error_msg)
        mock_chroma_fallback_embed_obj.embed_documents.side_effect = Exception(fallback_error_msg)
        mock_get_embed_func.return_value = mock_chroma_fallback_embed_obj

        retrieve_node = RetrieveNode()
        sample_state = {"entries": [{"account": "X", "debit": 1.0}], "valid": True}
        
        result_state = retrieve_node(sample_state)
        
        self.assertFalse(result_state["valid"])
        self.assertIn("error_message", result_state)
        self.assertIn("Retrieval error: All embedding methods failed.", result_state["error_message"])
        self.assertIn(f"Primary error: {primary_error_msg}", result_state["error_message"])
        self.assertIn(f"Fallback error: {fallback_error_msg}", result_state["error_message"])
        
        mock_get_embed_func.assert_called_once() # Fallback was attempted

    @patch('rtx_classifier.nodes.retrieve.OpenAIEmbeddings')
    @patch('rtx_classifier.nodes.retrieve.get_or_create_vectorstore')
    def test_retrieve_node_query_vectorstore_failure(self, mock_get_or_create_vs, mock_openai_embed):
        # Mock vector store setup, but make collection.query fail
        mock_collection_instance = MagicMock()
        query_error_msg = "ChromaDB is unavailable"
        mock_collection_instance.query.side_effect = Exception(query_error_msg)
        mock_get_or_create_vs.return_value = (MagicMock(), mock_collection_instance)

        # Mock OpenAIEmbeddings to succeed
        mock_openai_embed_instance = MagicMock()
        mock_openai_embed_instance.embed_query.return_value = [0.5, 0.5]
        mock_openai_embed.return_value = mock_openai_embed_instance
        
        retrieve_node = RetrieveNode()
        sample_state = {"entries": [{"account": "Y", "credit": 2.0}], "valid": True}
        
        result_state = retrieve_node(sample_state)
        
        self.assertFalse(result_state["valid"])
        self.assertIn("error_message", result_state)
        self.assertEqual(result_state["error_message"], f"Retrieval error: {query_error_msg}")

    def test_format_entries_for_search(self):
        retrieve_node = RetrieveNode() # Instance needed to call the method
        
        state_minimal = {
            "entries": [
                {"account": "Cash", "debit": 100.0, "credit": 0.0},
                {"account": "Revenue", "debit": 0.0, "credit": 100.0}
            ]
        }
        expected_query_minimal = """Transaction entries:
Dr. Cash: 100.0
Cr. Revenue: 100.0"""
        self.assertEqual(retrieve_node._format_entries_for_search(state_minimal), expected_query_minimal)

        state_full = {
            "entries": [
                {"account": "A/R", "debit": 50.0}, 
                {"account": "Sales", "credit": 50.0} 
            ],
            "context": "Sale on account",
            "adjustments": "Year-end accrual",
            "accounting_treatment": "IFRS 15"
        }
        expected_query_full = """Transaction entries:
Dr. A/R: 50.0
Cr. Sales: 50.0

Context: Sale on account

Adjustments: Year-end accrual

Accounting treatment: IFRS 15"""
        self.assertEqual(retrieve_node._format_entries_for_search(state_full), expected_query_full)

        state_no_entries = {"context": "General query about standards"}
        # Expected: "Transaction entries:\\n" (from join) + "\\n\\nContext..."
        expected_query_no_entries = """Transaction entries:


Context: General query about standards"""
        self.assertEqual(retrieve_node._format_entries_for_search(state_no_entries), expected_query_no_entries)

        state_only_entries = {
            "entries": [{"account": "Expense", "debit": 75.0}]
        }
        expected_query_only_entries = """Transaction entries:
Dr. Expense: 75.0"""
        self.assertEqual(retrieve_node._format_entries_for_search(state_only_entries), expected_query_only_entries)

    def test_persist_directory_true_uses_default(self):
        default_persist_dir_from_env = "/env/persist/dir"
        with patch.dict(os.environ, {"PERSIST_DIRECTORY": default_persist_dir_from_env}):
            # Pass persist_directory=True
            node = RetrieveNode(persist_directory=True)
            self.assertEqual(node.persist_directory, default_persist_dir_from_env)

            # Pass persist_directory=None (or not at all)
            node_none = RetrieveNode(persist_directory=None)
            self.assertIsNone(node_none.persist_directory) # Will be None

            # Pass persist_directory="some/path"
            custom_path = "some/specific/path"
            node_custom = RetrieveNode(persist_directory=custom_path)
            self.assertEqual(node_custom.persist_directory, custom_path)


if __name__ == '__main__':
    unittest.main()
