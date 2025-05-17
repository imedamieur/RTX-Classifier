"""
RationaleNode for the RTX Classifier.

Generates an explanation for the classification based on various inputs including
retrieved texts, transaction details, and classification results.
"""
import os
from typing import Dict, Any, Optional, List

import dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
dotenv.load_dotenv()

# Get configuration from environment variables
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o")
DEFAULT_MODEL_PROVIDER = os.getenv("DEFAULT_MODEL_PROVIDER", "openai")
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

class RationaleNode:
    """
    RationaleNode: This node is currently bypassed and does not generate explanations.
    It passes the state through without modification related to rationale generation.
    """

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None, # Kept for signature compatibility, but not used
        template_path: Optional[str] = None, # Kept for signature compatibility, but not used
    ):
        """
        Initialize the RationaleNode. LLM and template are not actively used.
        """
        # self.llm = llm # Not used
        # self.output_parser = JsonOutputParser() # Not used
        # self._prompt_template = self._get_default_template() # Not used
        print("RationaleNode initialized (currently bypassed for explanation generation).")

    def _get_default_template(self) -> Optional[ChatPromptTemplate]:
        """
        Returns None as the template is not used.
        """
        return None

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Passes the state through without generating an explanation.

        Args:
            state: Input state from the previous node.

        Returns:
            The same input state, with a note that rationale was skipped.
        """
        new_state = dict(state)
        # print("RationaleNode: Bypassed, passing state through.")
        return new_state

# Example Usage (for testing RationaleNode independently)
# if __name__ == '__main__':
#     # This is a simplified example. In a real pipeline, state would be richer.
#     mock_state = {
#         "entries": [{"description": "Sale of goods to Customer A for $1000"}],
#         "context": "End of Q2 reporting period.",
#         "adjustments": None,
#         "accounting_treatment": None,
#         "provisional_label_name": "Revenue",
#         "prob_vector": [0.1, 0.8, 0.05, 0.05], # Assuming Revenue is the second label
#         "retrieved_texts": [
#             "AAOIFI FAS 10, Para 15: Revenue from core operations is recognized when earned.",
#             "Company Policy Sec 4.1: All sales of goods are considered core operational revenue."
#         ],
#         "valid": True # Assuming previous steps were valid
#     #
#     # Initialize with default LLM (requires OPENAI_API_KEY or GOOGLE_API_KEY)
#     # Ensure your .env file is set up or keys are in environment
#     try:
#         rationale_node = RationaleNode()
#         updated_state = rationale_node(mock_state)
#         print("Updated State from RationaleNode:")
#         import json
#         print(json.dumps(updated_state, indent=2))
#     except Exception as e:
#         print(f"Error initializing or running RationaleNode: {e}")
#
# # End of file
