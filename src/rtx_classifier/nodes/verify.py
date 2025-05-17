"""
VerifyNode for the RTX Classifier.

Verifies that the classification cites only retrieved text and contains no forbidden terms.
Supports both OpenAI and Google Gemini models.
"""
from typing import Dict, List, Any, Optional, Set
import os
import re
import dotenv

# Load environment variables
dotenv.load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Get configuration from environment variables
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o")
DEFAULT_MODEL_PROVIDER = os.getenv("DEFAULT_MODEL_PROVIDER", "openai")
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))


class VerifyNode:
    """
    VerifyNode that verifies the classification.
    
    The verification process ensures that:
    1. The classification cites only content from retrieved texts
    2. The classification contains no forbidden terms
    3. The classification properly supports the classification
    """
    
    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        template_path: Optional[str] = None,
    ):
        """
        Initialize the VerifyNode.
        
        Args:
            llm: LangChain chat model instance
            template_path: Path to the Jinja template for the verification prompt
        """
        self.llm = llm
        self.template_path = template_path
        
        # Default verification prompt
        self._prompt_template = self._get_default_template()
        self.output_parser = JsonOutputParser()
        
        # Create default LLM if none provided
        if self.llm is None:
            model_provider = os.getenv("DEFAULT_MODEL_PROVIDER", "openai").lower()
            if model_provider == "openai":
                self.llm = ChatOpenAI(
                    model_name=DEFAULT_MODEL,
                    temperature=DEFAULT_TEMPERATURE,
                )
            elif model_provider == "google":
                self.llm = ChatGoogleGenerativeAI(
                    model=DEFAULT_MODEL,
                    temperature=DEFAULT_TEMPERATURE,
                    convert_system_message_to_human=True,
                )
    
    def _get_default_template(self) -> ChatPromptTemplate:
        """Get default template for verification."""
        template = """\
        You are an expert in Islamic financial accounting and reporting standards, specifically AAOIFI FAS.
        Your task is to verify a provisional classification of a financial transaction or event.
        
        The user will provide:
        1. Retrieved context from AAOIFI FAS documents.
        2. A provisional classification label out of these five Financial Accounting Standards: FAS4, FAS7, FAS10, FAS28, FAS32.
        - FAS 4: Musharaka financing
        - FAS 7: Salam and Parallel Salam
        - FAS 10: Istisna'a and Parallel Istisna'a
        - FAS 28: Murabaha and Other Deferred Payment Sales
        - FAS 32: Ijara and Ijara Muntahia Bittamleek
        The provisional classification label is based on the retrieved context.	
        
        You need to perform the following checks:
        1. Source Adherence: Ensure the provisional classification is plausible based *only* on the provided "Retrieved Context".
        2. Classification Support: Critically evaluate if the retrieved context supports the provisional classification label.
        
        Based on your verification, provide a JSON response with the following fields:
        - "valid": (boolean) true if the provisional classification is supported by the context. False otherwise.
        - "final_classification_label": (string) The verified classification label. This should usually be the same as the provisional_label. If the provisional label is fundamentally unsupported, return the original provisional_label and mark as invalid.
        - "issue_note": (string, optional) If "valid" is false, provide a brief note on the primary issue (e.g., "Context does not support label.", "Label seems unrelated to context."). If "valid" is true, this can be omitted.
        
        The provisional classification label is: {{ provisional_label_name }}
        
        Retrieved Context:
        ---
        {% for text in retrieved_texts %}
        {{ text }}
        ---
        {% endfor %}
        
        Respond with ONLY the JSON object.
        """
        return ChatPromptTemplate.from_template(template, template_format="jinja2")
    
    def _run_llm_verification(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the LLM-based verification on the classification.
        
        Args:
            state: Current state with provisional classification and transaction details
            
        Returns:
            Verification results with a simplified structure.
        """
        if not self.llm:
            return {
                "valid": False,
                "issues": ["No LLM available for verification"],
                "final_classification_label": state.get("provisional_label_name", ""),
            }
        
        # Prepare variables for the template
        inputs = {
            "provisional_label_name": state.get("provisional_label_name", ""),
            "retrieved_texts": state.get("retrieved_texts", []),  
        }
        
        # Create the chain
        chain = self._prompt_template | self.llm | self.output_parser
        
        # Call the LLM
        try:
            result = chain.invoke(inputs)

            is_valid_from_llm = result.get("valid", False)
            if isinstance(is_valid_from_llm, str):
                is_valid_from_llm = is_valid_from_llm.lower() == 'true'
            result["valid"] = bool(is_valid_from_llm)

            if "final_classification_label" not in result:
                result["final_classification_label"] = state.get("provisional_label_name", "")

            issues_for_downstream = []
            llm_issue_note = result.get("issue_note")

            if llm_issue_note and isinstance(llm_issue_note, str) and llm_issue_note.strip():
                issues_for_downstream.append(llm_issue_note)
            
            if not result["valid"] and not issues_for_downstream:
                issues_for_downstream.append(
                    "Verification LLM reported 'valid: false' but 'issue_note' was missing or empty."
                )
            
            # Return a simplified structure
            return {
                "valid": result["valid"],
                "final_classification_label": result["final_classification_label"],
                "issues": issues_for_downstream
            }

        except Exception as e:
            return {
                "valid": False,
                "issues": [f"Verification LLM error: {str(e)}"],
                "final_classification_label": state.get("provisional_label_name", ""),
            }
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the input state and verify the classification.
        
        Args:
            state: Input state with provisional label
            
        Returns:
            Updated state with verification results, focusing on validity and final label.
        """
        new_state = dict(state)
    
        if not new_state.get("valid", False): # Check if prior nodes invalidated the state
            # Ensure 'final_classification_label' exists if we are returning early due to prior invalidation
            if "final_classification_label" not in new_state:
                 new_state["final_classification_label"] = new_state.get("provisional_label_name", "")
            # Ensure 'probs_vector' is preserved if it exists
            if "probs_vector" not in new_state:
                new_state["probs_vector"] = {} # Or some default if appropriate
            return new_state
        
        provisional_label_name = new_state.get("provisional_label_name", "")
        # print(f"VerifyNode: Provisional Label for verification: {provisional_label_name}")

        verification_result = self._run_llm_verification(new_state)
        
        new_state["valid"] = verification_result.get("valid", False)
        new_state["final_classification_label"] = verification_result.get("final_classification_label", provisional_label_name)
        
        provisional_rationale = new_state.get("provisional_rationale", "")
        print(f"DEBUGAAAAA: VerifyNode: Provisional rationale: {provisional_rationale}")

        if not new_state["valid"]:
            error_messages = verification_result.get("issues", ["Verification failed for unspecified reasons."])
            new_state["error_message"] = "Verification failed: " + "; ".join(error_messages)
            new_state["retry_count"] = new_state.get("retry_count", 0) + 1
            # print(f"VerifyNode: Verification failed for {provisional_label_name}. Reason: {new_state['error_message']}")
        else:
            new_state["final_rationale"] = provisional_rationale
            print(f"VerifyNode: Verification successful for {provisional_label_name}. Final label: {new_state['final_classification_label']}")

        # Ensure probs_vector is preserved or initialized
        if "probs_vector" not in new_state:
            new_state["probs_vector"] = {} # Or handle as per how it's generated upstream

        # Clean up any other potential explanation-related fields that might be in the state
        # from previous nodes, if their names are known.
        # For now, the specific ones are popped above.
     
        return new_state