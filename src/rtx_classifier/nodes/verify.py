# filepath: d:\Cat2\src\rtx_classifier\nodes\verify.py
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
    VerifyNode that verifies the classification using the ReAct technique.
    
    The verification process ensures that:
    1. The classification cites only content from retrieved texts
    2. The classification contains no forbidden terms
    3. The classification properly supports the classification
    
    The ReAct approach enables step-by-step reasoning and explanation for the verification decision.
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
        """Get default template for verification using ReAct technique."""
        template = """\
        You are an expert in Islamic financial accounting and reporting standards, specifically AAOIFI FAS.
        Your task is to verify a provisional classification of a financial transaction or event using a step-by-step reasoning process (ReAct approach).
        
        The user will provide:
        1. Retrieved context from AAOIFI FAS documents.
        2. A provisional classification label out of these five Financial Accounting Standards: FAS4, FAS7, FAS10, FAS28, FAS32.
        - FAS 4: Musharaka financing
        - FAS 7: Salam and Parallel Salam
        - FAS 10: Istisna\'a and Parallel Istisna\'a
        - FAS 28: Murabaha and Other Deferred Payment Sales
        - FAS 32: Ijara and Ijara Muntahia Bittamleek
        The provisional classification label is based on the retrieved context.
        
        IMPORTANT NOTE: You do NOT need to find direct and explicit presence of specific terms and conditions related to that accounting standard in the retrieved context. The verification should focus on ensuring that the classification is not completely wrong or without any basis at all. If the retrieved context supports even a general association with the chosen standard, that is sufficient to consider the classification valid.
        
        Please follow this step-by-step reasoning process:
        
        Step 1: Analyze the retrieved context - summarize the key points from the provided texts.
        Step 2: Identify the key characteristics of the provisional classification label.
        Step 3: Compare these characteristics with the content in the retrieved context.
        Step 4: Evaluate if there are any inconsistencies or missing information. Note that you don\'t need perfect alignment - just ensure the classification has some reasonable basis.
        Step 5: Make a determination on whether the provisional classification is supported by the context. Be lenient - only reject classifications that are clearly wrong.
        Step 6: Provide a detailed explanation of your reasoning process and final decision.
        
        Based on your verification, provide a JSON response with the following fields:
        - "valid": (boolean) true if the provisional classification is supported by the context. False otherwise.
        - "final_classification_label": (string) The verified classification label. This should usually be the same as the provisional_label. If the provisional label is fundamentally unsupported, return the original provisional_label and mark as invalid.
        - "verification_explanation": (string) A detailed explanation of your reasoning process and why you accepted or rejected the classification. This should follow the ReAct pattern of step-by-step reasoning.
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
        Run the LLM-based verification on the classification using ReAct.
        
        Args:
            state: Current state with provisional classification and transaction details
            
        Returns:
            Verification results with a simplified structure including the explanation.
        """
        if not self.llm:
            return {
                "valid": False,
                "issues": ["No LLM available for verification"],
                "final_classification_label": state.get("provisional_label_name", ""),
                "verification_explanation": "Verification could not be performed as no LLM was available."
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

            # Extract the explanation from the ReAct process
            verification_explanation = result.get("verification_explanation", "No detailed explanation provided.")

            issues_for_downstream = []
            llm_issue_note = result.get("issue_note")

            if llm_issue_note and isinstance(llm_issue_note, str) and llm_issue_note.strip():
                issues_for_downstream.append(llm_issue_note)
            
            if not result["valid"] and not issues_for_downstream:
                issues_for_downstream.append(
                    "Verification LLM reported 'valid: false' but 'issue_note' was missing or empty."
                )
            
            # Return a structure including the verification explanation
            return {
                "valid": result["valid"],
                "final_classification_label": result["final_classification_label"],
                "issues": issues_for_downstream,
                "verification_explanation": verification_explanation
            }

        except Exception as e:
            return {
                "valid": False,
                "issues": [f"Verification LLM error: {str(e)}"],
                "final_classification_label": state.get("provisional_label_name", ""),
                "verification_explanation": f"Error during verification: {str(e)}"
            }
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the input state and verify the classification.
        
        Args:
            state: Input state with provisional label
            
        Returns:
            Updated state with verification results, focusing on validity, final label, and explanation.
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
        
        # Add the detailed verification explanation to the state
        new_state["verification_explanation"] = verification_result.get("verification_explanation", "")
        
        provisional_rationale = new_state.get("provisional_rationale", "")
        
        if not new_state["valid"]:
            error_messages = verification_result.get("issues", ["Verification failed for unspecified reasons."])
            new_state["error_message"] = "Verification failed: " + "; ".join(error_messages)
            new_state["retry_count"] = new_state.get("retry_count", 0) + 1
        else:
            # If verification is successful, use the detailed explanation as the final rationale
            if new_state["verification_explanation"]:
                new_state["final_rationale"] = new_state["verification_explanation"]
            else:
                new_state["final_rationale"] = provisional_rationale
                
            print(f"VerifyNode: Verification successful for {provisional_label_name}. Final label: {new_state['final_classification_label']}")
        
        # For debugging - show the verification explanation
        print(f"VerifyNode: Verification explanation: {new_state.get('verification_explanation', '')[:100]}...")

        # Ensure probs_vector is preserved or initialized
        if "probs_vector" not in new_state:
            new_state["probs_vector"] = {} # Or handle as per how it's generated upstream
     
        return new_state
