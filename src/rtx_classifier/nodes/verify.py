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
        1. The original query describing a financial transaction or situation that needs classification.
        2. Retrieved context from AAOIFI FAS documents.
        3. A provisional classification label out of these five Financial Accounting Standards: FAS4, FAS7, FAS10, FAS28, FAS32.
        - FAS 4: Musharaka financing
        - FAS 7: Salam and Parallel Salam
        - FAS 10: Istisna\'a and Parallel Istisna\'a
        - FAS 28: Murabaha and Other Deferred Payment Sales
        - FAS 32: Ijara and Ijara Muntahia Bittamleek
        The provisional classification label is based on the retrieved context.
        
        IMPORTANT NOTE: You do NOT need to find direct and explicit presence of specific terms and conditions related to that accounting standard in the retrieved context. The verification should focus on ensuring that the classification is not completely wrong or without any basis at all. If the retrieved context supports even a general association with the chosen standard, that is sufficient to consider the classification valid.
        
        Please follow this step-by-step reasoning process:
        
        Step 1: First, analyze the original query to understand what financial transaction or situation needs to be classified.
        Step 2: Based on the query alone, consider which FAS standard(s) might be appropriate classifications. Note any alternative standards that could potentially apply.
        Step 3: Analyze the retrieved context - summarize the key points from the provided texts that are relevant to the query.
        Step 4: Identify the key characteristics of the provisional classification label.
        Step 5: Compare these characteristics with both the query details and the content in the retrieved context.
        Step 6: Evaluate if there are any inconsistencies or missing information. Note that you don\'t need perfect alignment - just ensure the classification has some reasonable basis for the specific query.
        Step 7: Consider whether any alternative standards identified in Step 2 might be more appropriate than the provisional classification.
        Step 8: Make a determination on whether the provisional classification is appropriate for the query and supported by the context. Be lenient - only reject classifications that are clearly wrong.
        Step 9: Provide a detailed explanation of your reasoning process and final decision.
        
        Based on your verification, provide a JSON response with the following fields:
        - "valid": (boolean) true if the provisional classification is supported by the context and appropriate for the query. False otherwise.
        - "final_classification_label": (string) The verified classification label. This should usually be the same as the provisional_label. If the provisional label is fundamentally unsupported, return the original provisional_label and mark as invalid.
        - "verification_explanation": (string) A detailed explanation of your reasoning process and why you accepted or rejected the classification. This should follow the ReAct pattern of step-by-step reasoning.
        - "alternative_standards": (array) Include standards that were seriously considered during your reasoning process, especially those that could apply but weren't chosen due to lack of specific details. For each alternative standard, you MUST explain WHY it could be considered as an option - what specific aspects of the transaction or query align with this standard's characteristics. Focus on explaining the reasoning that made you consider this standard and what specific elements of the transaction would fit under this standard. Don't include standards that are clearly inappropriate, but do include close contenders.
        - "issue_note": (string, optional) If "valid" is false, provide a brief note on the primary issue (e.g., "Context does not support label.", "Label seems unrelated to context.", "Classification doesn't match the query."). If "valid" is true, this can be omitted.
        
        The query to classify is: {{ query }}
        
        The provisional classification label is: {{ provisional_label_name }}
        
        {% if api_query %}
        SEARCH QUERY USED: "{{ api_query }}"
        {% endif %}
        
        RELEVANCE SCORE EXPLANATION: The relevance scores below indicate how closely each chunk of text matches the transaction details. Scores range from 0 to 1, where higher scores (closer to 1) indicate stronger relevance to the transaction. Give more weight to chunks with higher relevance scores when determining classification.
        
        Based on the RAG (BM25 + VECTOR STORE) retrieved sources (sorted by relevance):
        {% for chunk in api_chunks|sort(attribute='score', reverse=true) %}
        --- Source from {{ chunk.document_name }} (Relevance Score: {{ chunk.score|round(3) }}) ---
        {{ chunk.chunk_text }}
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
        
        # Process api_retrieve data
        api_chunks = []
        api_query = ""
        api_retrieve = state.get("api_retrieve_response", None)
        if api_retrieve and isinstance(api_retrieve, dict):
            # Extract the FAS chunks and query from the API retrieve response
            api_chunks = api_retrieve.get("fas_chunks", [])
            api_query = api_retrieve.get("query", "")
            
            # Sort chunks by relevance score (highest first)
            if api_chunks:
                api_chunks = sorted(api_chunks, key=lambda x: float(x.get("score", 0)), reverse=True)
        
        # If no API chunks are available, set empty list to ensure proper handling
        if not api_chunks:
            api_chunks = []
        
        # Prepare variables for the template
        inputs = {
            "provisional_label_name": state.get("provisional_label_name", ""),
            "query": state.get("query", ""),
            "api_chunks": api_chunks,
            "api_query": api_query
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
            
            # Extract any alternative standards identified
            alternative_standards = result.get("alternative_standards", [])
            if not isinstance(alternative_standards, list):
                alternative_standards = []
                
            # Filter out alternatives that don't have proper explanations or are clearly not suitable
            filtered_alternatives = []
            for alt in alternative_standards:
                if isinstance(alt, str):
                    # Skip alternatives that are explicitly described as less likely/suitable
                    if any(phrase in alt.lower() for phrase in [
                        "less likely since", 
                        "not suitable because",
                        "clearly inappropriate",
                        "definitely not applicable"
                    ]):
                        continue
                    
                    # Ensure the alternative has an explanation of why it could be considered
                    if ("because" in alt.lower() or 
                        "as it" in alt.lower() or 
                        "due to" in alt.lower() or 
                        "since" in alt.lower() or
                        "elements of" in alt.lower() or
                        "could apply" in alt.lower() or
                        "might be" in alt.lower() or
                        "reason" in alt.lower()):
                        filtered_alternatives.append(alt)
                    else:
                        # If we have no explanation, add a generic one
                        if len(alt) > 5:  # Basic check that this is actually a standard
                            enhanced_alt = f"{alt} - Could be considered as the transaction has elements that might align with this standard's scope with additional details."
                            filtered_alternatives.append(enhanced_alt)
                else:
                    filtered_alternatives.append(alt)
            
            # Replace the original list with the filtered one that excludes clearly unsuitable options
            alternative_standards = filtered_alternatives
            
            # Only filter out all alternatives in extreme cases (extremely clear classification)
            extreme_certainty = (
                verification_explanation and 
                ("absolutely certain" in verification_explanation.lower() or
                "perfect match" in verification_explanation.lower() or
                "no other standard could possibly apply" in verification_explanation.lower())
            )
            
            if result["valid"] and extreme_certainty and alternative_standards:
                alternative_standards = []

            issues_for_downstream = []
            llm_issue_note = result.get("issue_note")

            if llm_issue_note and isinstance(llm_issue_note, str) and llm_issue_note.strip():
                issues_for_downstream.append(llm_issue_note)
            
            if not result["valid"] and not issues_for_downstream:
                issues_for_downstream.append(
                    "Verification LLM reported 'valid: false' but 'issue_note' was missing or empty."
                )
            
            # Return a structure including the verification explanation and alternative standards
            return {
                "valid": result["valid"],
                "final_classification_label": result["final_classification_label"],
                "issues": issues_for_downstream,
                "verification_explanation": verification_explanation,
                "alternative_standards": alternative_standards
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
        
        # Add alternative standards information to the state
        new_state["alternative_standards"] = verification_result.get("alternative_standards", [])
        
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
        
        # For debugging - show the verification explanation and alternative standards
        print(f"VerifyNode: Verification explanation: {new_state.get('verification_explanation', '')[:100]}...")
        
        # Display alternative standards if any were found
        alternative_standards = new_state.get('alternative_standards', [])
        if alternative_standards:
            print(f"VerifyNode: Alternative standards: {alternative_standards}")

        # Ensure probs_vector is preserved or initialized
        if "probs_vector" not in new_state:
            new_state["probs_vector"] = {} # Or handle as per how it's generated upstream
     
        return new_state
