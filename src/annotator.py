from typing import List, Dict
from .llm_client import LLMClient
from .config import LABEL_DESCRIPTIONS
from .utils import parse_llm_json


class PrivacyPolicyAnnotator:
    """
    Class responsible for extracting privacy policy provisions from text.
    Uses an LLMClient to classify and extract information based on predefined legal categories.
    """
    def __init__(self, model_name: str = "openai:gpt-4o"):
        """
        Initializes the PrivacyPolicyAnnotator with a specific model.

        Args:
            model_name (str): The name of the LLM model to use for annotation.
        """
        self.client = LLMClient(model=model_name)

    def build_system_prompt(self) -> str:
        """
        Constructs the system prompt containing legal taxonomy, instructions, and expected output format.

        Returns:
            str: The fully constructed system prompt.
        """
        prompt = (
            "You are a Forensic Legal Auditor. Your goal is to extract privacy policy provisions "
            "that match specific legal categories exactly.\n\n"
        )

        prompt += "### 1. LEGAL TAXONOMY:\n"
        for label, desc in LABEL_DESCRIPTIONS.items():
            prompt += f"- **{label}**: {desc}\n"

        prompt += (
            "\n### 2. INSTRUCTIONS:\n"
            "1. **Analyze** the text segment by segment.\n"
            "2. **Identify** matches for the categories above.\n"
            "3. **Extract** the exact text verbatim. Do not summarize.\n"
            "4. **Reasoning**: Briefly explain why this text fits the category.\n"
            "5. **Exhaustiveness**: Extract ALL occurrences, even if repetitive.\n"
        )

        prompt += (
            "\n### 3. OUTPUT FORMAT:\n"
            "Return a strictly valid JSON list. Example:\n"
            "[\n"
            "  {\n"
            "    \"label\": \"Categories of Personal Information Collected\",\n"
            "    \"text\": \"We collect name, email, and IP address...\",\n"
            "    \"reasoning\": \"Explicit list of collected data types.\"\n"
            "  }\n"
            "]"
        )
        return prompt

    def annotate(self, full_policy_text: str) -> List[Dict[str, str]]:
        """
        Annotates the given privacy policy text by extracting relevant sections.

        Args:
            full_policy_text (str): The full text of the privacy policy to analyze.

        Returns:
            List[Dict[str, str]]: A list of dictionaries, where each dictionary represents an extracted provision
                                  with keys like 'label', 'text', and 'reasoning'.
        """
        system_message = self.build_system_prompt()

        user_message = (
            f"### DOCUMENT START\n\n{full_policy_text}\n\n### DOCUMENT END\n\n"
            "Extract all relevant sections as JSON."
        )

        raw_response = self.client.classify(system_message, user_message)
        return parse_llm_json(raw_response)