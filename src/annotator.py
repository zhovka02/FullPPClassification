from typing import List, Dict
from .llm_client import LLMClient
from .config import LABEL_DESCRIPTIONS
from .utils import parse_llm_json


class PrivacyPolicyAnnotator:
    def __init__(self, model_name: str = "openai:gpt-4o"):
        self.client = LLMClient(model=model_name)

    def build_system_prompt(self) -> str:
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
        system_message = self.build_system_prompt()

        user_message = (
            f"### DOCUMENT START\n\n{full_policy_text}\n\n### DOCUMENT END\n\n"
            "Extract all relevant sections as JSON."
        )

        raw_response = self.client.classify(system_message, user_message)
        return parse_llm_json(raw_response)