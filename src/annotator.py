from typing import List, Dict
from .llm_client import LLMClient
from .config import LABEL_DESCRIPTIONS
from .utils import parse_llm_json


class PrivacyPolicyAnnotator:
    def __init__(self, model_name: str = "openai:gpt-4o"):
        self.client = LLMClient(model=model_name)

    def build_system_prompt(self) -> str:
        # 1. Persona & High-Level Goal
        prompt = (
            "You are a Meticulous Forensic Legal Auditor specializing in CCPA/CPRA compliance.\n"
            "Your goal is to perform a comprehensive, exhaustive extraction of privacy policy provisions.\n"
            "You will be penalized for missing any relevant text, no matter how redundant it seems.\n\n"
        )

        # 2. Taxonomy Injection
        prompt += "### 1. TARGET LEGAL CATEGORIES (The Taxonomy):\n"
        for label, desc in LABEL_DESCRIPTIONS.items():
            prompt += f"- **{label}**: {desc}\n"

        # 3. Detailed Rules (The "Chain of Thought" Logic)
        prompt += (
            "\n### 2. EXTRACTION RULES (CRITICAL):\n"
            "A. **EXHAUSTIVENESS IS MANDATORY**: Privacy policies often scatter information. "
            "If 'Data Collection' is mentioned in Section 1, Section 4, and Section 12, YOU MUST EXTRACT ALL THREE INSTANCES. "
            "Do not stop after finding the first match.\n"
            "B. **Include Full Context**: Do not just extract the keyword. Extract the full sentence or paragraph that provides the context. "
            "If a list of data types is provided, extract the *entire list*.\n"
            "C. **Exact Text Match**: Your extraction must be a verbatim copy of the text in the document. "
            "Do not summarize. Do not fix typos. Do not use '...' to skip text.\n"
            "D. **Headers & Definitions**: If a section header helps identify the category (e.g., 'Your Rights'), include the header in the extraction.\n"
            "E. **Negative Scope**: If the policy explicitly says 'We do not sell data', extract that under 'Categories of PI Sold'.\n"
        )

        # 4. JSON Output Specification
        prompt += (
            "\n### 3. OUTPUT FORMAT:\n"
            "Return a strictly valid JSON list of objects. No markdown fencing, no explanations.\n"
            "Format:\n"
            "[\n"
            "  {\n"
            "    \"label\": \"Exact Category Name from Taxonomy\",\n"
            "    \"text\": \"The exact substring extracted from the policy...\"\n"
            "  }, ...\n"
            "]"
        )
        return prompt

    def annotate(self, full_policy_text: str) -> List[Dict[str, str]]:
        system_message = self.build_system_prompt()

        # SANDWICH STRATEGY:
        # We place the instructions in the System Prompt, but we ALSO remind the model
        # of the critical constraints AFTER the massive block of text.
        user_message = (
            f"### DOCUMENT START\n\n{full_policy_text}\n\n### DOCUMENT END\n\n"
            "REMINDER: Review the document from start to finish. "
            "Extract EVERY instance of the categories defined in the system prompt. "
            "Do not miss scattered information. Return JSON only."
        )

        raw_response = self.client.classify(system_message, user_message)
        return parse_llm_json(raw_response)