import os
import threading
import json
from collections import deque
from typing import Optional, Dict, Any, Union, List
import aisuite as ai
# pip install json_repair
from json_repair import repair_json


class LLMClient:
    """
    Client wrapper for different LLM providers using AiSuite.
    """

    def __init__(self, model: str = "openai:gpt-4o", api_key: Optional[str] = None):
        if ":" in model:
            provider, model_name = model.split(":", 1)
        else:
            provider, model_name = "openai", model

        self.model = model
        self.provider = provider.lower()
        self.model_name = model_name
        self.api_key = api_key

        self._rate_limit_lock = threading.Lock()
        self._openrouter_timestamps = deque()

        # Provider-specific setup
        if self.provider == "gemini":
            provider_settings = {
                "openai": {
                    "api_key": os.getenv("GOOGLE_API_KEY"),
                    "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
                    "max_retries": 100,
                }
            }
            self.model = "openai:" + model_name
            self.client = ai.Client(provider_settings)

        elif self.provider == "openrouter":
            provider_settings = {
                "openai": {
                    "api_key": os.getenv("OPENROUTER_API_KEY"),
                    "base_url": "https://openrouter.ai/api/v1",
                    "max_retries": 100,
                }
            }
            self.model = "openai:" + model_name
            self.client = ai.Client(provider_settings)

        elif self.provider == "ollama":
            provider_settings = {
                "openai": {
                    "base_url": "https://f2ki-h100-1.f2.htw-berlin.de:11435/v1",
                    "max_retries": 100,
                }
            }
            self.model = "openai:" + model_name
            self.client = ai.Client(provider_settings)

        else:
            # Default: OpenAI
            self.client = ai.Client()
            if api_key:
                self.client.api_key = api_key

    def classify(self, system_prompt: str, user_prompt: str, response_format: Optional[Dict[str, Any]] = None) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self._call_model(messages, response_format)

    def get_completion(self, messages, response_format: Optional[Dict[str, Any]] = None) -> str:
        return self._call_model(messages, response_format)

    def _call_model(self, messages: List[Dict[str, str]], response_format: Optional[Dict[str, Any]] = None) -> str:
        if self.provider == "openai":
            temperature = 1.0
        else:
            temperature = 0.0

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }

        if response_format:
            kwargs["response_format"] = response_format

        try:
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM Error: {e}")
            return ""

    def parse_json(self, json_string: str) -> Union[Dict, List, None]:
        """
        Robustly parses a JSON string using json_repair.
        Returns the parsed object (dict or list) or None if parsing fails completely.
        """
        try:
            # json_repair returns the parsed object directly
            decoded_object = repair_json(json_string, return_objects=True)
            return decoded_object
        except Exception as e:
            print(f"JSON Parsing failed even with repair: {e}")
            return None