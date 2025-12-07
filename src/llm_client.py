import os
import threading
from collections import deque
from typing import Optional
import aisuite as ai

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

    def classify(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()