"""
OpenRouter API Client for Heptanesian NER
"""

import requests
import json
import time
import os
from typing import Optional, Dict, Any, List

# Models available via OpenRouter
MODELS = {
    "llama-3.1-8b": "meta-llama/llama-3.1-8b-instruct",
    "llama-3.1-70b": "meta-llama/llama-3.1-70b-instruct",
    "claude-3.7-sonnet": "anthropic/claude-3.7-sonnet",
    "claude-4.5-sonnet": "anthropic/claude-sonnet-4",
    "gpt-4o": "openai/gpt-4o",
    "gemini-2.0-flash-lite": "google/gemini-2.0-flash-lite-001",
    "gemini-2.5-flash-lite": "google/gemini-2.5-flash-preview-05-20",
}


class OpenRouterClient:
    """Client for interacting with OpenRouter API"""

    def __init__(self, api_key: str = None):
        self.api_key = (api_key or os.environ.get("OPENROUTER_API_KEY", "")).strip()
        if not self.api_key:
            raise ValueError("OpenRouter API key required. Set OPENROUTER_API_KEY env var or pass to constructor.")

        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://heptanesian-ner.local",
            "X-Title": "Heptanesian NER System"
        }

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "claude-3.7-sonnet",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        retries: int = 3,
        retry_delay: float = 2.0
    ) -> Optional[str]:
        """
        Send a chat completion request to OpenRouter.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model key from MODELS dict or full model name
            temperature: Sampling temperature (0.0 for deterministic)
            max_tokens: Maximum tokens in response
            retries: Number of retry attempts
            retry_delay: Delay between retries in seconds

        Returns:
            Response text or None if failed
        """
        # Resolve model name
        model_name = MODELS.get(model, model)

        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        for attempt in range(retries):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=120
                )

                if response.status_code == 200:
                    result = response.json()
                    return result["choices"][0]["message"]["content"]
                elif response.status_code == 429:
                    # Rate limited, wait and retry
                    wait_time = retry_delay * (attempt + 1)
                    print(f"Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"Error {response.status_code}: {response.text}")
                    if attempt < retries - 1:
                        time.sleep(retry_delay)

            except requests.exceptions.Timeout:
                print(f"Timeout on attempt {attempt + 1}")
                if attempt < retries - 1:
                    time.sleep(retry_delay)
            except Exception as e:
                print(f"Error on attempt {attempt + 1}: {e}")
                if attempt < retries - 1:
                    time.sleep(retry_delay)

        return None

    def list_models(self) -> Dict[str, str]:
        """Return available models"""
        return MODELS.copy()


def test_client(api_key: str = None):
    """Test the OpenRouter client with a simple request"""
    client = OpenRouterClient(api_key)

    test_messages = [
        {"role": "user", "content": "Say 'Hello Heptanesian NER!' in Greek."}
    ]

    print("Testing OpenRouter connection...")
    for model_key in ["gemini-2.0-flash-lite", "llama-3.1-8b"]:
        print(f"\nTesting {model_key}...")
        response = client.chat_completion(test_messages, model=model_key, max_tokens=50)
        if response:
            print(f"✓ {model_key}: {response[:100]}")
        else:
            print(f"✗ {model_key}: Failed")


if __name__ == "__main__":
    import sys
    api_key = sys.argv[1] if len(sys.argv) > 1 else None
    test_client(api_key)
