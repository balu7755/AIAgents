import requests

class ChatDeepSeek:
    """
    Wrapper for DeepSeek API (e.g. deepseek-chat or deepseek-reasoner).
    """
    def __init__(self, model: str, api_key: str, temperature: float = 1.0):
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.url = "https://api.deepseek.com/chat/completions"

    def invoke(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature
        }
        resp = requests.post(self.url,
                             headers={"Authorization": f"Bearer {self.api_key}"},
                             json=payload)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
