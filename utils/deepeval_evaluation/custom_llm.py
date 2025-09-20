from deepeval.models import DeepEvalBaseLLM
from groq import Groq
import ollama
from openai import OpenAI
from dotenv import load_dotenv
import os


class OllamaDeepEvalLLM(DeepEvalBaseLLM):
    def __init__(self, model: str = "llama3.1:latest"):
        self.model = model

    def get_model_name(self) -> str:
        return f"Ollama-{self.model}"

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"]

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)



class GroqDeepEvalLLM(DeepEvalBaseLLM):
    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        load_dotenv()
        self.model = model

    def get_model_name(self) -> str:
        return f"Groq-{self.model}"

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        client = Groq()
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.3-70b-versatile",
        )

        return (chat_completion.choices[0].message.content)

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)


class OpenRouterDeepEvalLLM(DeepEvalBaseLLM):
    def __init__(self, model: str = "openai/gpt-oss-120b:free"):
        load_dotenv()
        self.model = model
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )

    def get_model_name(self) -> str:
        return f"OpenRouter-{self.model}"

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
        return completion.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)