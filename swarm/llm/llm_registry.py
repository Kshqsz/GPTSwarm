from typing import Optional
import os
from class_registry import ClassRegistry

from swarm.llm.llm import LLM


class LLMRegistry:
    registry = ClassRegistry()

    @classmethod
    def register(cls, *args, **kwargs):
        return cls.registry.register(*args, **kwargs)
    
    @classmethod
    def keys(cls):
        return cls.registry.keys()

    @classmethod
    def get(cls, model_name: Optional[str] = None) -> LLM:
        if model_name is None:
            model_name = (
                os.getenv("OPENAI_MODEL_NAME")
                or os.getenv("DEEPSEEK_MODEL_NAME")
                or os.getenv("DEFAULT_MODEL_NAME")
                or "gpt-4-1106-preview"
            )

        if model_name == "deepseek":
            model_name = os.getenv("DEEPSEEK_MODEL_NAME") or "deepseek-chat"

        if model_name == 'mock':
            model = cls.registry.get(model_name)
        else: # any version of GPTChat like "gpt-4-1106-preview"
            model = cls.registry.get('GPTChat', model_name)

        return model
