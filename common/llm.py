import os
from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from common.logging_utils import get_logging_callbacks, multi_model_var, agent_name_var


class DynamicChatOpenAI(ChatOpenAI):
    _default_max_tokens: int | None = None

    @staticmethod
    def _multi_model_max_tokens() -> int:
        try:
            return int(os.getenv("MULTI_MODEL_MAX_TOKENS", "1200"))
        except ValueError:
            return 1200

    def __init__(self, **kwargs: Any) -> None:
        # Set default values for OpenRouter if not specified
        kwargs.setdefault("openai_api_key", os.getenv("OPENROUTER_API_KEY"))
        kwargs.setdefault("openai_api_base", "https://openrouter.ai/api/v1")
        kwargs.setdefault("temperature", 0.3)
        if "model" not in kwargs and "model_name" not in kwargs:
            kwargs["model"] = os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-5")
        
        # Inject our custom logging callbacks automatically
        callbacks = kwargs.get("callbacks") or []
        for cb in get_logging_callbacks():
            if not any(isinstance(c, type(cb)) for c in callbacks):
                callbacks.append(cb)
        kwargs["callbacks"] = callbacks
        
        super().__init__(**kwargs)
        self._default_max_tokens = self.max_tokens

    def _update_model(self) -> None:
        is_multi_model = multi_model_var.get()
        agent_name = agent_name_var.get() or os.getenv("AGENT_NAME", "")
        
        # Check if we should use Groq for specialist agents
        if is_multi_model and agent_name in ["tax_agent", "compliance_agent", "privacy_agent"]:
            self.max_tokens = self._multi_model_max_tokens()
            groq_key = os.getenv("GROQ_API_KEY")
            if groq_key:
                # Use Groq directly
                model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
                self.model_name = model
                self.openai_api_key = SecretStr(groq_key)
                self.openai_api_base = "https://api.groq.com/openai/v1"
                self.client = None
                self.root_client = None
                self.async_client = None
                self.root_async_client = None
                self.validate_environment()
                return
            else:
                # Fallback to OpenRouter fast model
                model = os.getenv("FAST_OPENROUTER_MODEL", "openai/gpt-4o-mini")
                self.model_name = model
                openrouter_key = os.getenv("OPENROUTER_API_KEY")
                self.openai_api_key = SecretStr(openrouter_key) if openrouter_key else None
                self.openai_api_base = "https://openrouter.ai/api/v1"
                self.client = None
                self.root_client = None
                self.async_client = None
                self.root_async_client = None
                self.validate_environment()
                return

        # Default: Use premium model on OpenRouter
        self.max_tokens = self._default_max_tokens
        model = os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-5")
        self.model_name = model
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        self.openai_api_key = SecretStr(openrouter_key) if openrouter_key else None
        self.openai_api_base = "https://openrouter.ai/api/v1"
        self.client = None
        self.root_client = None
        self.async_client = None
        self.root_async_client = None
        self.validate_environment()

    def _generate(self, *args: Any, **kwargs: Any) -> Any:
        self._update_model()
        return super()._generate(*args, **kwargs)
        
    async def _agenerate(self, *args: Any, **kwargs: Any) -> Any:
        self._update_model()
        return await super()._agenerate(*args, **kwargs)


def get_llm() -> ChatOpenAI:
    """Return a dynamic ChatOpenAI client wrapper pointed at OpenRouter or Groq."""
    return DynamicChatOpenAI()