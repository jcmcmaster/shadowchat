from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Literal
import os


Role = Literal["system", "user", "assistant"]


@dataclass
class ChatMessage:
    """A message in a chat conversation.
    
    Attributes:
        role: The role of the message sender (system, user, or assistant).
        content: The text content of the message.
    """
    role: Role
    content: str


class LLM:
    """Abstract base class for Large Language Model implementations."""
    
    def chat(self, messages: List[ChatMessage], max_tokens: int = 512) -> str:
        """Generate a chat completion from a list of messages.
        
        Args:
            messages: List of conversation messages.
            max_tokens: Maximum number of tokens to generate.
            
        Returns:
            The generated response text.
            
        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError


class OpenAILLM(LLM):
    """OpenAI LLM implementation using the OpenAI API."""
    
    def __init__(self, model: str) -> None:
        """Initialize the OpenAI LLM client.
        
        Args:
            model: The OpenAI model name to use (e.g., 'gpt-4o-mini').
            
        Raises:
            RuntimeError: If the openai package is not installed or OPENAI_API_KEY is not set.
        """
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:  # pragma: no cover - optional dependency
            raise RuntimeError("openai package is required for provider=openai. Install with `pip install openai`.") from e
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in the environment.")
        self._client = OpenAI()
        self._model = model

    def chat(self, messages: List[ChatMessage], max_tokens: int = 512) -> str:
        """Generate a chat completion using the OpenAI API.
        
        This method handles different OpenAI models and API versions, including
        fallback strategies for token limit parameters and temperature settings.
        
        Args:
            messages: List of conversation messages.
            max_tokens: Maximum number of tokens to generate.
            
        Returns:
            The generated response text from the model.
        """
        # Convert to OpenAI format
        payload = [{"role": m.role, "content": m.content} for m in messages]
        # Prefer correct token param for newer models; be resilient across client versions
        try:
            from openai import BadRequestError  # type: ignore
        except Exception:
            BadRequestError = Exception  # type: ignore

        def _call_with_kwargs(include_temp: bool = True, **kwargs):
            api_kwargs = dict(model=self._model, messages=payload)  # type: ignore[arg-type]
            if include_temp:
                api_kwargs["temperature"] = 0.2
            api_kwargs.update(kwargs)
            return self._client.chat.completions.create(**api_kwargs)  # type: ignore[no-any-return]

        # Heuristic: gpt-5 models require max_completion_tokens.
        if str(self._model).lower().startswith("gpt-5"):
            try:
                return _call_with_kwargs(include_temp=False, max_completion_tokens=max_tokens).choices[0].message.content or ""
            except TypeError:
                # Older SDK: pass through using extra_body
                return _call_with_kwargs(include_temp=False, extra_body={"max_completion_tokens": max_tokens}).choices[0].message.content or ""
            except BadRequestError:
                # Retry via extra_body in case of server-side validation path
                return _call_with_kwargs(include_temp=False, extra_body={"max_completion_tokens": max_tokens}).choices[0].message.content or ""

        # Default path: try legacy param first, then upgrade on error
        try:
            return _call_with_kwargs(include_temp=True, max_tokens=max_tokens).choices[0].message.content or ""
        except BadRequestError as e:
            # Retry without temperature if it's the problem
            try:
                return _call_with_kwargs(include_temp=False, max_tokens=max_tokens).choices[0].message.content or ""
            except BadRequestError:
                # Server demanded max_completion_tokens
                try:
                    return _call_with_kwargs(include_temp=False, max_completion_tokens=max_tokens).choices[0].message.content or ""
                except TypeError:
                    return _call_with_kwargs(include_temp=False, extra_body={"max_completion_tokens": max_tokens}).choices[0].message.content or ""


def make_llm(provider: str, model: Optional[str] = None) -> LLM:
    """Factory function to create an LLM instance.
    
    Args:
        provider: The LLM provider to use ('auto' or 'openai').
        model: The specific model name to use. If None, uses a default.
        
    Returns:
        An LLM instance for the specified provider.
        
    Raises:
        ValueError: If the provider is not supported.
    """
    provider = (provider or "auto").lower()
    if provider == "auto" or provider == "openai":
        return OpenAILLM(model or "gpt-4o-mini")
    raise ValueError(f"Unknown provider: {provider}. Only 'openai' and 'auto' are supported.")


