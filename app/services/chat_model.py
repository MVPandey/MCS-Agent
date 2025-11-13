"""Contains the chat model service class."""

from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from pydantic import BaseModel, Field

from app.utils.config import Config, config
from app.utils.exceptions import ChatModelError
from app.utils.logger import logger


class ChatModelHyperparams(BaseModel):
    """Hyperparameters for the chat model."""

    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    max_tokens: int = Field(default=1000, ge=1, le=8192)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    frequency_penalty: float = Field(default=0.0, ge=0.0, le=1.0)
    presence_penalty: float = Field(default=0.0, ge=0.0, le=1.0)
    n: int = Field(default=1, ge=1, le=10)
    stop: list[str] = Field(default=[], description="Stop tokens")
    stream: bool = Field(default=False, description="Stream the response")
    logprobs: int | None = Field(default=None, ge=0, le=10)


class ChatModelService:
    """Service class for the chat model."""

    def __init__(
        self,
        config: Config = config,
        hyperparams: ChatModelHyperparams | None = None,
        model_name: str | None = None,
    ):
        """Initialize the chat model service."""
        self.config = config
        self.llm_base_url = config.llm_base_url
        self.llm_api_key = config.llm_api_key
        self.llm_name = config.llm_name if not model_name else model_name
        self.hyperparams = hyperparams
        self._client: AsyncOpenAI = AsyncOpenAI(
            base_url=self.llm_base_url, api_key=self.llm_api_key.get_secret_value()
        )

        logger.info(
            "Initialized ChatModelService",
            extra={
                "model_name": self.llm_name,
                "base_url": str(self.llm_base_url),
                "hyperparams": self.hyperparams.model_dump()
                if self.hyperparams
                else None,
            },
        )

    async def chat(self, messages: list[ChatCompletionMessageParam]) -> ChatCompletion:
        """Chat with the model."""
        logger.debug(
            "Initiating chat completion",
            extra={
                "model": self.llm_name,
                "message_count": len(messages),
                "hyperparams": self.hyperparams.model_dump()
                if self.hyperparams
                else {},
            },
        )

        try:
            completion = await self._client.chat.completions.create(
                model=self.llm_name,
                messages=messages,
                **(self.hyperparams.model_dump() if self.hyperparams else {}),
            )

            logger.debug(
                "Chat completion received",
                extra={"completion": completion.model_dump()},
            )

            logger.info(
                "Chat completion successful",
                extra={
                    "model": completion.model,
                    "num_choices": len(completion.choices),
                    "finish_reasons": [
                        choice.finish_reason for choice in completion.choices
                    ]
                    if completion.choices
                    else [],
                    "usage": completion.usage.model_dump()
                    if completion.usage
                    else None,
                },
            )

            return completion

        except Exception as e:
            logger.error(
                "Chat completion failed",
                extra={
                    "model": self.llm_name,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                },
            )
            raise ChatModelError(f"Error chatting with model: {e}") from e
