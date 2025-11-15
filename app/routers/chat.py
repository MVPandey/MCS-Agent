"""Chat router for handling chat requests."""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion import Choice

from app.services.chat_model import ChatModelHyperparams, ChatModelService
from app.utils.exceptions import ChatModelError
from app.utils.logger import logger

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    messages: list[ChatCompletionMessageParam]
    hyperparams: ChatModelHyperparams | None = None
    model_name: str | None = None


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""

    choices: list[Choice]


@router.post("/", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat(request: ChatRequest) -> ChatResponse:
    """Chat with the model."""
    logger.info(
        "Received chat request",
        extra={
            "message_count": len(request.messages),
            "model_name": request.model_name,
            "has_hyperparams": request.hyperparams is not None,
        },
    )

    try:
        service = ChatModelService(
            hyperparams=request.hyperparams, model_name=request.model_name
        )
        completion = await service.chat(messages=request.messages)

        logger.info(
            "Chat request completed successfully",
            extra={
                "num_choices": len(completion.choices),
                "model_used": completion.model,
            },
        )

        return ChatResponse(choices=completion.choices)

    except ChatModelError as e:
        logger.error(
            "Chat request failed",
            extra={
                "error_type": type(e).__name__,
                "error_detail": str(e),
                "model_name": request.model_name,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        ) from e
