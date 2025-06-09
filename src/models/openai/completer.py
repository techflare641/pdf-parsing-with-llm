from typing import (
    TYPE_CHECKING,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
)
import os

import openai
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion_content_part_image_param import (
    ChatCompletionContentPartImageParam,
)
from openai.types.chat.chat_completion_content_part_param import (
    ChatCompletionContentPartParam,
)
from openai.types.chat.chat_completion_content_part_text_param import (
    ChatCompletionContentPartTextParam,
)
from openai.types.chat.completion_create_params import ResponseFormat
from openai.types.shared_params.chat_model import ChatModel
from pydantic import BaseModel

from utils.log_helper import LogHelper
from models.generic.completer import MessageCompleter
from models.model_exception import (
    ModelNonRetryableException,
    ModelRetryableException,
)
from models.openai.to_strict_json_schema import to_strict_json_schema

ModelT_contra = TypeVar("ModelT_contra", contravariant=True)
OpenAIModelParam = Union[ChatModel, str]

StructuredOutputT = TypeVar("StructuredOutputT", bound=BaseModel)

async def complete_message_openai(
    model: OpenAIModelParam,
    *,
    log: LogHelper,
    system: Optional[str] = None,
    user: Optional[str] = None,
    image_base64: Optional[str] = None,
    temperature: float,
    structured_output: bool = False,
    output_model: Optional[Type[StructuredOutputT]] = None,
) -> str:
    """Generates the assistant message for the given system and user messages with optional image

    If `structured_output` is True then the LLM is constrained to give its
    format to match `output_model`, which can be substantially slower but
    more reliable.

    the `structured_output` bool is used instead of just using if `output_model`
    is None to improve mypy's type inference.
    """
    log.debug(
        f"Generating completion using OpenAI's {model} model @ {temperature=} with {'image' if image_base64 else 'no image'}"
    )
    openai_client = openai.AsyncOpenAI(api_key="")
    if system is None:
        system = "You are a helpful assistant that follows instructions carefully."
    if user is None:
        user = ""
    if not structured_output:
        output_model = None
    elif output_model is None:
        structured_output = False

    try:
        clean_system = system.strip()
        clean_user = user.rstrip().strip()

        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": clean_system}
        ]

        # Handle image if provided
        if image_base64 is not None:
            content_parts: List[ChatCompletionContentPartParam] = [
                ChatCompletionContentPartTextParam(type="text", text=clean_user),
                ChatCompletionContentPartImageParam(
                    type="image_url",
                    image_url={"url": f"data:image/png;base64,{image_base64}"},
                ),
            ]
            messages.append({"role": "user", "content": content_parts})
        else:
            messages.append({"role": "user", "content": clean_user})

        if output_model is None:
            completion = await openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=8192,
            )
        else:
            response_format: ResponseFormat = {
                "type": "json_schema",
                "json_schema": {
                    "schema": to_strict_json_schema(output_model),
                    "name": output_model.__name__,
                    "strict": True,
                },
            }
            completion = await openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=8192,
                response_format=response_format,
            )

        if not completion.choices:
            raise ModelRetryableException("no completion choices available")

        assistant_message = completion.choices[0].message
        if not assistant_message.content:
            raise ModelRetryableException("no completion content to use")

        return assistant_message.content

    except openai.BadRequestError as e:
        log.exception(e)
        raise ModelNonRetryableException() from e
    except openai.APIConnectionError as e:
        log.exception(e)
        raise ModelRetryableException() from e
    except openai.RateLimitError as e:
        log.exception(e)
        raise ModelRetryableException() from e
    except openai.APIError as e:
        log.exception(e)
        raise ModelRetryableException() from e


if TYPE_CHECKING:
    _a: MessageCompleter[ChatModel] = complete_message_openai
