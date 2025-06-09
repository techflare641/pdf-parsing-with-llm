import asyncio
import random
from typing import Generic, Optional, Protocol, Tuple, TypeVar, overload

from utils.log_helper import LogHelper
from models.model_exception import (
    ModelNonRetryableException,
    ModelRetryableException,
)
from models.openai.completer import MessageCompleter

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
ModelT = TypeVar("ModelT")

class Parser(Generic[T_co], Protocol):
    def __call__(self, text: str, /) -> T_co:
        """Parses the given text; if it is malformed, raises ValueError"""


@overload
async def complete_and_parse_message(
    cancel_requested: asyncio.Event,
    *,
    always_try_once: bool,
    log: LogHelper,
    model: ModelT,
    system: Optional[str] = None,
    user: Optional[str] = None,
    image_base64: Optional[str] = None,
    completer: MessageCompleter[ModelT],
    parser: Parser[T],
    max_parser_retries: int = 10,
) -> Optional[Tuple[str, T]]: ...


@overload
async def complete_and_parse_message(
    *,
    cancel_requested: asyncio.Event,
    always_try_once: bool,
    log: LogHelper,
    model: ModelT,
    system: Optional[str] = None,
    user: Optional[str] = None,
    image_base64: Optional[str] = None,
    completer: MessageCompleter[ModelT],
    parser: None = None,
    max_parser_retries: int = 10,
) -> Optional[Tuple[str, None]]: ...


async def complete_and_parse_message(
    *,
    cancel_requested: asyncio.Event,
    always_try_once: bool,
    log: LogHelper,
    model: ModelT,
    system: Optional[str] = None,
    user: Optional[str] = None,
    image_base64: Optional[str] = None,
    completer: MessageCompleter[ModelT],
    parser: Optional[Parser[T]] = None,
    max_parser_retries: int = 10,
) -> Optional[Tuple[str, Optional[T]]]:
    """Uses the given model with the given system/user message and optional image to generate a response
    from the assistant, parses it with the given parser, and returns the raw and
    parsed completion.
    Starts with a temperature of 0 and increases it slightly if the first completion
    fails, then continues with that temperature for additional retries. After the max retries
    due to parsing errors, gives up and re-raises that exception.
    Args:
        cancel_requested (asyncio.Event): checked before each attempt, returning
            None if set
        always_try_once (bool): if False, `cancel_requested` is not checked before
            the first attempt. if True it is
        log (LogHelper): the logger to use
        model (ModelParam): the model to use
        system (str): the system message to use
        user (str): the user message to use
        image_base64 (str, optional): base64-encoded image data to include in the request
        completer (MessageCompleter[ModelT]): the completer to use
        parser (Parser[T]): the parser to use
        max_parser_retries (int): the maximum number of retries to attempt due to
            parsing errors
    Returns:
        Tuple[str, T]: the raw completion and the parsed completion
        None: if interrupted by `cancel_requested`
    """
    if not always_try_once and cancel_requested.is_set():
        log.debug("cancel requested before first attempt")
        return None

    temperature = 0.0
    attempt = 0
    parser_failures = 0
    if system is None:
        system = "You are a helpful assistant that follows instructions carefully. But you don't have any context now. Tell the user that you don't have any context now."
    if user is None:
        user = ""
    while True:
        if parser_failures >= max_parser_retries:
            raise ModelNonRetryableException("too many parsing failures, giving up")

        if attempt == 1:
            log.debug(f"increasing temperature from {temperature} to 0.1")
            temperature = 0.1

        if attempt > 0:
            sleep_time = 2 ** min(attempt, 7) + 2 * random.random()
            log.debug(f"waiting {sleep_time:.2f} seconds before attempt {attempt + 1}")
            try:
                await asyncio.wait_for(cancel_requested.wait(), timeout=sleep_time)
                log.debug("interrupted by cancel request")
                return None
            except asyncio.TimeoutError:
                pass

        attempt += 1
        log.debug(f"attempt {attempt} with temperature {temperature}")
        try:
            completion = await completer(
                log=log,
                model=model,
                system=system,
                user=user,
                image_base64=image_base64,
                temperature=temperature,
            )
        except ModelRetryableException as e:
            log.exception(e)
            continue

        log.debug(f"COMPLETION:\n{completion}")
        if parser is None:
            return completion, None
        else:
            try:
                parsed = parser(completion)
                log.debug(f"{parsed=}")
                return completion, parsed
            except ValueError as e:
                log.warning("response is not properly formatted")
                log.exception(e)
                parser_failures += 1
                continue
