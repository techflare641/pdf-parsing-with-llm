"""Retry helper"""

import asyncio
import random
from typing import Awaitable, Callable, Optional, Sequence, Type, TypeVar

from loguru import logger

T = TypeVar("T")


async def retry(
    fn: Callable[[], Awaitable[T]],
    /,
    *,
    on: Sequence[Type[BaseException]],
    cancel_requested: asyncio.Event,
    always_try_once: bool,
    max_attempts: Optional[int] = None,
    max_delay_pow: int = 4,
) -> Optional[T]:
    """Retry a function until it succeeds or the max attempts are reached

    Usage:
    ```py
    result = await retry(
        partial(fn, foo='bar'),
        on=(MyException,),
        cancel_requested=cancel_requested,
        always_try_once=True
    )
    ```

    Args:
        fn (async def fn() -> T): the function to retry
        on (Sequence[Type[BaseException]]): the exceptions to catch
        cancel_requested (asyncio.Event): the event to watch for cancellation
        always_try_once (bool): True to try at least once even if cancel_requested
            is set before this is called, False to return immediately in that case
        max_attempts (Optional[int]): the maximum number of attempts to make; None
            to retry indefinitely
        max_delay_pow (int): the maximum exponent to use for the delay calculation

    Returns:
        T: the result of the function

    Raises:
        any: any exception that the function raises
        None: cancel requested
    """
    if not always_try_once and cancel_requested.is_set():
        return None

    catchable = tuple(on) if not isinstance(on, tuple) else on

    attempt = -1
    while True:
        attempt += 1
        try:
            return await fn()
        except catchable:
            if max_attempts is not None and attempt >= max_attempts:
                logger.debug("max attempts reached")
                raise

            if cancel_requested.is_set():
                logger.debug("cancel requested")
                return None

            delay = 2 ** min(attempt, max_delay_pow) + random.random()
            logger.exception(f"retrying in {delay:.2f} seconds")

            try:
                await asyncio.wait_for(
                    cancel_requested.wait(),
                    timeout=delay,
                )
                logger.debug("cancel requested during sleep")
                return None
            except asyncio.TimeoutError:
                logger.debug("retrying after timeout")
        except BaseException as e:
            logger.debug(f"not retryable ({type(e)!r})")
            raise
