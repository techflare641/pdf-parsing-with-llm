from functools import wraps
from typing import Callable, Coroutine, TypeVar, cast

from utils.error import handle_error
from utils.log_helper import LogHelper

TFunc = TypeVar("TFunc", bound=Callable[..., Coroutine])


def log_exceptions(
    *, log: LogHelper, message: Callable[[], str]
) -> Callable[[TFunc], TFunc]:
    """Decorator-generator which catches exceptions, logs them, then raises them again

    Usage:

    ```py
    @log_exceptions(log=log, message=lambda: "error message")
    async def my_func() -> None:
        1 / 0

    my_func()  # logs the exception and raises it again
    ```
    """

    def decorator(func: TFunc) -> TFunc:
        @wraps(func)
        async def wrapper(*args, **kwargs):  # type: ignore
            try:
                return await func(*args, **kwargs)
            except BaseException as exc:
                log.exception(exc)
                msg = message()
                msg = f"{args=}; {kwargs=}; {msg}"
                await handle_error(exc, extra_info=msg)
                raise

        return cast(TFunc, wrapper)

    return decorator
