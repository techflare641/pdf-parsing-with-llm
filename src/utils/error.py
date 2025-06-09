import asyncio
import gzip
import io
import os
import secrets
import socket
import time
import traceback
from collections import deque
from typing import TYPE_CHECKING, Dict, Optional

import aiohttp
from loguru import logger

async def handle_error(
    exc: BaseException,
    *,
    extra_info: Optional[str] = None,
    stop_requested: Optional[asyncio.Event] = None,
) -> None:
    """Handles a generic error"""
    fmtd_exc = "\n".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    ctx = f"{extra_info=}\n\n{fmtd_exc}"
    await _handle_error_with_context(ctx, stop_requested=stop_requested)


async def handle_contextless_error(
    *,
    extra_info: Optional[str] = None,
    stop_requested: Optional[asyncio.Event] = None,
) -> None:
    """Handles an error that was found programmatically, i.e., which didn't cause an
    actual exception object to be raised. This will produce a stack trace and include
    the extra information.
    """
    full_exc = io.StringIO()
    full_exc.write(f"{extra_info=}\n\n")
    traceback.print_stack(file=full_exc)
    await _handle_error_with_context(
        full_exc.getvalue(), stop_requested=stop_requested
    )


async def _handle_error_with_context(
    ctx: str, /, *, stop_requested: Optional[asyncio.Event]
) -> None:
    # GITHUB_REPOSITORY is most likely not to be set in tests
    github_repository_full_name = os.environ.get(
        "GITHUB_REPOSITORY", "geniosai/unknown"
    )
    preview = f"Error in {github_repository_full_name}"

    message = f"{socket.gethostname()} {preview}\n\n{ctx}\n"
    logger.error(message)

    if stop_requested is None:
        stop_requested = asyncio.Event()


RECENT_WARNINGS: Dict[str, deque] = dict()  # deque[float] is not available on prod
"""Maps from a warning identifier to a deque of timestamps of when the warning was sent."""

WARNING_RATELIMIT_INTERVAL = 60 * 60
"""The interval in seconds we keep track of warnings for"""

MAX_WARNINGS_PER_INTERVAL = 5
"""The maximum number of warnings to send per interval for a particular identifier"""


async def handle_warning(
    identifier: str,
    text: str,
    exc: Optional[BaseException] = None,
    is_urgent: bool = False,
    stop_requested: Optional[asyncio.Event] = None,
) -> bool:
    """Sends a warning to slack, with basic ratelimiting

    Args:
        identifier (str): An identifier for ratelimiting the warning
        text (str): The text to send
        exc (Exception, None): If an exception occurred, formatted and added to
          the text appropriately
        is_urgent (bool): If true, the message may be sent to a more urgent channel
        stop_requested (asyncio.Event, None): if not None and set, sending the warning
            may be skipped to ensure a swift shutdown

    Returns:
        true if the warning was sent, false if it was suppressed
    """
    if exc is not None:
        logger.warning(
            f"{identifier}: full exception trace\n\n:{traceback.format_exc()}"
        )
        text += (
            "\n\n```"
            + "\n".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)[-5:]
            )
            + "```"
        )
    else:
        logger.warning(f"{identifier}: full stack trace\n\n:{traceback.format_stack()}")

    logger.warning(f"{identifier}: {text}")

    if identifier not in RECENT_WARNINGS:
        RECENT_WARNINGS[identifier] = deque()

    recent_warnings = RECENT_WARNINGS[identifier]
    now = time.time()

    while recent_warnings and recent_warnings[0] < now - WARNING_RATELIMIT_INTERVAL:
        recent_warnings.popleft()

    if len(recent_warnings) >= MAX_WARNINGS_PER_INTERVAL:
        logger.debug(f"warning suppressed (ratelimit): {identifier}")
        return False

    recent_warnings.append(now)
    total_warnings = len(recent_warnings)

    github_repository_full_name = os.environ.get("GITHUB_REPOSITORY", "<unknown repo>")
    message = f"WARNING: `{identifier}` in {github_repository_full_name} (warning {total_warnings}/{MAX_WARNINGS_PER_INTERVAL} per {WARNING_RATELIMIT_INTERVAL} seconds for `{socket.gethostname()}` - pid {os.getpid()})\n\n{text}"
    preview = f"WARNING: {identifier} in {github_repository_full_name}"

    if stop_requested is None:
        stop_requested = asyncio.Event()

    logger.error(f"{preview=} {message=}")