from typing import Generic, Optional, Protocol, TypeVar

from utils.log_helper import LogHelper

ModelT_contra = TypeVar("ModelT_contra", contravariant=True)


class MessageCompleter(Generic[ModelT_contra], Protocol):
    async def __call__(
        self,
        /,
        *,
        log: LogHelper,
        model: ModelT_contra,
        system: str,
        user: str,
        temperature: float,
        image_base64: Optional[str] = None,
    ) -> str:
        """Generates the assistant message for the given system and user message"""
