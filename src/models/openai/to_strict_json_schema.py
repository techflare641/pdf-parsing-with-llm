# uses internal openai helpers

from typing import Any, Dict, Type, Union

from openai.lib._pydantic import to_strict_json_schema as _to_strict_json_schema
from pydantic import BaseModel, TypeAdapter


def to_strict_json_schema(
    model: Union[Type[BaseModel], TypeAdapter[Any]],
) -> Dict[str, Any]:
    """Converts the given base model or type adapter to a strict JSON schema.
    The current implementation is a wrapper around an internal OpenAI helper
    function, so this intermediary is used to centralize the import in case it
    changes later.
    """
    return _to_strict_json_schema(model)
