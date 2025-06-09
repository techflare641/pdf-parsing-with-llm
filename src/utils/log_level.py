from enum import IntEnum, auto


class LogLevel(IntEnum):
    """the log levels within the helper"""

    TRACE = auto()
    DEBUG = auto()
    INFO = auto()
    SUCCESS = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()
