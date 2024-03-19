import logging
import time
from functools import wraps
from typing import Callable

from src.config import get_config

config = get_config()

logger = logging.getLogger(__name__)


def log_time(func: Callable) -> None:
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        message = f"Function {func.__module__}.{func.__name__} "
        message += f"took {end - start} seconds to execute"
        print(message)
        return result

    return wrapper
