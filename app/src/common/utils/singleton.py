from typing import Any, Dict


class Singleton(type):
    _instances: Dict["Singleton", "Singleton"] = {}

    def __call__(cls, *args: Any, **kwargs: Any) -> "Singleton":
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance

        return cls._instances[cls]

    @staticmethod
    def drop() -> None:
        Singleton._instances = {}
