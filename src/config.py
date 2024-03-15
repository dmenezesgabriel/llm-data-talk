import logging.config
import os

from decouple import config  # type: ignore

from src.common.utils.singleton import Singleton


class Config(metaclass=Singleton):
    LOG_LEVEL = config("LOG_LEVEL", default="INFO", cast=str)
    DATABASE_URI = config(
        "DATABASE_URI",
        default="../data/Chinook.db",
        cast=str,
    )
    OPENAI_API_KEY = config("OPENAI_API_KEY", cast=str)


class DevelopmentConfig(Config):
    LOG_LEVEL = config("LOG_LEVEL", default="DEBUG", cast=str)


class StagingConfig(Config):
    pass


class ProductionConfig(Config):
    pass


def config_factory(environment: str) -> type[Config]:
    configs = {
        "development": DevelopmentConfig,
        "staging": StagingConfig,
        "production": ProductionConfig,
    }
    return configs[environment]


def get_config() -> type[Config]:
    environment = os.getenv("ENVIRONMENT", "development")
    app_config = config_factory(environment)

    LOGGING = {
        "version": 1,
        "disable_existing_loggers": True,
        "formatters": {
            "standard": {
                "format": (
                    "[%(asctime)s] %(levelname)s "
                    "[%(name)s.%(funcName)s:%(lineno)d] "
                    "%(message)s"
                ),
                "datefmt": "%Y-%m-%d %H:%M:%S",
            }
        },
        "handlers": {
            "stdout_logger": {
                "formatter": "standard",
                "class": "logging.StreamHandler",
            }
        },
        "loggers": {
            "": {  # root
                "level": app_config.LOG_LEVEL,
                "handlers": ["stdout_logger"],
                "propagate": False,
            }
        },
    }
    logging.config.dictConfig(LOGGING)
    return app_config
