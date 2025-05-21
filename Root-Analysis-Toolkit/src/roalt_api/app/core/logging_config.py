import os
from logging.config import dictConfig

def setup_logging():
    os.makedirs("logs", exist_ok=True)

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "level": "DEBUG",
            },
            "file": {
                "class": "logging.FileHandler",
                "formatter": "default",
                "filename": "logs/app.log",
                "level": "INFO",
            },
        },
        "root": {
            "handlers": ["console", "file"],
            "level": "DEBUG",
        },
    }
    dictConfig(logging_config)