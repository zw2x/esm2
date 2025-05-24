import os
import dotenv

dotenv.load_dotenv()
import logging

_LOG_LEVEL = getattr(logging, (os.environ["ESM2_LOG_LEVEL"]).upper())


def _setup_root_logger(log_level=_LOG_LEVEL) -> None:
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


_setup_root_logger()


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    log = logging.getLogger(name)
    # print(f"Logger name: {name} {_LOG_LEVEL}")
    log.setLevel(_LOG_LEVEL)
    return log
