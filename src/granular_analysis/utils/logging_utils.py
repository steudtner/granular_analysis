"""Centralised logging helpers."""
import logging
import warnings


def suppress_noise() -> None:
    """Silence WARNING/INFO/DEBUG output from all loggers.

    logging.basicConfig() only works when no handlers exist yet.
    For already-initialised loggers (alphashape, root) we must
    set the level directly on each logger object.
    """
    # Root logger
    logging.getLogger().setLevel(logging.ERROR)

    # Known noisy third-party loggers
    for name in ("alphashape", "shapely", "root"):
        logging.getLogger(name).setLevel(logging.ERROR)

    # Suppress all existing handlers on all known loggers
    for logger in logging.Logger.manager.loggerDict.values():
        if isinstance(logger, logging.Logger):
            logger.setLevel(logging.ERROR)
            for handler in logger.handlers:
                handler.setLevel(logging.ERROR)

    # Also suppress Python warnings that mirror these messages
    warnings.filterwarnings("ignore", message="Singular matrix.*")
    warnings.filterwarnings("ignore", message=".*all points lying.*")
