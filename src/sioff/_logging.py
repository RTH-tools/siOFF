import sys

# The level setup_logging last chose, so the off-targets command can give its
# spawned workers the same one (loguru handlers do not cross spawn).
_level = "WARNING"


def setup_logging(verbose: bool) -> None:
    global _level
    # Imported here, not at module top: loguru costs ~0.4 s that `sioff --help`
    # (which never logs) should not pay.
    from loguru import logger

    logger.remove()
    if verbose:
        logger.add(
            sys.stderr,
            level="DEBUG",
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        )
        _level = "DEBUG"
    else:
        logger.add(sys.stderr, level="WARNING", format="<level>{message}</level>")
        _level = "WARNING"


def logging_level() -> str:
    """The level setup_logging chose: ``"DEBUG"`` under ``-v``, else ``"WARNING"``.

    ``"WARNING"`` too when setup_logging has not run (``off_targets.run``
    called from Python).
    """
    return _level
