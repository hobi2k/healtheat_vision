"""Utility for initializing and configuring the logging system."""

import logging

def init_logger(name=None, force=True):
    """
    Initialize and configure the logger with a specified name.

    Parameters:
    - name (str): The name of the logger to retrieve. Defaults to None.
    - force (bool): Whether to force the reconfiguration of the logging system.
      In Jupyter/Notebook 환경에서는 True를 권장. (기본값: True)

    This function removes all existing handlers from the root logger and sets up
    a new logging configuration with INFO level and a specific message format.
    The 'force' parameter controls whether to forcibly reconfigure logging.

    Returns:
    - logging.Logger: The configured logger instance with the specified name.
    """
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)d) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=force,
    )
    return logging.getLogger(name)