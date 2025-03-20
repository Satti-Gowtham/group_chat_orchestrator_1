import logging
import json
from pathlib import Path
from typing import Dict, Any


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def ensure_output_dir(output_dir: str) -> None:
    """Ensure output directory exists."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)


def write_output(content: str, output_path: str) -> None:
    """Write content to output file."""
    with open(output_path, 'w') as f:
        f.write(content) 