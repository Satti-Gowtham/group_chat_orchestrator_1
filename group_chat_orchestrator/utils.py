import logging
import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
from naptha_sdk.utils import get_logger

logger = get_logger(__name__)

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

def save_agent_results(run_id: str, agent_name: str, raw_result: Any, processed_data: Dict[str, Any], prompt: str = None) -> str:
    """Save agent results to a file.
    
    Args:
        run_id: Unique identifier for the run
        agent_name: Name of the agent
        raw_result: Raw output from the agent
        processed_data: Processed and structured data
        prompt: Optional prompt used for the agent
        
    Returns:
        str: Path to the saved file, or None if saving failed
    """
    try:
        results_dir = "agent_results"
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{results_dir}/{run_id}_{agent_name}_{timestamp}.json"
        
        output_data = {
            "run_id": run_id,
            "agent": agent_name,
            "timestamp": timestamp,
            "prompt": prompt,
            "raw_result": raw_result,
            "processed_data": processed_data
        }
        
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Saved {agent_name} results to {filename}")
        return filename
    except Exception as e:
        logger.error(f"Error saving {agent_name} results: {str(e)}")
        return None 