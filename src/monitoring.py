import logging
from typing import Union, Any

# Configure the logging system for the entire application
logging.basicConfig(
    filename="training.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def log_training(episode: int, loss: float, reward: float) -> None:
    """
    Log training information for a specific episode.
    
    Records the episode number, loss value, and reward value to the configured log file
    with INFO level severity.
    
    Args:
        episode: The current training episode number
        loss: The loss value from the training step
        reward: The reward value received from the environment
    """
    logging.info(f"Episode {episode}: Loss {loss:.4f}, Reward {reward:.2f}")

def log_warning(message: str) -> None:
    """
    Log a warning message to the configured log file.
    
    Used to record potential issues or unexpected behavior during execution.
    
    Args:
        message: The warning message to be logged
    """
    logging.warning(message)

def log_error(message: str, exception: Exception = None) -> None:
    """
    Log an error message with optional exception details.
    
    Used to record errors that occur during execution.
    
    Args:
        message: The error message to be logged
        exception: Optional exception object to include in the log
    """
    if exception:
        logging.error(f"{message}: {str(exception)}", exc_info=True)
    else:
        logging.error(message)

def log_metric(metric_name: str, value: Union[float, int]) -> None:
    """
    Log a training or evaluation metric.
    
    Records a named metric and its value to the configured log file.
    
    Args:
        metric_name: Name of the metric being recorded
        value: Value of the metric (numeric)
    """
    logging.info(f"Metric - {metric_name}: {value}")

def set_log_level(level: int) -> None:
    """
    Update the logging level for the application.
    
    Args:
        level: The logging level to set (use logging constants like logging.DEBUG)
    """
    logging.getLogger().setLevel(level)