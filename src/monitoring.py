import logging

logging.basicConfig(
    filename="training.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def log_training(episode, loss, reward):
    logging.info(f"Episode {episode}: Loss {loss:.4f}, Reward {reward:.2f}")

def log_warning(message):
    logging.warning(message)
