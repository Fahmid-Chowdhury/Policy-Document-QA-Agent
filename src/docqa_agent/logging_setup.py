import logging


def setup_logging(log_level: str) -> None:
    # Keep it simple and predictable.
    # We’ll expand later (file logs, JSON logs) only if needed.
    level = getattr(logging, log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    # Quiet noisy libs a bit (can tune later)
    logging.getLogger("chromadb").setLevel(logging.WARNING)