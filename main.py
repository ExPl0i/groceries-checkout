#!/usr/bin/env python3
"""
Retail Self-Checkout Control GUI Application
Main entry point for the PyQt5 desktop application.

This application runs a trained object detection model on retail checkout images
to recognize product classes and compute purchase totals.
"""

import sys
import logging
from pathlib import Path

from PyQt5.QtWidgets import QApplication

from ui.main_window import MainWindow


def setup_logging() -> None:
    """Configure logging for the application."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "app.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def main() -> int:
    """
    Initialize and run the PyQt5 application.
    
    Returns:
        Exit code of the application.
    """
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting Retail Checkout GUI Application")
    
    app = QApplication(sys.argv)
    
    try:
        window = MainWindow()
        window.show()
        exit_code = app.exec_()
        logger.info("Application closed successfully")
        return exit_code
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
