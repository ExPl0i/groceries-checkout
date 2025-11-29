"""
Pricing management module.

Handles loading/saving prices from JSON configuration,
computing totals, and managing price data.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Tuple
from dataclasses import dataclass, asdict


logger = logging.getLogger(__name__)


@dataclass
class PriceConfig:
    """Container for price configuration data."""
    prices: Dict[str, float]
    
    def save(self, config_path: Path) -> None:
        """
        Save price configuration to JSON file.
        
        Args:
            config_path: Path to save the JSON file.
        """
        config_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(config_path, "w") as f:
                json.dump({"prices": self.prices}, f, indent=2)
            logger.info(f"Price config saved to {config_path}")
        except Exception as e:
            logger.error(f"Failed to save price config: {e}")
            raise
    
    @staticmethod
    def load(config_path: Path) -> "PriceConfig":
        """
        Load price configuration from JSON file.
        
        Args:
            config_path: Path to the JSON configuration file.
        
        Returns:
            PriceConfig object.
        """
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}")
            return PriceConfig.create_default()
        
        try:
            with open(config_path, "r") as f:
                data = json.load(f)
            prices = data.get("prices", {})
            logger.info(f"Price config loaded from {config_path}")
            return PriceConfig(prices=prices)
        except Exception as e:
            logger.error(f"Failed to load price config: {e}")
            return PriceConfig.create_default()
    
    @staticmethod
    def create_default() -> "PriceConfig":
        """
        Create default price configuration.
        
        Returns:
            PriceConfig with default prices for all classes.
        """
        default_prices = {
            "Alcohol": 5.0,
            "Candy": 1.0,
            "Canned Food": 2.5,
            "Chocolate": 2.0,
            "Dessert": 3.0,
            "Dried Food": 2.0,
            "Dried Fruit": 2.5,
            "Drink": 1.5,
            "Gum": 0.5,
            "Instant Drink": 1.5,
            "Instant Noodles": 1.0,
            "Milk": 1.2,
            "Personal Hygiene": 4.0,
            "Puffed Food": 1.0,
            "Seasoner": 0.8,
            "Stationery": 1.5,
            "Tissue": 1.0,
        }
        logger.info("Using default price configuration")
        return PriceConfig(prices=default_prices)


class PricingManager:
    """
    Manages product pricing and transaction calculations.
    """
    
    def __init__(self, config_path: str | Path = "config/prices.json"):
        """
        Initialize the pricing manager.
        
        Args:
            config_path: Path to the price configuration JSON file.
        """
        self.config_path = Path(config_path)
        self.config = PriceConfig.load(self.config_path)
    
    def get_price(self, class_name: str) -> float:
        """
        Get unit price for a product class.
        
        Args:
            class_name: Name of the product class.
        
        Returns:
            Unit price or 0.0 if not found.
        """
        return self.config.prices.get(class_name, 0.0)
    
    def set_price(self, class_name: str, price: float) -> None:
        """
        Set unit price for a product class.
        
        Args:
            class_name: Name of the product class.
            price: Unit price.
        """
        if price < 0:
            logger.warning(f"Negative price for {class_name} ignored")
            return
        self.config.prices[class_name] = price
    
    def get_all_prices(self) -> Dict[str, float]:
        """
        Get all prices.
        
        Returns:
            Dictionary of class names to unit prices.
        """
        return self.config.prices.copy()
    
    def update_all_prices(self, prices: Dict[str, float]) -> None:
        """
        Update all prices at once.
        
        Args:
            prices: Dictionary of class names to unit prices.
        """
        self.config.prices = prices.copy()
    
    def save_prices(self) -> None:
        """Save current prices to configuration file."""
        self.config.save(self.config_path)
    
    def compute_transaction(
        self, class_counts: Dict[str, int]
    ) -> Tuple[Dict[str, float], float]:
        """
        Compute transaction totals from class counts.
        
        Args:
            class_counts: Dictionary of class names to detection counts.
        
        Returns:
            Tuple of:
                - Dictionary of class names to subtotal prices
                - Total price for the transaction
        """
        subtotals: Dict[str, float] = {}
        total = 0.0
        
        for class_name, count in class_counts.items():
            price = self.get_price(class_name)
            subtotal = price * count
            subtotals[class_name] = subtotal
            total += subtotal
        
        return subtotals, round(total, 2)
