"""
UI Settings Dialog module.

Implements a modal dialog for editing product prices.
"""

import logging
from typing import Dict

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QDoubleSpinBox, QHeaderView, QMessageBox, QAbstractItemView,
)
from PyQt5.QtCore import Qt


logger = logging.getLogger(__name__)


class SettingsDialog(QDialog):
    """
    Modal dialog for editing product prices.
    
    Users can view and modify unit prices for each product class.
    Changes are saved to the configuration file upon confirmation.
    """
    
    def __init__(self, prices: Dict[str, float], parent=None):
        """
        Initialize the settings dialog.
        
        Args:
            prices: Dictionary of class names to unit prices.
            parent: Parent widget.
        """
        super().__init__(parent)
        self.prices = prices.copy()
        self.original_prices = prices.copy()
        
        self.setWindowTitle("Settings - Edit Prices")
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)
        
        self._init_ui()
    
    def _init_ui(self) -> None:
        """Initialize the user interface."""
        layout = QVBoxLayout()
        
        # Create table widget for prices
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Product Class", "Unit Price ($)"])
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch
        )
        self.table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeToContents
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        
        # Populate table with prices
        self.table.setRowCount(len(self.prices))
        for row, (class_name, price) in enumerate(self.prices.items()):
            # Class name (read-only)
            class_item = QTableWidgetItem(class_name)
            class_item.setFlags(class_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 0, class_item)
            
            # Price (editable)
            price_item = QTableWidgetItem(f"{price:.2f}")
            price_item.setFlags(price_item.flags() | Qt.ItemIsEditable)
            self.table.setItem(row, 1, price_item)
        
        layout.addWidget(self.table)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        reset_button = QPushButton("Reset to Defaults")
        reset_button.clicked.connect(self._reset_to_defaults)
        button_layout.addWidget(reset_button)
        
        button_layout.addStretch()
        
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self._on_ok_clicked)
        button_layout.addWidget(ok_button)
        
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def _reset_to_defaults(self) -> None:
        """Reset all prices to original values."""
        self.prices = self.original_prices.copy()
        self._update_table()
        logger.info("Price settings reset to defaults")
    
    def _update_table(self) -> None:
        """Update table display from prices dictionary."""
        for row, (class_name, price) in enumerate(self.prices.items()):
            price_item = QTableWidgetItem(f"{price:.2f}")
            price_item.setFlags(price_item.flags() | Qt.ItemIsEditable)
            self.table.setItem(row, 1, price_item)
    
    def _on_ok_clicked(self) -> None:
        """Handle OK button click - validate and save prices."""
        try:
            # Read prices from table
            new_prices = {}
            for row in range(self.table.rowCount()):
                class_item = self.table.item(row, 0)
                price_item = self.table.item(row, 1)
                
                if class_item is None or price_item is None:
                    continue
                
                class_name = class_item.text()
                try:
                    price = float(price_item.text())
                    if price < 0:
                        raise ValueError("Price cannot be negative")
                    new_prices[class_name] = price
                except ValueError as e:
                    QMessageBox.warning(
                        self,
                        "Invalid Price",
                        f"Invalid price for {class_name}: {e}",
                    )
                    return
            
            self.prices = new_prices
            logger.info("Price settings updated")
            self.accept()
            
        except Exception as e:
            logger.error(f"Error saving prices: {e}")
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to save prices: {e}",
            )
    
    def get_prices(self) -> Dict[str, float]:
        """
        Get the updated prices dictionary.
        
        Returns:
            Dictionary of class names to unit prices.
        """
        return self.prices.copy()
