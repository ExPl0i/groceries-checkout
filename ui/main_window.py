"""
Main window UI module.

Implements the primary application window with image display,
detection results, and transaction logging.
"""

import logging
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QPushButton, QTableWidget, QTableWidgetItem, QFileDialog,
    QMessageBox, QToolBar, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QStatusBar, QScrollArea, QAction,
)
from PyQt5.QtGui import QIcon, QPixmap, QFont, QColor
from PyQt5.QtCore import Qt, QTimer, QSize

from model import ObjectDetector, Detection
from pricing import PricingManager
from ui.settings_dialog import SettingsDialog
from utils.image_utils import pil_to_qpixmap, draw_detections


logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """
    Main application window for retail checkout detection control.
    
    Displays images with detected objects, computes purchase totals,
    and logs transactions to CSV.
    """
    
    def __init__(self):
        """Initialize the main window and all UI components."""
        super().__init__()
        
        self.setWindowTitle("Retail Checkout Control - Object Detection")
        self.setMinimumSize(1400, 900)
        
        # Initialize core components
        self._init_detector()
        self._init_pricing()
        self._init_logging()
        
        # Current state
        self.current_image_path: Optional[Path] = None
        self.current_detections: List[Detection] = []
        self.current_image_pixmap: Optional[QPixmap] = None
        
        # Build UI
        self._init_ui()
        self._create_menu_bar()
        self._create_toolbar()
        self._create_status_bar()
        
        logger.info("Application initialized successfully")
    
    def _init_detector(self) -> None:
        """Initialize the object detection model."""
        try:
            model_path = Path("models/detector.pt")
            if not model_path.exists():
                logger.warning(f"Model file not found at {model_path}")
                self.detector = None
            else:
                self.detector = ObjectDetector(str(model_path))
                logger.info("Object detector initialized")
        except Exception as e:
            logger.error(f"Failed to initialize detector: {e}")
            self.detector = None
    
    def _init_pricing(self) -> None:
        """Initialize the pricing manager."""
        try:
            self.pricing_manager = PricingManager("config/prices.json")
            logger.info("Pricing manager initialized")
        except Exception as e:
            logger.error(f"Failed to initialize pricing: {e}")
            self.pricing_manager = PricingManager()
    
    def _init_logging(self) -> None:
        """Initialize transaction logging."""
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / "transactions.csv"
        
        # Create CSV file with headers if it doesn't exist
        if not self.log_file.exists():
            try:
                with open(self.log_file, "w", newline="") as f:
                    headers = ["DateTime", "ImagePath", "NumObjects"]
                    # Add per-class count columns
                    headers.extend(ObjectDetector.CLASSES)
                    headers.append("TotalPrice")
                    writer = csv.DictWriter(f, fieldnames=headers)
                    writer.writeheader()
                logger.info(f"Created transaction log: {self.log_file}")
            except Exception as e:
                logger.error(f"Failed to create transaction log: {e}")
    
    def _init_ui(self) -> None:
        """Initialize the main UI layout."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        # Left side: Image display with graphics view
        self.graphics_scene = QGraphicsScene()
        self.graphics_view = QGraphicsView(self.graphics_scene)
        self.graphics_view.setMinimumWidth(800)
        self.graphics_pixmap_item: Optional[QGraphicsPixmapItem] = None
        
        # Right side: Control panel
        right_panel = self._create_right_panel()
        
        main_layout.addWidget(self.graphics_view, stretch=2)
        main_layout.addWidget(right_panel, stretch=1)
        
        central_widget.setLayout(main_layout)
    
    def _create_right_panel(self) -> QWidget:
        """
        Create the right-side control panel.
        
        Returns:
            Widget containing detection results and transaction info.
        """
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Detection Results")
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # Results table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels(
            ["Class", "Count", "Unit Price", "Subtotal"]
        )
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        layout.addWidget(self.results_table)
        
        # Total price section
        total_layout = QHBoxLayout()
        total_label = QLabel("Total Price: ")
        total_label.setFont(title_font)
        self.total_price_label = QLabel("$0.00")
        self.total_price_label.setFont(title_font)
        self.total_price_label.setStyleSheet("color: green; font-weight: bold;")
        total_layout.addWidget(total_label)
        total_layout.addWidget(self.total_price_label)
        total_layout.addStretch()
        layout.addLayout(total_layout)
        
        # Object count section
        count_layout = QHBoxLayout()
        count_label = QLabel("Total Objects: ")
        self.object_count_label = QLabel("0")
        count_layout.addWidget(count_label)
        count_layout.addWidget(self.object_count_label)
        count_layout.addStretch()
        layout.addLayout(count_layout)
        
        # Buttons
        recalc_button = QPushButton("Recalculate")
        recalc_button.clicked.connect(self._recalculate_totals)
        layout.addWidget(recalc_button)
        
        log_button = QPushButton("Log Transaction")
        log_button.clicked.connect(self._log_transaction)
        layout.addWidget(log_button)
        
        layout.addStretch()
        
        panel.setLayout(layout)
        return panel
    
    def _create_menu_bar(self) -> None:
        """Create the application menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        open_action = QAction("Open Image...", self)
        open_action.triggered.connect(self._on_open_image)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Edit menu
        edit_menu = menubar.addMenu("Edit")
        
        settings_action = QAction("Settings...", self)
        settings_action.triggered.connect(self._on_settings)
        edit_menu.addAction(settings_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)
    
    def _create_toolbar(self) -> None:
        """Create the application toolbar."""
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)
        
        open_action = QAction("Open Image", self)
        open_action.triggered.connect(self._on_open_image)
        toolbar.addAction(open_action)
        
        toolbar.addSeparator()
        
        settings_action = QAction("Settings", self)
        settings_action.triggered.connect(self._on_settings)
        toolbar.addAction(settings_action)
    
    def _create_status_bar(self) -> None:
        """Create the application status bar."""
        self.status_label = QLabel("Ready")
        self.statusBar().addWidget(self.status_label, stretch=1)
        
        self.detection_time_label = QLabel("Detection time: -")
        self.statusBar().addPermanentWidget(self.detection_time_label)
    
    def _on_open_image(self) -> None:
        """Handle open image action."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Image Files (*.jpg *.jpeg *.png *.bmp);;All Files (*)",
        )
        
        if file_path:
            self._load_and_detect(Path(file_path))
    
    def _load_and_detect(self, image_path: Path) -> None:
        """
        Load an image and run detection.
        
        Args:
            image_path: Path to the image file.
        """
        if not self.detector:
            QMessageBox.critical(
                self,
                "Error",
                "Object detector not initialized. "
                "Please ensure models/detector.pt exists.",
            )
            return
        
        try:
            # Run detection
            start_time = datetime.now()
            pil_image, detections = self.detector.detect(str(image_path))
            detection_time = (datetime.now() - start_time).total_seconds()
            
            # Store current state
            self.current_image_path = image_path
            self.current_detections = detections
            
            # Draw detections on image
            image_with_boxes = draw_detections(
                pil_image,
                detections,
                ObjectDetector.CLASSES,
            )
            
            # Display image
            self.current_image_pixmap = pil_to_qpixmap(image_with_boxes)
            self._display_image(self.current_image_pixmap)
            
            # Update results
            self._update_results()
            
            # Update status bar
            self.status_label.setText(f"Image: {image_path.name}")
            self.detection_time_label.setText(
                f"Detection time: {detection_time:.2f}s"
            )
            
            logger.info(
                f"Detection complete: {len(detections)} objects found "
                f"in {detection_time:.2f}s"
            )
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to load or detect image:\n{e}",
            )
    
    def _display_image(self, pixmap: QPixmap) -> None:
        """
        Display an image in the graphics view.
        
        Args:
            pixmap: QPixmap to display.
        """
        self.graphics_scene.clear()
        self.graphics_pixmap_item = QGraphicsPixmapItem(pixmap)
        self.graphics_scene.addItem(self.graphics_pixmap_item)
        self.graphics_view.fitInView(
            self.graphics_scene.itemsBoundingRect(),
            Qt.KeepAspectRatio,
        )
    
    def _update_results(self) -> None:
        """Update the results table and totals based on current detections."""
        # Count objects per class
        class_counts: Dict[str, int] = {class_name: 0 for class_name in ObjectDetector.CLASSES}
        for detection in self.current_detections:
            class_name = ObjectDetector.get_class_name(detection.class_id)
            if class_name in class_counts:
                class_counts[class_name] += 1
        
        # Compute totals
        subtotals, total = self.pricing_manager.compute_transaction(class_counts)
        
        # Update table
        self.results_table.setRowCount(len(ObjectDetector.CLASSES))
        for row, class_name in enumerate(ObjectDetector.CLASSES):
            count = class_counts[class_name]
            if count == 0:
                continue  # Skip classes with zero count for cleaner display
            
            unit_price = self.pricing_manager.get_price(class_name)
            subtotal = subtotals.get(class_name, 0.0)
            
            # Add items to table
            self.results_table.setItem(row, 0, QTableWidgetItem(class_name))
            self.results_table.setItem(row, 1, QTableWidgetItem(str(count)))
            self.results_table.setItem(row, 2, QTableWidgetItem(f"${unit_price:.2f}"))
            self.results_table.setItem(row, 3, QTableWidgetItem(f"${subtotal:.2f}"))
        
        # Update summary
        total_objects = sum(class_counts.values())
        self.object_count_label.setText(str(total_objects))
        self.total_price_label.setText(f"${total:.2f}")
        
        logger.info(f"Results updated: {total_objects} objects, total ${total:.2f}")
    
    def _recalculate_totals(self) -> None:
        """Recalculate totals from current detections."""
        if not self.current_detections:
            QMessageBox.information(
                self,
                "Info",
                "No detections to recalculate.",
            )
            return
        
        self._update_results()
        QMessageBox.information(
            self,
            "Success",
            "Totals recalculated.",
        )
    
    def _log_transaction(self) -> None:
        """Log the current transaction to CSV file."""
        if self.current_image_path is None or not self.current_detections:
            QMessageBox.warning(
                self,
                "Warning",
                "No image loaded or no detections to log.",
            )
            return
        
        try:
            # Count objects per class
            class_counts: Dict[str, int] = {class_name: 0 for class_name in ObjectDetector.CLASSES}
            for detection in self.current_detections:
                class_name = ObjectDetector.get_class_name(detection.class_id)
                if class_name in class_counts:
                    class_counts[class_name] += 1
            
            # Compute totals
            _, total_price = self.pricing_manager.compute_transaction(class_counts)
            
            # Prepare log entry
            log_entry = {
                "DateTime": datetime.now().isoformat(),
                "ImagePath": str(self.current_image_path),
                "NumObjects": sum(class_counts.values()),
            }
            
            # Add per-class counts
            for class_name in ObjectDetector.CLASSES:
                log_entry[class_name] = class_counts[class_name]
            
            log_entry["TotalPrice"] = total_price
            
            # Append to CSV
            with open(self.log_file, "a", newline="") as f:
                fieldnames = ["DateTime", "ImagePath", "NumObjects"]
                fieldnames.extend(ObjectDetector.CLASSES)
                fieldnames.append("TotalPrice")
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(log_entry)
            
            QMessageBox.information(
                self,
                "Success",
                f"Transaction logged successfully.\nTotal: ${total_price:.2f}",
            )
            logger.info(f"Transaction logged: {self.current_image_path.name} - ${total_price:.2f}")
            
        except Exception as e:
            logger.error(f"Failed to log transaction: {e}")
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to log transaction:\n{e}",
            )
    
    def _on_settings(self) -> None:
        """Handle settings action."""
        try:
            current_prices = self.pricing_manager.get_all_prices()
            settings_dialog = SettingsDialog(current_prices, self)
            
            if settings_dialog.exec_() == settings_dialog.Accepted:
                updated_prices = settings_dialog.get_prices()
                self.pricing_manager.update_all_prices(updated_prices)
                self.pricing_manager.save_prices()
                
                # Recalculate if we have current detections
                if self.current_detections:
                    self._update_results()
                
                QMessageBox.information(
                    self,
                    "Success",
                    "Prices updated and saved.",
                )
                logger.info("Price settings updated")
            
        except Exception as e:
            logger.error(f"Settings dialog error: {e}")
            QMessageBox.critical(
                self,
                "Error",
                f"Settings error:\n{e}",
            )
    
    def _on_about(self) -> None:
        """Handle about action."""
        QMessageBox.about(
            self,
            "About",
            "Retail Checkout Control v1.0\n\n"
            "Object detection model for retail self-checkout control.\n"
            "Recognizes product classes and computes purchase totals.\n\n"
            "© 2025",
        )
    
    def closeEvent(self, event) -> None:
        """Handle window close event."""
        logger.info("Application closing")
        super().closeEvent(event)
