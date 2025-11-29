"""
Image utilities for conversion and drawing operations.

Handles conversion between PIL, NumPy, and PyQt5 image formats,
as well as drawing detection boxes and labels.
"""

from typing import Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PyQt5.QtGui import QImage, QPixmap


def pil_to_qpixmap(pil_image: Image.Image) -> QPixmap:
    """
    Convert PIL Image to PyQt5 QPixmap.
    
    Args:
        pil_image: PIL Image object (RGB or RGBA).
    
    Returns:
        QPixmap suitable for display in PyQt5 widgets.
    """
    # Convert PIL to NumPy array
    numpy_array = np.array(pil_image)
    
    # Handle different image formats
    if len(numpy_array.shape) == 2:  # Grayscale
        height, width = numpy_array.shape
        bytes_per_line = width
        q_image = QImage(
            numpy_array.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_Grayscale8,
        )
    elif numpy_array.shape[2] == 3:  # RGB
        height, width, channel = numpy_array.shape
        bytes_per_line = 3 * width
        q_image = QImage(
            numpy_array.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_RGB888,
        )
    elif numpy_array.shape[2] == 4:  # RGBA
        height, width, channel = numpy_array.shape
        bytes_per_line = 4 * width
        q_image = QImage(
            numpy_array.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_RGBA8888,
        )
    else:
        raise ValueError(f"Unsupported image format: {numpy_array.shape}")
    
    return QPixmap.fromImage(q_image)


def draw_detections(
    image: Image.Image,
    detections: list,
    class_names: list,
    box_color: Tuple[int, int, int] = (0, 255, 0),
    text_color: Tuple[int, int, int] = (255, 255, 255),
    box_width: int = 2,
    font_size: int = 12,
) -> Image.Image:
    """
    Draw bounding boxes and class labels on image.
    
    Args:
        image: PIL Image object.
        detections: List of Detection objects with x1, y1, x2, y2, class_id, confidence.
        class_names: List of class name strings.
        box_color: RGB color for bounding box (default green).
        text_color: RGB color for text labels (default white).
        box_width: Width of bounding box lines in pixels.
        font_size: Font size for labels.
    
    Returns:
        PIL Image with drawn detections.
    """
    # Create a copy to avoid modifying the original
    output_image = image.copy()
    draw = ImageDraw.Draw(output_image)
    
    # Try to load a nice font; fall back to default if unavailable
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()
    
    for detection in detections:
        x1, y1, x2, y2 = detection.x1, detection.y1, detection.x2, detection.y2
        class_id = detection.class_id
        confidence = detection.confidence
        
        # Get class name
        class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
        
        # Draw bounding box
        draw.rectangle(
            [(x1, y1), (x2, y2)],
            outline=box_color,
            width=box_width,
        )
        
        # Prepare label text
        label = f"{class_name} ({confidence*100:.1f}%)"
        
        # Get text bounding box for background
        bbox = draw.textbbox((x1, y1 - font_size - 4), label, font=font)
        
        # Draw semi-transparent background for text
        # Note: PIL doesn't support alpha transparency directly, so we draw filled rectangle
        draw.rectangle(
            bbox,
            fill=box_color,
        )
        
        # Draw text
        draw.text(
            (x1, y1 - font_size - 2),
            label,
            fill=text_color,
            font=font,
        )
    
    return output_image


def numpy_to_qpixmap(numpy_array: np.ndarray) -> QPixmap:
    """
    Convert NumPy array to QPixmap.
    
    Args:
        numpy_array: NumPy array with shape (height, width, 3) or (height, width, 4).
    
    Returns:
        QPixmap suitable for display in PyQt5 widgets.
    """
    # Ensure array is contiguous
    numpy_array = np.ascontiguousarray(numpy_array)
    
    height, width = numpy_array.shape[:2]
    
    if len(numpy_array.shape) == 2:  # Grayscale
        bytes_per_line = width
        q_image = QImage(
            numpy_array.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_Grayscale8,
        )
    elif numpy_array.shape[2] == 3:  # RGB
        bytes_per_line = 3 * width
        q_image = QImage(
            numpy_array.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_RGB888,
        )
    elif numpy_array.shape[2] == 4:  # RGBA
        bytes_per_line = 4 * width
        q_image = QImage(
            numpy_array.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_RGBA8888,
        )
    else:
        raise ValueError(f"Unsupported array shape: {numpy_array.shape}")
    
    return QPixmap.fromImage(q_image)
