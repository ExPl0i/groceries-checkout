# Retail Checkout Control - Object Detection GUI

A production-quality PyQt5 desktop application for retail self-checkout control using object detection.

## Features

- **Object Detection**: Load trained PyTorch model to detect retail products on checkout images
- **Class Recognition**: Recognizes 17 product classes (Alcohol, Candy, Canned Food, etc.)
- **Price Management**: Configurable unit prices per product class stored in JSON
- **Real-time Calculation**: Computes total purchase price automatically
- **Transaction Logging**: Records all transactions to CSV file with per-class counts
- **User-Friendly GUI**: PyQt5 interface with image display and detection results panel

## Installation

### Prerequisites
- Python 3.11+
- PyTorch 2.0+
- CUDA support (optional, for GPU acceleration)

### Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create required directories:
```bash
mkdir models logs
```

3. Place your trained detector model at `models/detector.pt`

## Directory Structure

```
retail-checkout-gui/
├── main.py                   # Application entry point
├── model.py                  # Object detector wrapper
├── pricing.py                # Price management
├── ui/
│   ├── __init__.py
│   ├── main_window.py        # Main application window
│   └── settings_dialog.py    # Settings dialog
├── utils/
│   ├── __init__.py
│   └── image_utils.py        # Image conversion utilities
├── config/
│   └── prices.json           # Price configuration
├── models/
│   └── detector.pt           # Trained detection model (user-provided)
├── logs/
│   ├── app.log               # Application log file
│   └── transactions.csv      # Transaction history
└── requirements.txt          # Python dependencies
```

## Usage

### Running the Application

```bash
python main.py
```

### Basic Workflow

1. **Open Image**: Use File menu → "Open Image..." or toolbar button
2. **View Detections**: Image displays with bounding boxes and class labels
3. **Review Results**: Right panel shows detection counts and totals
4. **Adjust Prices**: Edit → Settings to modify unit prices
5. **Log Transaction**: Click "Log Transaction" to save to CSV

### Configuration

#### Editing Prices

1. Click "Settings" in toolbar or Edit menu
2. Modify unit prices in the dialog table
3. Click "OK" to save and apply

Prices are saved to `config/prices.json` for persistence.

#### Model File

Place your trained PyTorch model at `models/detector.pt`. The model should:
- Accept RGB images as input
- Return detections with format: [x1, y1, x2, y2, class_id, confidence, ...]
- Support classes matching the predefined class list

## Model Requirements

The detector model must output detections in the following format:

```python
Detection(
    x1: float,      # Top-left X coordinate
    y1: float,      # Top-left Y coordinate
    x2: float,      # Bottom-right X coordinate
    y2: float,      # Bottom-right Y coordinate
    class_id: int,  # Class index (0-16)
    confidence: float  # Confidence score (0-1)
)
```

## Class Mapping

The application recognizes these 17 product classes in order:

0. Alcohol
1. Candy
2. Canned Food
3. Chocolate
4. Dessert
5. Dried Food
6. Dried Fruit
7. Drink
8. Gum
9. Instant Drink
10. Instant Noodles
11. Milk
12. Personal Hygiene
13. Puffed Food
14. Seasoner
15. Stationery
16. Tissue

## Output Files

### Transaction Log (logs/transactions.csv)

CSV file with columns:
- DateTime: ISO format timestamp
- ImagePath: Full path to processed image
- NumObjects: Total objects detected
- [Class names]: Count per class (17 columns)
- TotalPrice: Final purchase total

Example:
```
DateTime,ImagePath,NumObjects,Alcohol,Candy,...,TotalPrice
2024-01-15T10:30:45.123456,/path/to/image.jpg,5,0,1,...,12.50
```

### Application Log (logs/app.log)

Timestamped log of all application events for debugging.

## API Reference

### ObjectDetector

```python
from model import ObjectDetector

detector = ObjectDetector("models/detector.pt")
image, detections = detector.detect("path/to/image.jpg")

for det in detections:
    print(f"{det.class_id}: ({det.x1}, {det.y1}) - ({det.x2}, {det.y2})")
    print(f"Confidence: {det.confidence:.2%}")
```

### PricingManager

```python
from pricing import PricingManager

manager = PricingManager("config/prices.json")

# Get price for a class
price = manager.get_price("Milk")

# Set price
manager.set_price("Milk", 1.50)

# Compute transaction
class_counts = {"Milk": 2, "Bread": 1}
subtotals, total = manager.compute_transaction(class_counts)

# Save changes
manager.save_prices()
```

## Error Handling

The application handles common errors gracefully:

- **Model file not found**: Falls back to demo mode
- **Image file not found**: Shows error dialog
- **Invalid prices**: Validates before saving
- **CSV write errors**: Logs error and notifies user
- **Model inference failures**: Reports error with details

## Code Quality

- **Type Hints**: Full type annotations throughout
- **Docstrings**: Comprehensive module and function documentation
- **Logging**: Detailed application and error logging
- **Error Handling**: Graceful exception handling with user feedback
- **Comments**: Non-obvious logic explained inline

## Performance Notes

- GPU acceleration is used automatically if CUDA is available
- Image inference time depends on model size and image resolution
- Batch processing is not implemented (single image per transaction)

## Customization

### Adding New Product Classes

1. Update the `CLASSES` list in `model.py`:
```python
CLASSES = [
    "Alcohol", "Candy", ..., "YourNewClass"
]
```

2. Update `config/prices.json` with new class and default price

### Changing Default Prices

Edit `config/prices.json` or use the GUI Settings dialog.

### Modifying Input Size

Adjust the image resize dimensions in `model.py` `_preprocess_image()` method:
```python
transforms.Resize((800, 800))  # Change from default 640x640
```

## Troubleshooting

**Q: Model fails to load**
- Verify `models/detector.pt` exists and is a valid PyTorch model
- Check that PyTorch version matches model requirements

**Q: No objects detected**
- Verify model output format matches expected format
- Check confidence threshold (currently 0.5) in `model.py`
- Review image quality and compatibility with training data

**Q: CSV file not created**
- Ensure `logs/` directory is writable
- Check file permissions

## Future Enhancements

- Batch image processing
- Export transaction reports
- Real-time camera feed integration
- Model performance metrics dashboard
- Multi-model support

## License

Copyright © 2025

## Support

For issues, questions, or suggestions, please refer to the application log files in `logs/` directory.
