"""
Object detection model wrapper.

Handles loading and inference with PyTorch-based detection models.
Supports models that output bounding boxes, class indices, and confidence scores.
"""

import logging
from pathlib import Path
from typing import List, Tuple, NamedTuple

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


logger = logging.getLogger(__name__)


class Detection(NamedTuple):
    """Represents a single object detection."""
    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int
    confidence: float


class ObjectDetector:
    """
    Wrapper for PyTorch-based object detection models.

    Uses OpenCV for robust image loading and supports common model output
    formats (torch.Tensor arrays, list of detections, and torchvision-style
    outputs: list[dict] with 'boxes', 'labels', 'scores').
    """

    # Class mapping (must match model training)
    CLASSES = [
        "Alcohol", "Candy", "Canned Food", "Chocolate", "Dessert",
        "Dried Food", "Dried Fruit", "Drink", "Gum", "Instant Drink",
        "Instant Noodles", "Milk", "Personal Hygiene", "Puffed Food",
        "Seasoner", "Stationery", "Tissue"
    ]

    def __init__(self, model_path: str | Path, device: str = "auto", conf_threshold: float = 0.5):
        """
        Initialize the detector with a model file.

        Args:
            model_path: Path to the PyTorch model file (.pt).
            device: Device to run inference on ("cuda", "cpu", or "auto").
                If "auto", uses CUDA if available.
            conf_threshold: Confidence threshold for keeping detections.
        """
        self.model_path = Path(model_path)
        self.conf_threshold = float(conf_threshold)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading model from {self.model_path}")

        try:
            # Load the model object or scripted module
            self.model = torch.load(self.model_path, map_location=self.device)

            # If a state_dict is accidentally provided, raise helpful error
            if isinstance(self.model, dict):
                logger.error("Model file appears to be a state dict, not a model object")
                raise RuntimeError(
                    "State dict detected. Please provide a full model object or a scripted/traced model.")

            self.model = self.model.to(self.device)
            self.model.eval()
            logger.info(f"Model loaded successfully on device: {self.device}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e

    def detect(self, image_path: str | Path) -> Tuple[Image.Image, List[Detection]]:
        """
        Run detection on an image.

        Args:
            image_path: Path to the input image file.

        Returns:
            Tuple of:
                - Original PIL Image
                - List of Detection objects
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        logger.info(f"Running detection on {image_path.name}")

        try:
            # Use OpenCV for robust loading (handles many encodings)
            bgr = cv2.imread(str(image_path))
            if bgr is None:
                raise FileNotFoundError(f"Could not read image via OpenCV: {image_path}")

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)

            image_tensor = self._preprocess_image(pil_image)

            # Run inference
            with torch.no_grad():
                output = self.model(image_tensor)

            # Parse output and return detections
            detections = self._parse_output(output, pil_image.size)
            logger.info(f"Found {len(detections)} objects")

            return pil_image, detections

        except Exception as e:
            logger.error(f"Detection failed: {e}")
            raise RuntimeError(f"Detection failed: {e}") from e

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess image for model inference.

        Args:
            image: PIL Image object.

        Returns:
            Preprocessed image tensor on the configured device.
        """
        transform = transforms.Compose([
            transforms.Resize((640, 640)),  # Keep square resize for many detectors
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        tensor = transform(image)
        return tensor.unsqueeze(0).to(self.device)

    def _parse_output(self, output, image_size: Tuple[int, int]) -> List[Detection]:
        """
        Parse model output into Detection objects.

        Supports several common output formats:
        - torchvision detection models: list[dict] with keys 'boxes','labels','scores'
        - single torch.Tensor of shape [1, N, >=6] with columns [x1,y1,x2,y2,class,conf]
        - numpy array or list of detections
        """
        detections: List[Detection] = []

        # Helper to append a detection with checks
        def add_det(x1, y1, x2, y2, cls, conf):
            if conf < self.conf_threshold:
                return
            cls = int(cls)
            if cls < 0 or cls >= len(self.CLASSES):
                return
            # Clip to image bounds
            width, height = image_size
            x1c = max(0.0, min(float(x1), float(width)))
            y1c = max(0.0, min(float(y1), float(height)))
            x2c = max(0.0, min(float(x2), float(width)))
            y2c = max(0.0, min(float(y2), float(height)))
            detections.append(Detection(x1c, y1c, x2c, y2c, cls, float(conf)))

        # Case 1: torchvision-style list[dict]
        if isinstance(output, (list, tuple)) and len(output) > 0 and isinstance(output[0], dict):
            res = output[0]
            boxes = res.get('boxes')
            labels = res.get('labels')
            scores = res.get('scores')
            if boxes is not None and labels is not None and scores is not None:
                boxes = boxes.cpu().numpy() if isinstance(boxes, torch.Tensor) else np.array(boxes)
                labels = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else np.array(labels)
                scores = scores.cpu().numpy() if isinstance(scores, torch.Tensor) else np.array(scores)
                for (box, lab, sc) in zip(boxes, labels, scores):
                    x1, y1, x2, y2 = box[:4]
                    add_det(x1, y1, x2, y2, int(lab), float(sc))
                return detections

        # Case 2: single torch tensor or numpy array
        if isinstance(output, torch.Tensor):
            arr = output.cpu().numpy()
        elif isinstance(output, (list, tuple)) and len(output) > 0:
            # maybe a list of tensors or arrays - try to pick the first meaningful
            first = output[0]
            if isinstance(first, torch.Tensor):
                arr = first.cpu().numpy()
            else:
                arr = np.array(first)
        else:
            try:
                arr = np.array(output)
            except Exception:
                arr = np.array([])

        if arr.size == 0:
            return detections

        # Ensure shape is (N, M)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]

        if arr.ndim == 2 and arr.shape[1] >= 6:
            # assumed format [x1, y1, x2, y2, class, conf, ...]
            for row in arr:
                x1, y1, x2, y2, cls, conf = row[:6]
                add_det(x1, y1, x2, y2, cls, conf)
            return detections

        # Fallback: if arr is (N,4) and separate lists for classes/conf exist, we already handled above
        logger.debug("Output format not recognized; returning empty detections list")
        return detections

    @staticmethod
    def get_class_name(class_id: int) -> str:
        """
        Get class name from class ID.

        Args:
            class_id: Class index.

        Returns:
            Class name string.
        """
        if 0 <= class_id < len(ObjectDetector.CLASSES):
            return ObjectDetector.CLASSES[class_id]
        return f"Unknown ({class_id})"
