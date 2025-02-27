from pathlib import Path

import torch
from transformers import AutoModelForObjectDetection

from image_reader import ImageReader
from plotting import visualize_detected_tables
from table_detector import TableDetector

IMAGE_DIR = Path("data")

# Device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-detection", revision="no_timm")

# update id2label to include "no object"
id2label = model.config.id2label
id2label[len(model.config.id2label)] = "no object"

# Create detector
table_detector = TableDetector(model)

# Read images
image_reader = ImageReader(IMAGE_DIR)
images = image_reader.read_images()

for image in images:
    # Get detected tables
    objects = table_detector.get_tables(image, device)

    # Save image of detected tables
    fig = visualize_detected_tables(image, objects)

    # Crop image to biggest detected table

    # Parse table

    # Write table to CSV


