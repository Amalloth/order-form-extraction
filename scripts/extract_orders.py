from pathlib import Path

import torch
from transformers import (
    AutoModelForObjectDetection,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    AutoTokenizer,
)
from transformers import TableTransformerForObjectDetection

from general_ocr_parser import GeneralOCRParser
from image_reader import ImageReader
from plotting import visualize_detected_tables
from table_cropper import TableCropper
from table_detector import TableDetector
from table_parser import TableParser

IMAGE_DIR = Path("images")

# Device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load models
table_model = AutoModelForObjectDetection.from_pretrained(
    "microsoft/table-transformer-detection", revision="no_timm"
)
table_model.to(device)

structure_model = TableTransformerForObjectDetection.from_pretrained(
    "microsoft/table-structure-recognition-v1.1-all"
)
structure_model.to(device)

ocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-handwritten")
# tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased") # Probably need another tokenizer?
# ocr_processor.tokenizer = tokenizer
ocr_model = VisionEncoderDecoderModel.from_pretrained(
    "microsoft/trocr-large-handwritten"
)

# TODO: Move this to constructors?
# update id2label to include "no object"
id2label = table_model.config.id2label
id2label[len(table_model.config.id2label)] = "no object"

structure_id2label = structure_model.config.id2label
structure_id2label[len(structure_id2label)] = "no object"

# Create table detector, cropper and parser
table_detector = TableDetector(table_model, device, id2label)
table_cropper = TableCropper()
table_parser = TableParser(
    structure_model, device, structure_id2label, ocr_processor, ocr_model
)
general_ocr_parser = GeneralOCRParser(ocr_processor, ocr_model)

# Read images
image_reader = ImageReader(IMAGE_DIR)
images = image_reader.read_images()

for image in images:
    # Get detected tables
    detected_tables = table_detector.get_tables(image=image)

    # Save image of detected tables
    # TODO: Save image to out folder
    # fig = visualize_detected_tables(image, detected_tables)

    # Crop image to to the size of tables
    cropped_tables = table_cropper.crop_detected_tables(
        image=image, tokens=[], tables=detected_tables
    )

    # Pick largest / most likely table
    # TODO: Find better solution than taking first result
    order_table = cropped_tables[0]["image"].convert("RGB")

    cropped_tables[0]["image"].save("table.jpg")

    general_ocr_parser.parse(cropped_tables[0]["image"])

    # Parse table
    orders = table_parser.parse(order_table)

    print(orders)

    # TODO: Save cell visualisation to image

    # Write table to CSV
