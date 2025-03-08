from pathlib import Path

import torch
from doctr.models import ocr_predictor

from doctr_flow import process_image
from readers.image_reader import ImageReader

# Get the root directory of this repository
ROOT_DIR = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT_DIR / Path("images")
OUTPUT_FOLDER = Path("/home/clara/CodingProjects/order-form-extraction/output/test_2")

# Device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load models
model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)

# Read images
image_reader = ImageReader(IMAGE_DIR)
image_paths = image_reader.list_image_paths()

for image_path in image_paths:
    result = process_image(image_path, OUTPUT_FOLDER, model, None, None)

    text_objects = result["pages"][0]["blocks"][0]
    lines = []
    for line in text_objects["lines"]:
        lines.append(" ".join([word["value"] for word in line["words"]]))

    # Save raw recognized text to file
    with open(OUTPUT_FOLDER / f"{image_path.parts[-1]}_recognized_text.txt", "w") as f:
        for text in lines:
            f.write(f"{text}\n")


