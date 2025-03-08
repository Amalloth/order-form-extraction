import cv2
import easyocr
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from langdetect import detect
import symspellpy
from symspellpy import SymSpell
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# Language-specific SymSpell instances
sym_spell_dicts = {
    "en": "frequency_dictionary_en_82_765.txt",
    # "de": "frequency_dictionary_de_100_000.txt",
}

def init_symspell(lang):
    sym_spell = SymSpell()
    if lang in sym_spell_dicts:
        sym_spell.load_dictionary(sym_spell_dicts[lang], 0, 1)
    return sym_spell

def detect_text(image_path, reader):
    """ Detect text regions using EasyOCR """

    image = cv2.imread(image_path)
    results = reader.detect(image, min_size=2, text_threshold=0, mag_ratio=1.5)
    boxes = results[0][0]
    cropped_regions = []

    for box in boxes:
        x_min, x_max, y_min, y_max = box
        cropped = image[y_min:y_max, x_min:x_max]
        cropped_regions.append(cropped)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    colored_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(colored_image)
    plt.title("Detected Text Regions")
    plt.axis("off")

    # Write image with detected boxes to jpg file
    cv2.imwrite(f"{Path(image_path).parts[-1]}_detected_text_regions.jpg", colored_image)

    plt.show()
    
    return cropped_regions

def recognize_text(cropped_regions, model, processor):
    """ Recognize text with TrOCR """
    texts = []
    for region in cropped_regions:
        region = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(region)
        pixel_values = processor(pil_image, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        texts.append(text)
    return texts

def correct_text(text):
    """ Detect language and apply SymSpell corrections """
    try:
        language = detect(text)
        if language in sym_spell_dicts:
            sym_spell = init_symspell(language)
            suggestions = sym_spell.lookup(text, symspellpy.Verbosity.CLOSEST, max_edit_distance=2)
            return suggestions[0].term if suggestions else text
        else:
            return text
    except:
        return text

def process_image(image_path, output_folder: Path, reader, model, processor):
    cropped_regions = detect_text(image_path, reader)
    recognized_texts = recognize_text(cropped_regions, model, processor)

    # Save raw recognized text to file
    with open(output_folder / f"{Path(image_path).parts[-1]}_recognized_text.txt", "w") as f:
        for text in recognized_texts:
            f.write(f"{text}\n")