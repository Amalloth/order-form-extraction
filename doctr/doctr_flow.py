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
from doctr.io import DocumentFile
from doctr.utils.visualization import visualize_page

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
    """ Detect text regions using DocTR OCR. Returns detected text and image with bounding boxes."""
    doctr_image = DocumentFile.from_images(image_path)
    results = reader(doctr_image)

    return results

def recognize_text(cropped_regions, model, processor):
    """ Recognize text with Paddle OCR """
    texts = []
    for region in cropped_regions:
        region = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(region)
        pixel_values = processor(pil_image, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
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

def visualize_intermediate_results(image_path, detected_text, ax_image_synthesized):
    image = cv2.imread(image_path)
    image_with_boxes = visualize_page(detected_text.pages[0].export(), image);
    ax_image_synthesized.imshow(detected_text.synthesize(font_family='DejaVuSans.ttf')[0]);
    ax_image_synthesized.axis("off")
    return image_with_boxes

def process_image(image_path: Path, output_folder: Path, reader, model, processor):
    result = detect_text(image_path, reader)
    # recognized_texts = recognize_text(cropped_regions, model, processor)

    # Save intermediate result
    fig_image_synthesized, ax_image_synthesized = plt.subplots()
    image_with_boxes = visualize_intermediate_results(image_path, result, ax_image_synthesized)
    
    image_with_boxes.savefig(output_folder / f"{image_path.parts[-1]}_detected_boxes.png")
    fig_image_synthesized.savefig(output_folder / f"{image_path.parts[-1]}_synthesized_image.png", dpi=900)

    return result.export()
    
