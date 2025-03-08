import os
from pdf2image import convert_from_path
from pathlib import Path
import shutil

# Path to your PDF file
input_folder = Path("/home/clara/Documents/order-scans/wetransfer_onedrive_2025-03-01-zip_2025-03-01_0957/Verkoopbonnen Bomedys/")
output_folder = Path("images")  # Folder to save the images

for root, dirs, files in os.walk(input_folder):
    for file in files:
        if file.endswith('.pdf'):
            # Get the file name without the extension
            file_name = file.split('.')[0]
            # Convert PDF to images
            pages = convert_from_path(Path(root) / file, 300)  # 300 dpi (can adjust depending on quality)

            # Save each page as an image
            for i, page in enumerate(pages):
                image_path = f"{output_folder}/{file_name}_page_{i + 1}.png"  # Adjust file extension as needed
                page.save(image_path, 'PNG')  # Save the page as an image

        if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
            shutil.copyfile(input_folder / file, output_folder / file)

print(f"PDFs have been converted to images and saved in '{output_folder}' folder.")
