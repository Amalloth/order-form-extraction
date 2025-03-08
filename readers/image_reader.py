import os
from pathlib import Path
from typing import List

from PIL import Image

class ImageReader:
    """Reads images from a directory and returns them as a list of numpy arrays."""
    def __init__(self, directory: Path):
        self.directory = directory

    def read_images(self) -> List[Image]:
        images = []
        for filename in os.listdir(self.directory):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image = Image.open(os.path.join(self.directory, filename)).convert('RGB')
                images.append(image)
        return images

    def list_image_paths(self) -> List[Path]:
        return list(self.directory.glob('*.jpg')) + list(self.directory.glob('*.png'))