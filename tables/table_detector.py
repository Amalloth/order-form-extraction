from typing import List

import torch

from image_preprocessing import detection_transform
from models.detection_model_output import DetectionModelOutput
from post_processing import outputs_to_objects


class TableDetector:
    def __init__(self, model, device, id2label):
        self.model = model
        self.device = device
        self.id2label = id2label

    def get_tables(self, image) -> List[DetectionModelOutput]:
        pixel_values = detection_transform(image).unsqueeze(0)
        pixel_values = pixel_values.to(self.device)

        with torch.no_grad():
            outputs = self.model(pixel_values)

        objects = outputs_to_objects(outputs, image.size, self.id2label)
        return objects