from typing import List

from models.detection_model_output import DetectionModelOutput


class TableCropper:
    def __init__(self, padding=10):
        self.padding = padding
        # TODO: Make this a config that is passed to constructor
        self.detection_class_thresholds = {
            "table": 0.5,
            "table rotated": 0.5,
            "no object": 10,
        }

    def crop_detected_tables(self, image, tokens, tables: List[DetectionModelOutput]):
        """
        Process the bounding boxes produced by the table detection model into
        cropped table images and cropped tokens.
        """

        table_crops = []
        for obj in tables:
            if obj.score < self.detection_class_thresholds[obj.label]:
                continue

            cropped_table = {}

            bbox = obj.bbox
            bbox = [
                bbox[0] - self.padding,
                bbox[1] - self.padding,
                bbox[2] + self.padding,
                bbox[3] + self.padding,
            ]

            cropped_img = image.crop(bbox)

            table_tokens = [
                token for token in tokens if iob(token["bbox"], bbox) >= 0.5
            ]
            for token in table_tokens:
                token["bbox"] = [
                    token["bbox"][0] - bbox[0],
                    token["bbox"][1] - bbox[1],
                    token["bbox"][2] - bbox[0],
                    token["bbox"][3] - bbox[1],
                ]

            # If table is predicted to be rotated, rotate cropped image and tokens/words:
            if obj.label == "table rotated":
                cropped_img = cropped_img.rotate(270, expand=True)
                for token in table_tokens:
                    bbox = token["bbox"]
                    bbox = [
                        cropped_img.size[0] - bbox[3] - 1,
                        bbox[0],
                        cropped_img.size[0] - bbox[1] - 1,
                        bbox[2],
                    ]
                    token["bbox"] = bbox

            cropped_table["image"] = cropped_img
            cropped_table["tokens"] = table_tokens

            table_crops.append(cropped_table)

        return table_crops
