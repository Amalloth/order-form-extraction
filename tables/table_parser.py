from typing import List

import numpy as np
import pandas as pd
import torch
from PIL.Image import Image
from tqdm import tqdm
from tqdm.auto import tqdm

from image_preprocessing import structure_transform
from models.detection_model_output import DetectionModelOutput
from post_processing import outputs_to_objects


class TableParser:
    def __init__(
        self, table_structure_model, device, id2label, ocr_processor, ocr_model
    ):
        self.model = table_structure_model
        self.device = device
        self.id2label = id2label
        self.ocr_processor = ocr_processor
        self.ocr_model = ocr_model

    def parse(self, cropped_table: Image):
        pixel_values = structure_transform(cropped_table).unsqueeze(0)
        pixel_values = pixel_values.to(self.device)

        # forward pass
        with torch.no_grad():
            outputs = self.model(pixel_values)

        cells = outputs_to_objects(outputs, cropped_table.size, self.id2label)
        cell_coordinates = self._get_cell_coordinates_by_row(cells)
        raw_parsed_table = self._apply_ocr(cropped_table, cell_coordinates)
        result = pd.DataFrame(raw_parsed_table)
        return result

    def _get_cell_coordinates_by_row(self, table_data: List[DetectionModelOutput]):
        # Extract rows and columns
        rows = [entry for entry in table_data if entry.label == "table row"]
        columns = [entry for entry in table_data if entry.label == "table column"]

        # Sort rows and columns by their Y and X coordinates, respectively
        rows.sort(key=lambda x: x.bbox[1])
        columns.sort(key=lambda x: x.bbox[0])

        # Function to find cell coordinates
        def find_cell_coordinates(row, column):
            cell_bbox = [column.bbox[0], row.bbox[1], column.bbox[2], row.bbox[3]]
            return cell_bbox

        # Generate cell coordinates and count cells in each row
        cell_coordinates = []

        for row in rows:
            row_cells = []
            for column in columns:
                cell_bbox = find_cell_coordinates(row, column)
                row_cells.append({"column": column.bbox, "cell": cell_bbox})

            # Sort cells in the row by X coordinate
            row_cells.sort(key=lambda x: x["column"][0])

            # Append row information to cell_coordinates
            # TODO: Make model for cell coordinates
            cell_coordinates.append(
                {"row": row.bbox, "cells": row_cells, "cell_count": len(row_cells)}
            )

        # Sort rows from top to bottom
        cell_coordinates.sort(key=lambda x: x["row"][1])

        return cell_coordinates

    def _apply_ocr(self, table: Image, cell_coordinates):
        # let's OCR row by row
        data = dict()
        max_num_columns = 0
        for idx, row in enumerate(tqdm(cell_coordinates)):
            row_text = []
            for cell in row["cells"]:
                # crop cell out of image
                cell_image = table.crop(cell["cell"])
                # apply OCR
                pixel_values = self.ocr_processor(
                    cell_image, return_tensors="pt"
                ).pixel_values
                generated_ids = self.ocr_model.generate(pixel_values)
                generated_text = self.ocr_processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0]

                if len(generated_text) > 0:
                    row_text.append(generated_text)

            if len(row_text) > max_num_columns:
                max_num_columns = len(row_text)

            data[idx] = row_text

        print("Max number of columns:", max_num_columns)

        # pad rows which don't have max_num_columns elements
        # to make sure all rows have the same number of columns
        for row, row_data in data.copy().items():
            if len(row_data) != max_num_columns:
                row_data = row_data + [
                    "" for _ in range(max_num_columns - len(row_data))
                ]
            data[row] = row_data

        return data
