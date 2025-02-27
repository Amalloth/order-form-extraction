import torch

from image_preprocessing import detection_transform
from models.detected_table import DetectedTable


class TableDetector:
    def __init__(self, model, device):
        self.model = model

    def get_tables(self, image, id2label):
        pixel_values = detection_transform(image).unsqueeze(0)
        pixel_values = pixel_values.to(self.device)

        with torch.no_grad():
            outputs = self.model(pixel_values)

        objects = self.outputs_to_objects(outputs, image.size, id2label)
        return objects

    # for output bounding box post-processing
    @staticmethod
    def _box_cxcywh_to_xyxy(self, x):
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    @staticmethod
    def _rescale_bboxes(self, out_bbox, size):
        img_w, img_h = size
        b = self._box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b

    def outputs_to_objects(self, outputs, img_size, id2label):
        m = outputs.logits.softmax(-1).max(-1)
        pred_labels = list(m.indices.detach().cpu().numpy())[0]
        pred_scores = list(m.values.detach().cpu().numpy())[0]
        pred_bboxes = outputs['pred_boxes'].detach().cpu()[0]
        pred_bboxes = [elem.tolist() for elem in self._rescale_bboxes(pred_bboxes, img_size)]

        objects = []
        for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
            class_label = id2label[int(label)]
            if not class_label == 'no object':
                objects.append(DetectedTable(
                    label=class_label,
                    score=float(score),
                    bbox=[float(elem) for elem in bbox]
                ))

        return objects