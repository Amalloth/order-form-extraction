from PIL.Image import Image


class GeneralOCRParser:
    def __init__(self, ocr_processor, ocr_model):
        self.ocr_processor = ocr_processor
        self.ocr_model = ocr_model

    def parse(self, image: Image):
        pass
