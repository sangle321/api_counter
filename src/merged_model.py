import numpy as np

from src.detector.detector import Detector
from src.config import pipes_detection


class CompletedModel(object):
    def __init__(self):
        self.pipes_detection_model = Detector(path_config=pipes_detection['path_to_config'],
                                              path_ckpt=pipes_detection['path_ckpt'],
                                               path_to_labels=pipes_detection['path_to_labels'],
                                               nms_threshold=pipes_detection['nms_ths'],
                                               score_threshold=pipes_detection['score_ths'])

    def detect_pipes(self, image):
        img, img_origin, list_location = self.pipes_detection_model.predict(image)
        return list_location


    def response_data(self, image):
        coordinate_dict = self.detect_pipes(image)
        print(coordinate_dict)
        data = {}
        length_prod = len(coordinate_dict)
        if length_prod > 0:
            data['count'] = length_prod
            data["results"] = coordinate_dict
        return data
            