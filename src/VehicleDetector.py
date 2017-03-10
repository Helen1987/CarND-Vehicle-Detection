import numpy as np
from scipy.ndimage.measurements import label

from .WindowSlider import WindowSlider


class VehicleDetector:
    def __init__(self, slider):
        self.heatmap = np.array([])
        self.slider = slider

    def add_heat(self, bbox_list):
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            self.heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    def apply_threshold(self, threshold):
        # Zero out pixels below the threshold
        self.heatmap[self.heatmap <= threshold] = 0

    def run_detector(self, image):
        box_list = self.slider.find_cars(image, 400, 656, 1.5, )

        self.heatmap = np.array([])
        self.add_heat(box_list)
        self.apply_threshold(1)

        # Visualize the heatmap when displaying
        #heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        return label(self.heatmap)