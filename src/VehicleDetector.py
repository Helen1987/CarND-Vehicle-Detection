import numpy as np
from collections import deque
from scipy.ndimage.measurements import label


class VehicleDetector:
    def __init__(self, slider, history_count):
        self.current_heatmap = np.array([])
        self.slider = slider
        # keep track of latest heatmaps
        self.heatmap_history = deque([])
        # the value of a heatmap
        self.average_heatmap = np.array([])
        self.n = history_count

    def init_heatmap(self, width, height):
        self.average_heatmap = np.zeros((height, width)).astype(np.float)

    def add_heat(self, bbox_list):
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            self.current_heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    @staticmethod
    def apply_threshold(heatmap, threshold):
        # Zero out pixels below the threshold
        heatmap[heatmap <= threshold] = 0

    def run_detector(self, image):
        box_list = self.slider.find_cars(image)

        self.current_heatmap = np.zeros_like(image[:, :, 0]).astype(np.float)
        self.add_heat(box_list)

        # remove noise
        self.apply_threshold(self.current_heatmap, 1)

        # add the latest heatmap for averaging and in history
        self.average_heatmap += self.current_heatmap
        self.heatmap_history.append(self.current_heatmap)

        if len(self.heatmap_history) >= self.n:
            # remove too old heatmap from history and averaging
            old_heatmap = self.heatmap_history.popleft()
            self.average_heatmap -= old_heatmap

        # it's important to leave original average_heatmap unchanged
        # so, it will be possible to track ALL latest predictions
        heatmap_copy = np.copy(self.average_heatmap)
        self.apply_threshold(heatmap_copy, len(self.heatmap_history))
        return label(heatmap_copy)

        # Visualize the heatmap when displaying
        #heatmap = np.clip(heat, 0, 255)
