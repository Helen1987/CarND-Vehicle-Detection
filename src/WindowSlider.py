import numpy as np
import cv2
from .FeatureExtractor import FeatureExtractor


class WindowSlider:
    def __init__(self, transformer, classifier, extractor):
        self.svc = classifier
        self.X_scaler = transformer
        self.extractor = extractor
        self.bounding_boxes = []
        self.window = (64, 64)
        self.pix_per_cell = 8
        self.cells_per_window = (self.window//self.pix_per_cell)-1

    # find all windows in region with specified overlap
    # assume that region is scaled already
    def slide_window(self, region_shape, xy_overlap=(0.5, 0.5)):
        window_list = []

        # Instead of overlap, define how many cells to step
        cells_per_step = np.int(self.cells_per_window*(1-xy_overlap))

        # Define blocks and steps as above
        n_x_cells = (region_shape[1] // self.pix_per_cell) - 1
        n_y_cells = (region_shape[0] // self.pix_per_cell) - 1

        # Compute the number of windows in x/y
        n_x_steps = (n_x_cells - self.cells_per_window[0]) // cells_per_step[0]
        n_y_steps = (n_y_cells - self.cells_per_window[1]) // cells_per_step[1]

        # Loop through finding x and y window positions
        # Note: you could vectorize this step, but in practice
        # you'll be considering windows one by one with your
        # classifier, so looping makes sense
        for y in range(n_y_steps):
            for x in range(n_x_steps):
                # Calculate window position (for HOG)
                x_pos = x*cells_per_step
                y_pos = y*cells_per_step
                # Calculate window scaled coordinates
                x_start = x_pos*self.pix_per_cell
                y_start = y_pos*self.pix_per_cell

                window_list.append(((x_pos, y_pos), (x_start, y_start)))

        return window_list

    # function find windows on particular area which presumably contains cars
    def find_hot_windows(self, scale, img, x_start_stop=[None, None], y_start_stop=[None, None],
                         xy_overlap=(0.5, 0.5)):
        # If x and/or y start/stop positions not defined, set to image size
        if x_start_stop[0] is None:
            x_start_stop[0] = 0
        if x_start_stop[1] is None:
            x_start_stop[1] = img.shape[1]
        if y_start_stop[0] is None:
            y_start_stop[0] = 0
        if y_start_stop[1] is None:
            y_start_stop[1] = img.shape[0]

        img_to_search = img[y_start_stop[0]:y_start_stop[1], x_start_stop[0]:x_start_stop[1], :]
        converted_image = FeatureExtractor.convert_color(img_to_search, conv='RGB2YCrCb')
        if scale != 1:
            scaled_image = cv2.resize(converted_image, np.int(img_to_search.shape/scale))
        else:
            scaled_image = converted_image

        # precalculate HOG features
        self.extractor.extract_hog_features_from_img(scaled_image, feature_vec=False)

        windows = self.slide_window(scaled_image.shape, xy_overlap=xy_overlap)
        for (x_pos, y_pos), (x_start, y_start) in windows:
            window_img = cv2.resize(scaled_image[x_start:x_start+self.window[0], y_start:y_start+self.window[1]], self.window)
            features = self.extractor.extract_features(window_img, y_pos, x_pos, self.cells_per_window)

            scaled_features = self.X_scaler.transform(features)
            test_prediction = self.svc.predict(scaled_features)

            if test_prediction == 1:
                x_top_left = np.int(x_start*scale)
                y_top_left = np.int(y_start*scale)
                win_draw = np.int(self.window*scale)
                self.bounding_boxes.append([
                    x_top_left, y_top_left,
                    x_top_left+win_draw[0]+x_start_stop[0],
                    y_top_left+win_draw[1]+y_start_stop[0]
                ])

    def find_cars(self, img):
        scale_and_region = []
        for scale, region in scale_and_region:
            self.find_hot_windows(scale, img, region[0], region[1])

        return self.bounding_boxes
