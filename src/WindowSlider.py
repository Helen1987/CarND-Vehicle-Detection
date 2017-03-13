import numpy as np
import cv2


class WindowSlider:
    def __init__(self, transformer, classifier, extractor):
        self.svc = classifier
        self.X_scaler = transformer
        self.extractor = extractor
        self.bounding_boxes = []
        self.window = 64
        self.pix_per_cell = 8
        self.cells_per_window = (self.window//self.pix_per_cell)

    # find all windows in region with specified overlap
    # assume that region is scaled already
    def slide_window(self, region_shape, xy_overlap):
        window_list = []

        # Instead of overlap, define how many cells to step
        x_cells_per_step = np.int(self.cells_per_window*(1-xy_overlap[0]))
        y_cells_per_step = np.int(self.cells_per_window*(1-xy_overlap[1]))
        # print('per step', x_cells_per_step, y_cells_per_step)

        # Define blocks and steps as above
        # print('region', region_shape[1], region_shape[0])
        n_x_cells = (region_shape[1]//self.pix_per_cell)
        n_y_cells = (region_shape[0]//self.pix_per_cell)
        # print('cells', n_x_cells, n_y_cells)

        # Compute the number of windows in x/y
        n_x_steps = (n_x_cells-self.cells_per_window)//x_cells_per_step+1
        n_y_steps = (n_y_cells-self.cells_per_window)//y_cells_per_step+1
        # print('steps', n_x_steps, n_y_steps)

        # Loop through finding x and y window positions
        # Note: you could vectorize this step, but in practice
        # you'll be considering windows one by one with your
        # classifier, so looping makes sense
        for y in range(n_y_steps):
            for x in range(n_x_steps):
                # Calculate window position (for HOG)
                x_pos = x*x_cells_per_step
                y_pos = y*y_cells_per_step
                # Calculate window scaled coordinates
                x_start = x_pos*self.pix_per_cell
                y_start = y_pos*self.pix_per_cell

                window_list.append(((x_pos, y_pos), (x_start, y_start)))

        return window_list

    @staticmethod
    def get_scaled_image_region(converted_image, scale, x_start_stop, y_start_stop):
        img_to_search = converted_image[y_start_stop[0]:y_start_stop[1], x_start_stop[0]:x_start_stop[1], :]
        if scale != 1:
            im_shape = img_to_search.shape
            scaled_image = cv2.resize(img_to_search, (np.int(im_shape[1]/scale), np.int(im_shape[0]/scale)))
        else:
            scaled_image = img_to_search
        return scaled_image

    # function find windows on particular area which presumably contains cars
    def find_hot_windows(self, scale, converted_image, x_start_stop=[None, None], y_start_stop=[None, None],
                         xy_overlap=(0.5, 0.5)):
        # If x and/or y start/stop positions not defined, set to image size
        if x_start_stop[0] is None:
            x_start_stop[0] = 0
        if x_start_stop[1] is None:
            x_start_stop[1] = converted_image.shape[1]
        if y_start_stop[0] is None:
            y_start_stop[0] = 0
        if y_start_stop[1] is None:
            y_start_stop[1] = converted_image.shape[0]

        scaled_image = self.get_scaled_image_region(converted_image, scale, x_start_stop, y_start_stop)

        # pre-calculate HOG features
        self.extractor.prepare_hog_features(scaled_image)

        windows = self.slide_window(scaled_image.shape, xy_overlap=xy_overlap)
        win_draw = np.int(self.window*scale)

        for (x_pos, y_pos), (x_start, y_start) in windows:
            if (x_start+self.window > scaled_image.shape[1]) or (y_start+self.window > scaled_image.shape[0]):
                continue

            window_img = cv2.resize(scaled_image[y_start:y_start+self.window, x_start:x_start+self.window],
                                    (self.window, self.window))
            features = self.extractor.extract_features(window_img, y_pos, x_pos, self.cells_per_window)

            scaled_features = self.X_scaler.transform(features)
            test_prediction = self.svc.predict(scaled_features)

            if test_prediction == 1:
                x_top_left = np.int(x_start*scale)+x_start_stop[0]
                y_top_left = np.int(y_start*scale)+y_start_stop[0]

                self.bounding_boxes.append([
                    (x_top_left, y_top_left),
                    (x_top_left+win_draw, y_top_left+win_draw)
                ])

    def find_cars(self, img):
        self.bounding_boxes = []
        scale_and_region = [
            [1.0, (300, img.shape[1]-300), (380, 450)],
            [1.2, (0, img.shape[1]), (380, 530)],
            [1.3, (0, img.shape[1]), (400, 580)],
            [1.5, (0, img.shape[1]), (400, 600)],
            [2.0, (0, img.shape[1]), (380, 650)]
            ]
        converted_image = self.extractor.convert(img)
        for scale, x_region, y_region in scale_and_region:
            self.find_hot_windows(scale, converted_image, x_region, y_region, xy_overlap=(0.75, 0.75))

        return self.bounding_boxes
