import cv2
import numpy as np

BLUR_KERNEL = 3


class Frame:
    def __init__(self, img):
        self.img = cv2.undistort(img, Frame.mtx, Frame.dist, None, Frame.mtx)
        self.bird_view_img = None

    @staticmethod
    def init(width, height, mtx, dist):
        top_offset = 100
        bottom_offset = 10
        top_line_offset = 75
        bottom_line_offset = 115

        s_points = np.float32([
            (bottom_line_offset, height - bottom_offset),
            (width / 2 - top_line_offset, height / 2 + top_offset),
            (width / 2 + top_line_offset, height / 2 + top_offset),
            (width - bottom_line_offset, height - bottom_offset)])

        offset = 100
        new_height = height * 10
        d_points = np.float32([
            [offset, new_height], [offset, 0],
            [width - offset, 0], [width - offset, new_height]])

        Frame.matrix = cv2.getPerspectiveTransform(s_points, d_points)
        Frame.inverse_matrix = cv2.getPerspectiveTransform(d_points, s_points)
        Frame.mtx = mtx
        Frame.dist = dist
        Frame.bv_size = (width, new_height)

    def draw_labeled_bboxes(self, labels):
        # Iterate through all detected cars
        for car_number in range(1, labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(self.img, bbox[0], bbox[1], (0, 0, 255), 6)

        return self.img
