import os
import sys
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

from .Frame import Frame
from .VehicleDetector import VehicleDetector
from .WindowSlider import WindowSlider


class Video:
    def __init__(self, path, output_folder, transformer, svm_classifier, feature_extractor):
        # calibrate the camera
        self.path = path
        self.output_folder = os.path.join(os.getcwd(), output_folder)
        self.vehicle_detector = VehicleDetector(
            WindowSlider(transformer, svm_classifier, feature_extractor))

    def handle_frame(self, image):
        try:
            current_frame = Frame(image)
            labels = self.vehicle_detector.run_detector(current_frame.img)
            result = current_frame.draw_labeled_bboxes(labels)
        except:
            print("Unexpected error:", sys.exc_info()[0])
            project_dir = os.getcwd()
            output_folder_path = os.path.join(project_dir, 'output_images')
            output_image_path = os.path.join(output_folder_path, 'error_.jpg')
            mpimg.imsave(output_image_path, image)
            raise

        return result

    def process(self, mtx, dist):
        project_video = VideoFileClip(self.path)
        Frame.init(project_video.size[0], project_video.size[1], mtx, dist)

        new_video = project_video.fl_image(self.handle_frame)
        output_file_name = os.path.join(self.output_folder, "result_" + self.path)
        new_video.write_videofile(output_file_name, audio=False)

    def save_frame(self):
        project_video = VideoFileClip(self.path)
        project_video.save_frame('bug.jpg', (0, 16))
