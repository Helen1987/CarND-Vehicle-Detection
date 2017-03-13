import numpy as np
import matplotlib.image as mpimg
import cv2
from skimage.feature import hog


class FeatureExtractor:
    def __init__(self, use_color, use_space, use_hog, color_space):
        self.use_HOG = use_hog
        self.use_color = use_color
        self.use_space = use_space
        self.color_space = color_space
        # hog parameters
        self.orient = 7
        self.pix_per_cell = 8
        self.cell_per_block = 2
        self.hog_channel = 0
        # space features
        self.spatial_size = 32
        # color features
        self.hist_bins = 32
        self.hist_range = (0, 256)
        # to keep precalculated hog features
        self.hog_features = []

    def init_hog_parameters(self, orient, pix_per_cell, cell_per_block, hog_channel):
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.hog_channel = hog_channel

    def init_space_parameters(self, spatial_size):
        self.spatial_size = spatial_size

    def init_color_parameters(self, hist_bins, hist_range):
        self.hist_bins = hist_bins
        self.hist_range = hist_range

    def extract_features(self, img, ypos, xpos, n_blocks_per_window):
        spatial_features = np.array([])
        hist_features = np.array([])
        hog_features = np.array([])

        if self.use_space:
            spatial_features = FeatureExtractor.bin_spatial(img, (self.spatial_size, self.spatial_size))
        if self.use_color:
            hist_features = FeatureExtractor.color_hist(img, self.hist_bins, self.hist_range)
        if self.use_HOG:
            hog_features = self.get_precalculated_hog_features(
                ypos, xpos, n_blocks_per_window)

        return np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)

    @staticmethod
    def convert_color(image, color_space):
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(image)
        return feature_image

    def convert(self, image):
        return FeatureExtractor.convert_color(image, self.color_space)

    @staticmethod
    def bin_spatial(img, size=(32, 32)):
        features = cv2.resize(img, size).ravel() 
        return features

    # Define a function to compute color histogram features  
    @staticmethod
    def color_hist(img, n_bins=32, bins_range=(0, 256)):
        channel1_hist = np.histogram(img[:, :, 0], bins=n_bins, range=bins_range)
        channel2_hist = np.histogram(img[:, :, 1], bins=n_bins, range=bins_range)
        channel3_hist = np.histogram(img[:, :, 2], bins=n_bins, range=bins_range)
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
        return hist_features

    @staticmethod
    def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
        # Call with two outputs if vis==True
        if vis:
            features, hog_image = hog(
                img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                visualise=vis, feature_vector=feature_vec)
            return features, hog_image
        # Otherwise call with one output
        else:
            features = hog(
                img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                visualise=vis, feature_vector=feature_vec)
            return features

    def get_precalculated_hog_features(self, y_pos, x_pos, n_cells_per_window):
        shift = n_cells_per_window-(self.cell_per_block-1)
        if self.hog_channel == 'ALL':
            hog_feat1 = self.hog_features[0, y_pos:y_pos+shift, x_pos:x_pos+shift].ravel()
            hog_feat2 = self.hog_features[1, y_pos:y_pos+shift, x_pos:x_pos+shift].ravel()
            hog_feat3 = self.hog_features[2, y_pos:y_pos+shift, x_pos:x_pos+shift].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
        else:
            hog_features = self.hog_features[y_pos:y_pos+shift, x_pos:x_pos+shift].ravel()
        return hog_features

    def extract_hog_features_from_img(self, img):
        if self.hog_channel == 'ALL':
            self.hog_features = [None, None, None]
            self.hog_features[0] = FeatureExtractor.get_hog_features(
                img[:, :, 0], self.orient, self.pix_per_cell,
                self.cell_per_block, vis=False, feature_vec=False)
            self.hog_features[1] = FeatureExtractor.get_hog_features(
                img[:, :, 1], self.orient, self.pix_per_cell,
                self.cell_per_block, vis=False, feature_vec=False)
            self.hog_features[2] = FeatureExtractor.get_hog_features(
                img[:, :, 2], self.orient, self.pix_per_cell,
                self.cell_per_block, vis=False, feature_vec=False)
            #if feature_vec:
            #    self.hog_features = np.ravel(np.append((hog_feat1, hog_feat2, hog_feat3)))
            #else:
            #self.hog_features = np.vstack((hog_feat1, hog_feat2, hog_feat3))
        else:
            self.hog_features = FeatureExtractor.get_hog_features(
                img[:, :, self.hog_channel], self.orient,
                self.pix_per_cell, self.cell_per_block, vis=False, feature_vec=False)

    @staticmethod
    def extract_hog_features(images, cspace, orient, pix_per_cell, cell_per_block, hog_channel):
        features = []
        for file in images:
            image = cv2.imread(file)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            feature_image = FeatureExtractor.convert_color(rgb_image, cspace)
            hog_features = FeatureExtractor.get_hog_features(
                feature_image[:, :, hog_channel], orient, pix_per_cell, cell_per_block)
            features.append(hog_features)

        return features

    @staticmethod
    def extract_features_from_images(imgs, cspace='RGB', spatial_size=(32, 32), hist_bins=32, hist_range=(0, 256)):
        features = []
        for file in imgs:
            image = cv2.imread(file)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            feature_image = FeatureExtractor.convert_color(rgb_image, cspace)
            # Apply bin_spatial() to get spatial color features
            spatial_features = FeatureExtractor.bin_spatial(feature_image, size=spatial_size)
            # Apply color_hist() also with a color space option now
            hist_features = FeatureExtractor.color_hist(feature_image, n_bins=hist_bins, bins_range=hist_range)

            features.append(np.concatenate((spatial_features, hist_features)))
        return features
