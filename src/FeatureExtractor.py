import matplotlib.image as mpimg
import numpy as np
import cv2
from skimage.feature import hog


class FeatureExtractor:
    def __init__(self):
        pass

    @staticmethod
    def bin_spatial(img, size=(32, 32)):
        features = cv2.resize(img, size).ravel() 
        return features

    # Define a function to compute color histogram features  
    @staticmethod
    def color_hist(img, nbins=32, bins_range=(0, 256)):
        channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
        channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
        channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
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

    @staticmethod
    def extract_features(imgs, cspace='RGB', spatial_size=(32, 32), hist_bins=32, hist_range=(0, 256)):
        # Create a list to append feature vectors to
        features = []
        # Iterate through the list of images
        for file in imgs:
            # Read in each one by one
            image = mpimg.imread(file)
            # apply color conversion if other than 'RGB'
            if cspace != 'RGB':
                if cspace == 'HSV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                elif cspace == 'LUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
                elif cspace == 'HLS':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
                elif cspace == 'YUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            else: feature_image = np.copy(image)      
            # Apply bin_spatial() to get spatial color features
            spatial_features = FeatureExtractor.bin_spatial(feature_image, size=spatial_size)
            # Apply color_hist() also with a color space option now
            hist_features = FeatureExtractor.color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
            # Append the new feature vector to the features list
            features.append(np.concatenate((spatial_features, hist_features)))
        return features
