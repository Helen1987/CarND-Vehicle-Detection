import pickle
import argparse
from src.Video import Video

from .FeatureExtractor import FeatureExtractor

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video Processing')
    parser.add_argument(
        'source',
        type=str,
        help='Name of the video file to process'
    )
    parser.add_argument(
        'output_folder',
        type=str,
        help='Folder to save the final video'
    )
    parser.add_argument(
        'svc',
        nargs='?',
        type=str,
        default='sparse_color.pkl',
        help='Path to pickle with svc data'
    )

    args = parser.parse_args()

    # load classifier and transformer
    svc_pickle = pickle.load(open(args.svc, "rb"))
    features = svc_pickle["features"]  # color, space, hog
    color_space = svc_pickle["color_space"]

    feature_extractor = FeatureExtractor(features, color_space)
    svc = svc_pickle["svc"]
    X_scaler = svc_pickle["X_scaler"]

    if features[0]:  # color features
        hist_bins = svc_pickle["hist_bins"]
        hist_range = svc_pickle["hist_range"]
        feature_extractor.init_color_parameters(hist_bins, hist_range)
    if features[1]:  # space features
        spatial_size = svc_pickle["spatial_size"]
        feature_extractor.init_space_parameters(spatial_size)
    if features[2]:
        orient = svc_pickle["orient"]
        pix_per_cell = svc_pickle["pix_per_cell"]
        cell_per_block = svc_pickle["cell_per_block"]
        hog_channel = svc_pickle["hog_channel"]
        feature_extractor.init_hog_parameters(orient, pix_per_cell, cell_per_block, hog_channel)

    distortion_pickle = pickle.load(open("dist_pickle.p", "rb"))
    mtx = distortion_pickle["mtx"]
    dist = distortion_pickle["dist"]

    video = Video(args.source, args.output_folder, X_scaler, svc, feature_extractor)
    video.process(mtx, dist)
