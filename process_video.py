import pickle
import argparse
from src.Video import Video

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
    args = parser.parse_args()

    dist_pickle = pickle.load(open("dist_pickle.p", "rb"))
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]

    video = Video(args.source, args.output_folder)
    video.process(mtx, dist)
