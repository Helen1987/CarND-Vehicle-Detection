# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[//]: # (Image References)
[final]: ./examples/final2.jpg "Result image"

In this project, the goal is to write a software pipeline to detect vehicles in a video (start with the test_video.mp4 and later implement on full project_video.mp4), but the main output or product we want you to create is a detailed writeup of the project.  Check out the [writeup](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup.md) for this project.  

### The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

![alt text][final]

Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train classifier.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.   You are welcome and encouraged to take advantage of the recently released [Udacity labeled dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) to augment your training data.  

### My project include files and folders:
* [test_images](/test_images) contains images on which my pipeline was tested to create satisfied results
* [src](/src) contains source code of my pipeline
* [research](/research) contains jupyter notebooks which I use to build my pipeline
* [examples](/examples) contains images for writeup.md
* [writeup.md](/writeup.md) short description of chosen approach
* [project_video.mp4](/project_video.mp4), [test_video.mp4](/test_video.mp4) videos used to test my pipeline
* [hog_color.pkl](/hog_color.pkl), [hog.pkl](/hog.pkl), [sparse_color.pkl](/sparse_color.pkl) classifiers for pipeline (I use *hog_color.pkl*)
* [dist_pickle.p](/dist_pickle.p) is used to undistort the image

### Result video

You can find [here](https://www.youtube.com/watch?v=c6c7OA39n-A)
