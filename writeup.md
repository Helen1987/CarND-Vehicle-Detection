**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[hog]: ./examples/HOG_example.jpg "Example of HOG features"
[reg1]: ./examples/region1.jpg "Scale 1.0"
[reg2]: ./examples/region2.jpg "Scale 1.2"
[reg3]: ./examples/region3.jpg "Scale 1.3"
[reg4]: ./examples/region4.jpg "Scale 1.5"
[reg5]: ./examples/region5.jpg "Scale 2.0"
[threshold1]: ./examples/threshold1.jpg
[threshold2]: ./examples/threshold2.jpg
[threshold3]: ./examples/threshold3.jpg
[result1]: ./examples/th_result1.jpg
[result2]: ./examples/th_result2.jpg
[result3]: ./examples/th_result3.jpg
[final1]: ./examples/final1.jpg
[final2]: ./examples/final2.jpg
[final3]: ./examples/final3.JPG

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for creating classifier you can find in [jupyter notebook](/research/Classifier.ipynb). The code which extract features from the images you can find in [`FeatureExtractor` class](src/FeatureExtractor.py).

I started by reading in all the `vehicle` and `non-vehicle` images. As a starting point I decided to use HOG features for SVM classifier.

![alt text][hog]

####2. Explain how you settled on your final choice of HOG parameters.

I tried different colorspaces and end up using HSV over "ALL" channels with `orient = 9`, `pix_per_cell = 8`, `cell_per_block = 2`. I was able to reach accuracy 0.987. 

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

It was possible to use other features for classifier. I decided that spatial features reflect similar concept as HOG features, as result not much help in using it. At the same time histogram features inspect totally different aspect of car\non-car images, so I decided to add histogram features with `histbin = 16` to my classifier.

After extracting features I applied `StandardScaler` to normalize features:

```
X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)
```

To train classifier I divided features into train and test set via `train_test_split`.

Eventually, I was able to reach 0.9904 accuracy.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

To identify regions of search for my classifier I examined test images and tried to find an appropriate scale (check [jupyter notebook](/research/Region%20and%20Scale.ipynb)).I ended up with next regions:

![alt text][reg1]
![alt text][reg2]
![alt text][reg3]
![alt text][reg4]
![alt text][reg5]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on 5 scales using HSV 3-channel HOG features plus histograms of color in the feature vector, which provided a nice result. Here are some example images:

![final1]
![final2]
![final3]

Performance is too slow. I [precalculated HOG features](/src/FeatureExtractor.py#L139) to improve performance. But I have too many regions for search, so resulted pipeine is slow.

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](https://www.youtube.com/watch?v=c6c7OA39n-A)

####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I extracted the positions of positive detections in each frame of the video. You can find code in [`WindowSlider` class](/src/WindowSlider.py). From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions (check code in [`VehicleDetector` class](src/VehicleDetector.py)). In my case I applied threshold of 6 detections.

To improve results I keep history of thresholded heatmaps over last [5 frames](/src/VehicleDetector.py#L41). Then I summed up heatmaps and applied threshold once again to get more stable result. 

Afterwards, I used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected and [draw the result](/src/Frame.py#L37)

### Here I provided heatmaps and indentified bounding boxes:

![threshold1]
![result1]
![threshold2]
![result2]
![threshold2]
![result2]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The largets issue with my pipeline is performance. For 1 minutes video I have 20 minutes of processing. It is not allowed for for self-driving car which must identify vehicles in real-time.

I still observe false positives. I believe I must play more to find better regions for cars search.
