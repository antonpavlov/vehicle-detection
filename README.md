# vehicle-detection
Vehicle Detection and Tracking - Udacity Self-Driving Car Engineer Nanodegree. 

The goal of this project is to build a software pipeline for an automatic recognition of vehicles from a video stream. The main piece of the script is a classifier that uses a[Support Vector Machine](https://en.wikipedia.org/wiki/Support_vector_machine) - a machine learning algorithm for classification based on supervised learning. This project strongly relies and contains references to Udacity's self-driving car Nanodegree program.


### Contents of the repo ###
`<camera_cal>` - A folder with calibration images <br />
`<dataset>` - A folder with images used for classifier training. It is a placeholder. <br />
`<output_images>` - A folder with intermediate results of processing. <br /> 
`<test_images>` - A folder with test images; Results of processing will be saved there. <br />
`<tests>` - A folder used in a couple of script tests. <br /> 
`.gitignore` - .gitignore for Python. <br />
`LICENSE` - The GNU GENERAL PUBLIC LICENSE, Version 2. <br />
`README.md` - this file. <br />
`vehicle-detector.py` - The script for vehicle detection.

### Environment set-up ###

Before run a script from this repo, please setup an environment with all required dependencies. Once for all it may be done by using [Anaconda](https://www.anaconda.com/download/) and the following repository: [https://github.com/udacity/CarND-Term1-Starter-Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)
    
Download also a training data from Udacity's project repository: [here (vehicles)](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip), and [here (non-vehicles)](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) Please place them into `dataset` folder.

To run the script in a debug mode, please use the -O option.


### Reflection ###
The following approach was suggested during the course:
1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
2. Apply a distortion correction to every raw image to be processed.
3. Train a classifier on provided dataset.
4. In each video frame or a still image, search for a objects using a sliding window approach. Three sizes different sizes are used in the script 64 by 64, 100 by 100, and 200 by 200, overlap each sliding window by a half of its size.
5. Feed a classifier with all windowed images and get a number of cars  

### Technical restrictions and weaknesses ###
- Calibration coefficients are related to a specific camera used to record images. 
- Despite a certain robustness, the proposed pipeline depends on classifier's ability to generalize. Failures on that matter may lead to wrong results during the tracking.


### Examples of processing ###
Camera calibration - Original image
![Original](https://github.com/antonpavlov/adv-lanelines/blob/master/support_files/calibration1_processed.png)


Camera calibration - Undistorted image
![Undistorted](https://github.com/antonpavlov/adv-lanelines/blob/master/support_files/calibration1_undistorted.png)

<br />

Let's build the following pipeline:

1. Open an image

![Processed](https://github.com/antonpavlov/adv-lanelines/blob/master/support_files/test3_A_processed.png)

<br />


2. Apply image correction

![Undistort](https://github.com/antonpavlov/adv-lanelines/blob/master/support_files/test3_B_undistort.png)

<br />


3. Application of Sobel operator on undistorted image 

![Gradient](https://github.com/antonpavlov/adv-lanelines/blob/master/support_files/test3_C_abs_sobel_thresh.png)

<br />


4. Filter an image by gradient magnitude in both (x and y) directions 

![Magnitude](https://github.com/antonpavlov/adv-lanelines/blob/master/support_files/test3_D_mag_thresh.png)

<br />


5. Filter an image considering gradient orientation 

![Orientation](https://github.com/antonpavlov/adv-lanelines/blob/master/support_files/test3_E_dir_binary.png)

<br />


6. HLS color space threshold 

![HLS](https://github.com/antonpavlov/adv-lanelines/blob/master/support_files/test3_F_hls_select.png)

<br />


7. All thresholds applied together to undistorted image

![All_together](https://github.com/antonpavlov/adv-lanelines/blob/master/support_files/test3_G_combined_thresh.png)

<br />


8. Perspective transform; warp-in image

![Perspective](https://github.com/antonpavlov/adv-lanelines/blob/master/support_files/test3_H_perspective.png)

<br />


9. Find lanes in a binary warped image

![Lanes](https://github.com/antonpavlov/adv-lanelines/blob/master/support_files/test3_I_lanes.png)

<br />


10. Make curvature calculations; vehicle position and draw results over an original image

![Lanes](https://github.com/antonpavlov/adv-lanelines/blob/master/support_files/test3_K_final.png)

<br />


### Future work ###
As a future work, it might be possible to get better classification results using Convolutional Neural Networks. It is also very interesting to implement `vehicle-detector` script in an embedded system and deploy it in a real environment. Of course, several improvements and parameter tunning can be done in terms of code. 


### License ###

Python script `vehicle-detector.py` is distributed under the terms described in the GPLv2 license. 
Please refer to [Udacity](https://github.com/udacity) regarding all other supporting materials.