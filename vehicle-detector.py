# -*- coding: utf-8 -*-

# *** IMPORTS ***
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from sklearn import svm
from sklearn.utils import shuffle
from scipy.ndimage.measurements import label
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import time

from moviepy.editor import VideoFileClip


# *** FUNCTIONS ***
# Ready for release
def calibration():
    """
    Camera calibration function for compensation of radial and tangential distortions.
    :return: Success flag and correction coefficients
    Ref: Course notes and https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_calib3d/py_calibration/py_calibration.html
    """
    # ./camera_cal/calibration1.jpg nx=9 ny=5
    # ./camera_cal/calibration2.jpg nx=9 ny=6
    # ./camera_cal/calibration3.jpg nx=9 ny=6
    # ./camera_cal/calibration4.jpg nx=5 ny=6
    # ./camera_cal/calibration5.jpg nx=7 ny=6
    # ./camera_cal/calibration6.jpg nx=9 ny=6
    # ...
    # ./camera_cal/calibration20.jpg nx=9 ny=6

    # Setup object points
    nx = 9  # Internal corners on horizontal
    ny = 6  # Internal corners on vertical
    object_point = np.zeros((ny * nx, 3), np.float32)
    object_point[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # Arrays to store object points and image points
    object_points = []  # Real world space
    image_points = []   # Image plane

    # Image List
    images = glob.glob('./camera_cal/calibration*.jpg')

    for idx, fname in enumerate(images):

        # Skip following files; nx and ny should be 9 and 6 only
        if fname == './camera_cal/calibration1.jpg':
            pass
        elif fname == './camera_cal/calibration4.jpg':
            pass
        elif fname == './camera_cal/calibration5.jpg':
            pass
        else:
            # Open an input image
            image = cv2.imread(fname)

            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

            cal_mtx = []
            cal_dist = []
            # If found, draw corners
            if ret is True:
                success_flag = True
                image_points.append(corners)
                object_points.append(object_point)

                # Calibrate camera
                cal_ret, cal_mtx, cal_dist, cal_rvecs, cal_tvecs = cv2.calibrateCamera(object_points,
                                                                               image_points,
                                                                               gray.shape[::-1], None, None)
    # End of FOR loop
    return cal_mtx, cal_dist

# Ready for release
def draw_boxes(input_image, bboxes, color=(0, 0, 255), thick=6):
    """
    Draw boxes in an input image
    :param img: An input image
    :param bboxes: Tuples with coordinates of boxes
    :param color: Line color in RGB
    :param thick: Line thickness
    :return: An image with boxes
    Reference: Udacity's project specification
    """
    # Make a copy of the image
    draw_img = np.copy(input_image)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(draw_img, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return draw_img


def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    """
    Define a function that takes an image,
    # start and stop positions in both x and y,
    # window size (x and y dimensions),
    # and overlap fraction (for both x and y)
    :param img:
    :param x_start_stop:
    :param y_start_stop:
    :param xy_window:
    :param xy_overlap:
    :return:
    """
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] is None:
        x_start_stop[0] = 0
    if x_start_stop[1] is None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] is None:
        y_start_stop[0] = 0
    if y_start_stop[1] is None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0] * (xy_overlap[0]))
    ny_buffer = np.int(xy_window[1] * (xy_overlap[1]))
    nx_windows = np.int((xspan - nx_buffer) / nx_pix_per_step)
    ny_windows = np.int((yspan - ny_buffer) / ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


# Ready for test - not commented
def color_hist(img, nbins=32, bins_range=(0, 256)):
    """
    Get color features
    :param img:
    :param nbins:
    :param bins_range:
    :return:
    """
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


# Ready for test - not commented
def bin_spatial(img, size=(32, 32)):
    """
    Return the feature vector
    :param img:
    :param size:
    :return:
    """
    features = cv2.resize(img, size).ravel()
    return features


# Ready for test - not commented
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    """
    Get HOG features
    :param img:
    :param orient:
    :param pix_per_cell:
    :param cell_per_block:
    :param vis:
    :param feature_vec:
    :return:
    """
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features


# Ready for test - not commented
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32), hist_bins=32, orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0, spatial_feat=True, hist_feat=True, hog_feat=True):
    """
    Combine and normalize
    :param imgs:
    :param color_space:
    :param spatial_size:
    :param hist_bins:
    :param orient:
    :param pix_per_cell:
    :param cell_per_block:
    :param hog_channel:
    :param spatial_feat:
    :param hist_feat:
    :param hog_feat:
    :return:
    """
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # Apply color conversion
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

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                         orient, pix_per_cell, cell_per_block,
                                                         vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                                pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the list of features
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features


# Ready for test - not commented
def single_img_features(img, color_space='RGB', spatial_size=(32, 32), hist_bins=32, orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0, spatial_feat=True, hist_feat=True, hog_feat=True):
    # Empty list
    img_features = []
    # Apply color conversion
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)
    # Compute spatial features
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # Append features to list
        img_features.append(spatial_features)
    # Compute histogram features
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        # Append features to list
        img_features.append(hist_features)
    # Compute HOG features
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel],
                                    orient, pix_per_cell, cell_per_block,
                                    vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # Append features to list
        img_features.append(hog_features)

    # Return concatenated array of features
    return np.concatenate(img_features)


# Ready for test - not commented
def search_windows(img, windows, clf, scaler, color_space='RGB', spatial_size=(32, 32), hist_bins=32, hist_range=(0, 256), orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0):
    # Create an empty list for the found windows
    on_windows = []
    # Iterate over all windows in the windows list
    for window in windows:
        # Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        # Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # Predict using your classifier
        prediction = clf.predict(test_features)
        #If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    # Return windows for positive detections
    return on_windows


# Ready for test - not commented - Not necessary
def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)


# Ready for test - not commented
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    """
    find cars function as shown in the lession
    function uses a scale-factor to search with different window sizes
    function as well replaces the overlapping with cells_per_steps
    :param img:
    :param ystart:
    :param ystop:
    :param scale:
    :param svc:
    :param X_scaler:
    :param orient:
    :param pix_per_cell:
    :param cell_per_block:
    :param spatial_size:
    :param hist_bins:
    :return:
    """
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    boxes = []
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch,(np.int(imshape[1]/scale),(np.int(imshape[0]/scale))))
        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]
        #Blocks and steps
        nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
        nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
        nfeat_per_block = orient * cell_per_block**2
        window = 64
        nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
        # Replacing overlapping with cells_per_step
        cells_per_step = 2
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step
        #get hog features
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False)

        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb * cells_per_step
                xpos = xb * cells_per_step
                # Extract hog features
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                x_left = xpos * pix_per_cell
                y_top = ypos * pix_per_cell

                # Extract img patch
                subimg = cv2.resize(ctrans_tosearch[y_top:y_top+window, x_left:x_left+window],(64,64))
                spatial_features = bin_spatial(subimg, size=spatial_size)
                hist_features = color_hist(subimg, nbins=hist_bins)

                #test_features2 = np.concatenate((spatial_features, hist_features, hog_features))
                #test_features = X_scaler.transform(np.array(test_features2)).reshape(1, -1)
                test_features = X_scaler.transform(np.hstack((spatial_features, hist_features,
                                                             hog_features)).reshape(1, -1))
                test_prediction = svc.predict(test_features)

                if test_prediction == 1:
                    xbox_left = np.int(x_left * scale)
                    ytop_draw = np.int(y_top * scale)
                    win_draw = np.int(window*scale)
                    cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw, ytop_draw+
                                                                         win_draw+ystart),(0,0,255),6)
                    boxes.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw, ytop_draw+
                                                                         win_draw+ystart)))
        return draw_img, boxes


# Ready for test - not commented
def add_heat(heatmap, bbox_list):
    for box in bbox_list:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return heatmap


# Ready for test - not commented
def apply_threshold(heatmap, threshold):
    heatmap[heatmap <= threshold] = 0
    return heatmap


# Ready for test - not commented
def draw_labeled_boxes(img, labels):
    for car_number in range(1, labels[1]+1):
        nonzero = (labels[0] == car_number).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        #print(bbox[0], bbox[1])
        #cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
        img = draw_boxes(img, [bbox], color=(0, 0, 255), thick=6)
    return img


# Ready for test - not commented
def vehicle_detection(image, ystart, ystop, scale, svc, X_Scaler, orient,pix_per_cell, cell_per_block, spatial_size,
                      hist_bins, threshold):
    #find cars in image
    out_img, boxes = find_cars(image, ystart, ystop, scale, svc, X_Scaler, orient,
                               pix_per_cell, cell_per_block, spatial_size, hist_bins)
    heat = np.zeros_like(image[:, :, 0]).astype(np.float)
    box_list = boxes
    heat = add_heat(heat, box_list)
    heat= apply_threshold(heat, threshold)
    heatmap = np.clip(heat, 0, 255)
    labels = label(heatmap)
    draw_img = draw_labeled_boxes(np.copy(image), labels)
    return draw_img

# Ready for test - not commented
def process_image(image):
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    processed_img = vehicle_detection(image, ystart, ystop, scale, svc, X_rescaled, orient,pix_per_cell, cell_per_block, spatial_size,
                      hist_bins, threshold)
    return processed_img


# *** GLOBAL VARIABLES ***
global mtx, dist


# *** MAIN FUNCTION ***
if __name__ == "__main__":
    # For debug, please run this script with -O option

    # Unit test of draw_boxes function
    if __debug__:
        fname = './tests/bbox-example-image.jpg'
        image = mpimg.imread(fname)
        bboxes = [((275, 572), (380, 510)), ((488, 563), (549, 518)), ((554, 543), (582, 522)), ((601, 555), (646, 522)), ((657, 545), (685, 517)), ((849, 678), (1135, 512))]
        result = draw_boxes(image, bboxes)
        ax1 = plt.clf()
        ax1 = plt.imshow(result)
        ax1 = plt.savefig(fname[:-4] + '_A_boxes.png')

    # Unit test on Support Vector Machine from sklearn
    if __debug__:
        X = [[0, 0], [1, 1]]
        y = [0, 1]
        clf = svm.SVC()
        clf.fit(X, y)
        # Make a prediction
        pred = clf.predict([[2., 2.]])
        assert pred == [1]

    # Calibrate camera once
    mtx, dist = calibration()

    # Unit test on calibration
    if __debug__:
        fname = './tests/calibration-test.jpg'
        image = mpimg.imread(fname)
        undist_image = cv2.undistort(image, mtx, dist, None, mtx)
        ax2 = plt.clf()
        ax2 = plt.imshow(undist_image)
        ax2 = plt.savefig(fname[:-4] + '_B_undistort.png')

    # Unit test on sliding window function
    if __debug__:
        fname = './tests/bbox-example-image.jpg'
        image = mpimg.imread(fname)
        windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[None, None], xy_window=(128, 128), xy_overlap=(0.5, 0.5))
        window_img = draw_boxes(image, windows, color=(0, 0, 255), thick=6)
        ax3 = plt.clf()
        ax3 = plt.imshow(window_img)
        ax3 = plt.savefig(fname[:-4] + '_C_slidingWindows.png')

    # MAIN PARAMETERS
    color_space = 'YCrCb'  # RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientation
    pix_per_cell = 8  # Pix per cell
    cell_per_block = 2  # Cells per block
    hog_channel = "ALL"  # Hog channels 1, 2, 3 or ALL
    spatial_size = (32, 32)  # Spatial binning
    hist_bins = 32  # Histogram bins
    spatial_feat = True
    hist_feat = True
    hog_feat = True
    y_start_stop = [350, None]
    hist_range = (0, 256)

    # Load training dataset
    vehicles = glob.glob('./dataset/vehicles/*/*.png')
    cars = []
    noncars = []

    # Populate cars
    for image in vehicles:
        cars.append(image)

    # Populate non-cars
    non_vehicles = glob.glob('./dataset/non-vehicles/*/*.png')

    for image in non_vehicles:
        noncars.append(image)

    if __debug__:
        print("Images were loaded successfully!")
        car_ind = np.random.randint(0, len(cars))
        notcar_ind = np.random.randint(0, len(noncars))

        # Read in car / not-car images
        car_image = mpimg.imread(cars[car_ind])
        notcar_image = mpimg.imread(noncars[notcar_ind])
        # Plot the examples
        ax1 = plt.figure()
        ax1 = plt.clf()
        ax1 = plt.subplot(121)
        ax1 = plt.imshow(car_image)
        ax1 = plt.title('Example Car Image')
        ax1 = plt.subplot(122)
        ax1 = plt.imshow(notcar_image)
        ax1 = plt.title('Example Not-car Image')
        ax1 = plt.savefig('datasetExamples.png')

    # Get the features of cars and noncars
    car_features = extract_features(cars, color_space, spatial_size, hist_bins, orient, pix_per_cell,
                                    cell_per_block, hog_channel)

    noncar_features = extract_features(noncars, color_space, spatial_size, hist_bins, orient, pix_per_cell,
                                       cell_per_block, hog_channel)

    if __debug__:
        print("Features were extracted successfully!")
        print("Length of car features: ", len(car_features))
        print("Length of non-car features: ", len(noncar_features))
        print("Shape of car features: ", np.shape(car_features))
        print("Shape of non-car features: ", np.shape(noncar_features))

    # Build-up a dataset
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(noncar_features))))
    if __debug__:
        print("Create a label vector with the following shape: ", y.shape)

    X = np.vstack((car_features, noncar_features)).astype(np.float64)
    if __debug__:
        print("Create a vertical stack of features X with the following shape: ", X.shape)

    # Normalize training data
    X_rescaled = StandardScaler().fit(X)
    X_transformed = X_rescaled.transform(X)

    # Shuffle and split the data
    X_transformed, y = shuffle(X_transformed, y)
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2)
    if __debug__:
        print("Training data was shuffled and split successfully!")

    # Train support vector machine
    svc = LinearSVC()

    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    if __debug__:
        print(round(t2 - t, 2), 'Seconds to train SVC...')
        print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
        t = time.time()
        n_predict = 10
        print('The SVC predicts: ', svc.predict(X_test[0:n_predict]))
        print('For these', n_predict, 'labels: ', y_test[0:n_predict])
        t2 = time.time()
        print(round(t2 - t, 5), 'Seconds to predict', n_predict, 'labels with SVC')

    # Read test images from folder
    test_images = glob.glob('./test_images/*.jpg')

    for idx, f_name in enumerate(test_images):
        image = mpimg.imread(f_name)
        draw_image = np.copy(image)

        # Scale the image since its a .jpg
        image = image.astype(np.float32) / 255

        # Search with three different window sizes
        windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                                xy_window=(100, 100), xy_overlap=(0.5, 0.5))
        windows2 = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                                xy_window=(200, 200), xy_overlap=(0.3, 0.3))
        windows3 = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                                xy_window=(64, 64), xy_overlap=(0.3, 0.3))

        # Get the found windows that match the features as list
        hot_windows = search_windows(image, windows, svc, X_rescaled, color_space=color_space,
                                    spatial_size=spatial_size, hist_bins=hist_bins,
                                    orient=orient, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel)
        hot_windows2 = search_windows(image, windows2, svc, X_rescaled, color_space=color_space,
                                    spatial_size=spatial_size, hist_bins=hist_bins,
                                    orient=orient, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel)
        hot_windows3 = search_windows(image, windows3, svc, X_rescaled, color_space=color_space,
                                    spatial_size=spatial_size, hist_bins=hist_bins,
                                    orient=orient, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel)

        # Draw the found windows that match the features in boxes
        window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)
        window_img2 = draw_boxes(window_img, hot_windows2, color=(0, 0, 255), thick=6)
        window_img3 = draw_boxes(window_img2, hot_windows3, color=(0, 0, 255), thick=6)

        # Plot the examples
        ax1 = plt.figure()
        ax1 = plt.clf()
        ax1 = plt.imshow(window_img3)
        ax1 = plt.savefig('test_images.png')

        ystart = 400
        ystop = 656
        scale = 1.5

        out_img, boxes = find_cars(image, ystart, ystop, scale, svc, X_rescaled, orient, pix_per_cell, cell_per_block,
                                   spatial_size, hist_bins)
        if __debug__:
            # Plot the examples
            ax2 = plt.figure()
            ax2 = plt.clf()
            ax2 = plt.imshow(out_img)
            ax2 = plt.savefig('test_image_boxes.png')

            print("Boxes found: ", boxes)

        threshold = 1
        heat = np.zeros_like(image[:, :, 0]).astype(np.float)
        heat = add_heat(heat, boxes)
        heat = apply_threshold(heat, threshold)
        heatmap = np.clip(heat, 0, 255)

        labels = label(heatmap)
        if __debug__:
            print("Vehicles found: ", labels[1])
            plt.imshow(labels[0], cmap='hot')

        draw_img2 = draw_labeled_boxes(np.copy(image), labels)
        if __debug__:
            # Plot the examples
            ax3 = plt.figure()
            ax3 = plt.clf()
            ax3 = plt.imshow(draw_img2)
            ax3 = plt.savefig('heatmap.png')
        # End of FOR loop

    # Process video
    video_input = VideoFileClip("./project_video.mp4")
    video_output = './OUTPUT_VIDEO.mp4'

    output_clip = video_input.fl_image(process_image)
    output_clip.write_videofile(video_output, audio=False)
