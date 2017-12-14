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
    Slide a window from start to stop position over a region of interest (ROI) with overlapping
    :param img: Input image
    :param x_start_stop: ROI
    :param y_start_stop: ROI
    :param xy_window: Window size
    :param xy_overlap: Overlap
    :return: A list of windows
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


def color_hist(img, nbins=32, bins_range=(0, 256)):
    """
    Get color features
    :param img: Input image
    :param nbins: Number of bins
    :param bins_range: A value range within a bin
    :return: Image features derived from its histogram
    """
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def bin_spatial(img, size=(32, 32)):
    """
    Return the feature vector
    :param img: Input image
    :param size: Desired size
    :return: A contiguous flattened array
    """
    features = cv2.resize(img, size).ravel()
    return features


def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    """
    Get HOG features
    :param img: Input image
    :param orient: HOG orientations
    :param pix_per_cell: HOG pixels per cell
    :param cell_per_block: HOG cells per block
    :param vis: Generate a HOG image or not
    :param feature_vec: HOG features on or off
    :return: Feature vector
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


def extract_features(imgs, color_space='RGB', spatial_size=(32, 32), hist_bins=32, orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0, spatial_feat=True, hist_feat=True, hog_feat=True):
    """
    Extract features, combine them and normalize
    :param imgs: Input image
    :param color_space: A color space
    :param spatial_size: Spatial binning dimensions
    :param hist_bins: Number of histogram bins
    :param orient: HOG orientations
    :param pix_per_cell: HOG pixels per cell
    :param cell_per_block: HOG cells per block
    :param hog_channel: Channel of interest (0, 1, 2, ALL)
    :param spatial_feat: Spatial features on or off
    :param hist_feat: Histogram features on or off
    :param hog_feat: HOG features on or off
    :return: List of feature vectors
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


def single_img_features(img, color_space='RGB', spatial_size=(32, 32), hist_bins=32, orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0, spatial_feat=True, hist_feat=True, hog_feat=True):
    """
    Extract features from single image, combine them and normalize
    :param imgs: Input image
    :param color_space: A color space
    :param spatial_size: Spatial binning dimensions
    :param hist_bins: Number of histogram bins
    :param orient: HOG orientations
    :param pix_per_cell: HOG pixels per cell
    :param cell_per_block: HOG cells per block
    :param hog_channel: Channel of interest (0, 1, 2, ALL)
    :param spatial_feat: Spatial features on or off
    :param hist_feat: Histogram features on or off
    :param hog_feat: HOG features on or off
    :return: List of feature vectors
    """
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


def search_windows(img, windows, clf, scaler, color_space='RGB', spatial_size=(32, 32), hist_bins=32, hist_range=(0, 256), orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0):
    """
    Find windows of interest
    :param img: Input image
    :param windows: List of windows
    :param clf: Classifier
    :param scaler: A per-column scaler
    :param color_space: A color space
    :param spatial_size: Spatial binning dimensions
    :param hist_bins: Number of histogram bins
    :param hist_range: A range for values within a bin
    :param orient: HOG orientations
    :param pix_per_cell: HOG pixels per cell
    :param cell_per_block: HOG cells per block
    :param hog_channel: Channel of interest (0, 1, 2, ALL)
    :return: List of windows of interest
    """
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


def convert_color(img, conv='RGB2YCrCb'):
    """
    Perform a color space conversion.
    :param img: Input image
    :param conv: Desired color space
    :return: Converted image
    """
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)


def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    """
    Search for vehicles in an image
    Function uses a scale-factor to search with different window sizes
    :param img: Input image
    :param ystart: Vertical higher limit for search
    :param ystop: Vertical lower limit for search
    :param scale: Scaling factor
    :param svc: A classifier
    :param X_scaler: A per-column scaler
    :param orient: HOG orientations
    :param pix_per_cell: HOG pixels per cell
    :param cell_per_block: HOG cells per block
    :param spatial_size: Spatial binning dimensions
    :param hist_bins: Number of histogram bins
    :return: Processed image, list of boxes
    """
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255
    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    boxes = []
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), (np.int(imshape[0]/scale))))
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

        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False)

        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb * cells_per_step
                xpos = xb * cells_per_step
                # Extract hog features
                hog_feature1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_feature2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_feature3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feature1, hog_feature2, hog_feature3))

                x_left = xpos * pix_per_cell
                y_top = ypos * pix_per_cell

                # Extract img patch
                subimg = cv2.resize(ctrans_tosearch[y_top:y_top+window, x_left:x_left+window],(64,64))
                spatial_features = bin_spatial(subimg, size=spatial_size)
                hist_features = color_hist(subimg, nbins=hist_bins)

                test_features = X_scaler.transform(np.hstack((spatial_features, hist_features,
                                                             hog_features)).reshape(1, -1))
                test_prediction = svc.predict(test_features)

                if test_prediction == 1:
                    xbox_left = np.int(x_left * scale)
                    ytop_draw = np.int(y_top * scale)
                    win_draw = np.int(window*scale)
                    cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw, ytop_draw +
                                                                         win_draw + ystart), (0, 0, 255), 6)
                    boxes.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw, ytop_draw +
                                                                         win_draw + ystart)))
        return draw_img, boxes


def add_heat(heatmap, bbox_list):
    """
    Add a heatmap for each box in a list of boxes
    :param heatmap: a heatmap dummy
    :param bbox_list: list with boxes
    :return: Updated image
    """
    for box in bbox_list:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return heatmap


def apply_threshold(heatmap, threshold):
    """
    Clean up a heatmap image
    :param heatmap: a heatmap image
    :param threshold: Threshold
    :return: Updated image
    """
    heatmap[heatmap <= threshold] = 0
    return heatmap


def draw_labeled_boxes(img, labels):
    """
    Draw boxes for each target
    :param img: Input image
    :param labels: List of labels
    :return: Image with boxes over targets
    """
    for car_number in range(1, labels[1]+1):
        nonzero = (labels[0] == car_number).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        img = draw_boxes(img, [bbox], color=(0, 0, 255), thick=6)
    return img


def vehicle_detection(image, ystart, ystop, scale, svc, X_Scaler, orient,pix_per_cell, cell_per_block, spatial_size,
                      hist_bins, threshold):
    """
    Find vehicles in an image
    :param image: Input image
    :param ystart: Upper limit for search
    :param ystop: Lower limit for search
    :param scale: Scaling factor
    :param svc: A classifier for prediction
    :param X_Scaler: A per-column scaler
    :param orient: HOG orientations
    :param pix_per_cell: HOG pixels per cell
    :param cell_per_block: HOG cells per block
    :param spatial_size: patial binning dimensions
    :param hist_bins: Number of histogram bins
    :param threshold: Threshold
    :return: processed image with targets (if any) highlighted by boxes
    """
    image = cv2.undistort(image, mtx, dist, None, mtx)
    out_img, boxes = find_cars(image, ystart, ystop, scale, svc, X_Scaler, orient,
                               pix_per_cell, cell_per_block, spatial_size, hist_bins)
    heat = np.zeros_like(image[:, :, 0]).astype(np.float)
    box_list = boxes
    heat = add_heat(heat, box_list)
    heat = apply_threshold(heat, threshold)
    heatmap = np.clip(heat, 0, 255)
    labels = label(heatmap)
    draw_img = draw_labeled_boxes(np.copy(image), labels)
    return draw_img


def process_image(image):
    """
    Run a pipeline on each video frame
    :param image: frame image
    :return: processed image
    """
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
        ax1 = plt.figure()
        ax1 = plt.clf()
        ax1 = plt.subplot(121)
        ax1 = plt.imshow(image)
        ax1 = plt.subplot(122)
        ax1 = plt.imshow(result)
        ax1 = plt.savefig(fname[:-4] + '_1_UnitTestboxes.png')

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
        ax2 = plt.figure()
        ax2 = plt.clf()
        ax2 = plt.subplot(121)
        ax2 = plt.imshow(image)
        ax2 = plt.subplot(122)
        ax2 = plt.imshow(undist_image)
        ax2 = plt.savefig(fname[:-4] + '_2_UnitTestundistort.png')

    # Unit test on sliding window function
    if __debug__:
        fname = './tests/bbox-example-image.jpg'
        image = mpimg.imread(fname)
        windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[None, None], xy_window=(128, 128), xy_overlap=(0.5, 0.5))
        window_img = draw_boxes(image, windows, color=(0, 0, 255), thick=6)
        ax3 = plt.figure()
        ax3 = plt.clf()
        ax3 = plt.subplot(121)
        ax3 = plt.imshow(image)
        ax3 = plt.subplot(122)
        ax3 = plt.imshow(window_img)
        ax3 = plt.savefig(fname[:-4] + '_3_UnitTestslidingWindows.png')

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
        ax4 = plt.figure()
        ax4 = plt.clf()
        ax4 = plt.subplot(121)
        ax4 = plt.imshow(car_image)
        ax4 = plt.title('Example Car Image')
        ax4 = plt.subplot(122)
        ax4 = plt.imshow(notcar_image)
        ax4 = plt.title('Example Not-car Image')
        ax4 = plt.savefig('datasetExamples.png')

    # Get the features of cars and non-cars
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

    # Get an example of HOG
    if __debug__:
        print("An example of HOG feature extraction, check HOG_features.png file")
        test_img = mpimg.imread(cars[car_ind])
        gray = cv2.cvtColor(test_img, cv2.COLOR_RGB2GRAY)
        feature, hog_image = get_hog_features(gray, orient, pix_per_cell, cell_per_block, vis=True, feature_vec=False)
        # Plot the examples
        ax5 = plt.figure()
        ax5 = plt.clf()
        ax5 = plt.subplot(121)
        ax5 = plt.imshow(test_img, cmap='gray')
        ax5 = plt.title('Example Car Image')
        ax5 = plt.subplot(122)
        ax5 = plt.imshow(hog_image, cmap='gray')
        ax5 = plt.title('HOG Visualization')
        ax5 = plt.savefig('HOG_features.png')

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
        windows1 = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                                xy_window=(100, 100), xy_overlap=(0.5, 0.5))
        windows2 = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                                xy_window=(200, 200), xy_overlap=(0.3, 0.3))
        windows3 = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                                xy_window=(64, 64), xy_overlap=(0.3, 0.3))

        # Get the found windows that match the features as list
        hot_windows1 = search_windows(image, windows1, svc, X_rescaled, color_space=color_space,
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
        window_img1 = draw_boxes(draw_image, hot_windows1, color=(0, 0, 255), thick=6)
        window_img2 = draw_boxes(window_img1, hot_windows2, color=(0, 0, 255), thick=6)
        window_img3 = draw_boxes(window_img2, hot_windows3, color=(0, 0, 255), thick=6)

        if __debug__:
            print("Check files with boxes/windows...")
            # Plot the examples
            ax6 = plt.figure()
            ax6 = plt.clf()
            ax6 = plt.subplot(121)
            ax6 = plt.imshow(image)
            ax6 = plt.title('Test Image')
            ax6 = plt.subplot(122)
            ax6 = plt.imshow(window_img3)
            ax6 = plt.title('Boxes')
            ax6 = plt.savefig(f_name[:-4] + '_A_Window3.png')
            ax7 = plt.figure()
            ax7 = plt.clf()
            ax7 = plt.subplot(121)
            ax7 = plt.imshow(image)
            ax7 = plt.title('Test Image')
            ax7 = plt.subplot(122)
            ax7 = plt.imshow(window_img2)
            ax7 = plt.title('Boxes')
            ax7 = plt.savefig(f_name[:-4] + '_B_Window2.png')
            ax8 = plt.figure()
            ax8 = plt.clf()
            ax8 = plt.subplot(121)
            ax8 = plt.imshow(image)
            ax8 = plt.title('Test Image')
            ax8 = plt.subplot(122)
            ax8 = plt.imshow(window_img1)
            ax8 = plt.title('Boxes')
            ax8 = plt.savefig(f_name[:-4] + '_C_Window1.png')

        ystart = 400  # Upper limit
        ystop = 656  # Lower limit
        scale = 1.5  # Scale factor

        out_img, boxes = find_cars(image, ystart, ystop, scale, svc, X_rescaled, orient, pix_per_cell, cell_per_block,
                                   spatial_size, hist_bins)
        if __debug__:
            print("Boxes found: ", boxes)

        threshold = 1
        heat = np.zeros_like(image[:, :, 0]).astype(np.float)
        heat = add_heat(heat, boxes)
        heat = apply_threshold(heat, threshold)
        heatmap = np.clip(heat, 0, 255)

        labels = label(heatmap)
        if __debug__:
            print("Vehicles found: ", labels[1])
            ax9 = plt.figure()
            ax9 = plt.clf()
            ax9 = plt.subplot(121)
            ax9 = plt.imshow(image)
            ax9 = plt.title('Test Image')
            ax9 = plt.subplot(122)
            ax9 = plt.imshow(heatmap, cmap='hot')
            ax9 = plt.title('Detected targets - heatmap')
            ax9 = plt.savefig(f_name[:-4] + '_D_heat.png')

        draw_img2 = draw_labeled_boxes(np.copy(image), labels)

        if __debug__:
            ax10 = plt.figure()
            ax10 = plt.clf()
            ax10 = plt.subplot(121)
            ax10 = plt.imshow(image)
            ax10 = plt.title('Test Image')
            ax10 = plt.subplot(122)
            ax10 = plt.imshow(draw_img2)
            ax10 = plt.title('Detected targets')
            ax10 = plt.savefig(f_name[:-4] + '_E_targets.png')

        # End of FOR loop

    # Process video
    video_input = VideoFileClip("./project_video.mp4")
    video_output = './OUTPUT_VIDEO.mp4'

    output_clip = video_input.fl_image(process_image)
    output_clip.write_videofile(video_output, audio=False)
