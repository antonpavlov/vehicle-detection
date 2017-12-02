# -*- coding: utf-8 -*-

# *** IMPORTS ***
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from sklearn import svm


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