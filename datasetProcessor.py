# -*- coding: utf-8 -*-
# TODO: define path
# TODO: For each file at that path: read it and append to dataset
# TODO: Save that dataset as a pickle

import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import time
# from skimage import color, exposure

def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    """
    Define a function to return HOG features and visualization
    :param img:
    :param orient:
    :param pix_per_cell:
    :param cell_per_block:
    :param vis:
    :param feature_vec:
    :return:
    """
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                                  visualise=True, feature_vector=False)
        return features, hog_image
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                       visualise=False, block_norm='L2-Hys', feature_vector=feature_vec)
        return features


# Define a function to compute color histogram features
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(img, cspace='RGB', spatial_size=(32, 32), hist_bins=32, hist_range=(0, 256)):
    # Create a list to append feature vectors to
    features = []
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
        else:
            feature_image = np.copy(image)
        # Apply bin_spatial() to get spatial color features
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # Apply color_hist() also with a color space option now
        hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
        # Append the new feature vector to the features list
        features.append(np.concatenate((spatial_features, hist_features)))
    # Return list of feature vectors
    return features


# Define a function to compute binned color features
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features


if __name__ == "__main__":
    # For debug mode, please run this script with -O option
    if __debug__:
        print("Execution has been started...")

    vehicles = glob.glob('./dataset/vehicles/*/*.png')
    objects = []
    labels = []

    for image in vehicles:
        objects.append(image)
        labels.append(1)

    non_vehicles = glob.glob('./dataset/non-vehicles/*/*.png')

    for image in non_vehicles:
        objects.append(image)
        labels.append(0)

    if __debug__:
        print("Images were loaded successfully!")
        # Just for fun choose random car / not-car indices and plot example images
        car_ind = np.random.randint(0, len(vehicles))
        notcar_ind = np.random.randint(len(vehicles), len(non_vehicles))

        # Read in car / not-car images
        car_image = mpimg.imread(objects[car_ind])
        notcar_image = mpimg.imread(objects[notcar_ind])
        print("Car image has label: ", labels[car_ind])
        print("Not-a-Car image has label: ", labels[notcar_ind])

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

    images = []

    for object in objects:
        image = cv2.imread(object)
        images.append(image)

    X_dataset = np.array(images)
    y_dataset = np.array(labels)

    if __debug__:
        print("Dataset is ready!")
        print(X_dataset.shape)
        print(y_dataset.shape)

    shuffle(X_dataset, y_dataset)
    X_train, X_validation, y_train, y_validation = train_test_split(X_dataset, y_dataset, test_size=0.2)

    if __debug__:
        print("Dataset is shuffled and split!")
        print("X_train shape: ", X_train.shape)
        print("y_train shape: ", y_train.shape)
        print("X_validation shape: ", X_validation.shape)
        print("y_validation shape: ", y_validation.shape)

    X_train_features = []
    X_train_HOGimage = []

    for image in range(X_train.shape[0]):
        img = X_train[image,:,:,:]
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Define HOG parameters
        orient = 9
        pix_per_cell = 8
        cell_per_block = 2
        # Call our function with vis=True to see an image output
        feature, hog_image = get_hog_features(gray, orient, pix_per_cell, cell_per_block, vis=True, feature_vec=False)
        X_train_features.append(feature)
        X_train_HOGimage.append(hog_image)

    if __debug__:
        print("HOG feature extraction is done!")
        print("X_train_features shape: ", len(X_train_features))
        print("X_train_HOGimage shape: ", len(X_train_HOGimage))
        # Plot the examples
        ax2 = plt.figure()
        ax2 = plt.clf()
        ax2 = plt.subplot(121)
        ax2 = plt.imshow(X_train[100], cmap='gray')
        ax2 = plt.title('Example Car Image')
        ax2 = plt.subplot(122)
        ax2 = plt.imshow(X_train_HOGimage[100], cmap='gray')
        ax2 = plt.title('HOG Visualization')
        ax2 = plt.savefig('HOG_features.png')

    #X_train_color = []

    #for image in range(X_train.shape[0]):
    #    img = X_train[image, :, :, :]
    #    car_feature = extract_features(img, cspace='HLS', spatial_size=(32, 32), hist_bins=32, hist_range=(0, 256))
    #    X_train_color.append(car_feature)

    #if __debug__:
    #    pass

    #print("Color feature extraction is done!")

    #if len(X_train_color) > 0:
        # Create an array stack of feature vectors
    #    X = np.vstack((car_features, notcar_features)).astype(np.float64)
        # Fit a per-column scaler
    #    X_scaler = StandardScaler().fit(X)
    #    # Apply the scaler to X
    #    scaled_X = X_scaler.transform(X)
    #    car_ind = np.random.randint(0, len(cars))
    #    # Plot an example of raw and scaled features
    #    fig = plt.figure(figsize=(12, 4))
    #    plt.subplot(131)
    #    plt.imshow(mpimg.imread(cars[car_ind]))
    #    plt.title('Original Image')
    #    plt.subplot(132)
    #    plt.plot(X[car_ind])
    #    plt.title('Raw Features')
    #    plt.subplot(133)
    #    plt.plot(scaled_X[car_ind])
    #    plt.title('Normalized Features')
    #    fig.tight_layout()
    #else:
    #    print('Your function only returns empty feature vectors...')


    if __debug__:
        print("Classifier section begin...")

    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train_features, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_validation, y_validation), 4))
    # Check the prediction time for a single sample
    t = time.time()
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_validation[0:n_predict]))
    print('For these', n_predict, 'labels: ', y_validation[0:n_predict])
    t2 = time.time()
    print(round(t2 - t, 5), 'Seconds to predict', n_predict, 'labels with SVC')