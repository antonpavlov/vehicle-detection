# -*- coding: utf-8 -*-

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn import svm


def draw_boxes(input_image, bboxes, color=(0, 0, 255), thick=6):
    """
    Draw boxes in an input image
    :param img: An input image
    :param bboxes: Tuples with coordinates of boxes
    :param color: Line color RGB
    :param thick: Line thickness
    :return: An image with boxes
    """
    # Make a copy of the image
    draw_img = np.copy(input_image)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(draw_img, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return draw_img


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