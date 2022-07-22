import gc
import glob
from multiprocessing import Pool

import numpy as np
import cv2
from PIL import Image


def artifact_filter(im, thresh):
    # Threshold values
    red_thresh = 256 * 256 * 1
    green_thresh = 256 * 256 * 0.001
    white_thresh = 256 * 256 * 0.3

    hsv_im = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)

    # Red color (invert to cyan)
    # low_red = np.array([83, 58, 145])
    # high_red = np.array([138, 255, 240])
    # inv_im = cv2.bitwise_not(im)
    # hsv_inv_im = cv2.cvtColor(inv_im, cv2.COLOR_RGB2HSV)
    # red_mask = cv2.inRange(hsv_inv_im, low_red, high_red)
    # red = cv2.bitwise_and(im, im, mask=red_mask)
    # red = (red > 0).sum(axis=2)

    # Green color
    low_green = np.array([25, 52, 72])
    high_green = np.array([102, 255, 255])
    green_mask = cv2.inRange(hsv_im, low_green, high_green)
    green = cv2.bitwise_and(im, im, mask=green_mask)
    green = (green > 0).sum(axis=2)

    # White background
    low = np.array([0, 0, 0])
    high = np.array([255, 16, 255])
    mask = cv2.inRange(hsv_im, low, high)
    white = cv2.bitwise_and(im, im, mask=mask)
    white = (white > 0).sum(axis=2)

    # reject = (red > 0).sum() > red_thresh  # or \
    reject = (green > 0).sum() > green_thresh or \
             (white > 0).sum() > white_thresh
    return not reject


def filter(data, out='data/test_W/'):
    img = np.array(Image.open(data[1]).convert('RGB'))
    if artifact_filter(img, data[2]):
        Image.fromarray(img).save(out + str(data[0]) + '.png')


def get_thresh_im(thresh):
    path = 'data/test/*.png'
    f = [(count, f, thresh) for count, f in enumerate(glob.glob(path))]
    p = Pool(20)
    p.map(filter, f)
    gc.collect()


get_thresh_im(0.3)

