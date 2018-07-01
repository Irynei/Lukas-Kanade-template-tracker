import argparse
import glob
import os
import cv2
from copy import copy
import numpy as np


def get_corner_points(roi):
    top_left = tuple(roi[:2])
    bottom_right = (roi[0] + roi[2], roi[1] + roi[3])
    return top_left, bottom_right


def show_image(image, roi, roi_opencv, roi_camshift):
    top_left, bottom_right = get_corner_points(roi)
    top_left_opencv, bottom_right_opencv = get_corner_points(roi_opencv)
    top_left_camshift, bottom_right_camshift = get_corner_points(roi_camshift)
    cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), 2)
    cv2.rectangle(image, top_left_opencv, bottom_right_opencv, 255, 2)
    cv2.rectangle(image, top_left_camshift, bottom_right_camshift, (0, 255, 0), 2)
    cv2.imshow("Red is custom meanshift, blue - opencv meanshift, green - camshift", image)
    k = cv2.waitKey(60) & 0xff


def cut_patch(image, patch):
    return image[patch[1] - 1:patch[1] + patch[3] - 1, patch[0] - 1:patch[0] + patch[2] - 1]


def get_roi(center, window_size):
    return np.array([
        int(np.ceil(center[0])) - window_size[0] // 2,
        int(np.ceil(center[1])) - window_size[1] // 2,
        window_size[0],
        window_size[1]
    ])


def mean_shift(src, window):
    num_of_iterations = 70
    min_distance = 1
    centroid = np.zeros(2)
    for i in range(num_of_iterations):
        roi = cut_patch(src, get_roi(centroid + window[:2], window[2:4]))
        new_centroid = np.array([np.mean(np.argwhere(roi > 0)[:, 1]), np.mean(np.argwhere(roi > 0)[:, 0])])
        if np.linalg.norm(centroid - new_centroid) < min_distance:
            # print("stopping early")
            break
        else:
            centroid = new_centroid.copy()

    return get_roi(centroid + window[:2], window[2:4])


def preprocess_image(img, track_window):
    roi = cut_patch(img, track_window)
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    return roi_hist


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--roi', nargs='+', type=int, default=[270, 150, 100, 100])
    parser.add_argument('--dataset', type=str, default='Coke/img/')
    args = parser.parse_args()

    image_list = sorted(glob.glob(os.path.join(args.dataset, '*.jpg')))
    images = [cv2.imread(img_name) for img_name in image_list]

    window_size = [100, 100]
    target_point = np.array([320, 200])
    prev_img = images.pop()
    track_window = args.roi or get_roi(target_point, window_size)

    roi_hist = preprocess_image(prev_img, track_window)
    track_window_meanshift_opencv = copy(track_window)
    track_window_camshift = copy(track_window)
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    for idx, curr_img in enumerate(images):
        # print("Image: {}".format(idx))
        hsv = cv2.cvtColor(curr_img, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        track_window = mean_shift(dst, track_window)
        ret, track_window_meanshift_opencv = cv2.meanShift(dst, tuple(track_window_meanshift_opencv), term_crit)
        ret_1, track_window_camshift = cv2.CamShift(dst, tuple(track_window_camshift), term_crit)
        show_image(curr_img, track_window, track_window_meanshift_opencv, track_window_camshift)

    cv2.destroyAllWindows()
