import os
import cv2
import glob
import argparse
import numpy as np
from matplotlib import pyplot as plt


def affine_2d_transform(image, p):
    rows, cols = image.shape[:2]
    M = np.array([[1 + p[0], p[2], p[4]], [p[1], 1 + p[3],  p[5]]])
    result = cv2.warpAffine(image, M, (cols, rows), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return result


def cut_patch(image, patch):
    return image[patch[1] - 1:patch[1] + patch[3] - 1, patch[0] - 1:patch[0] + patch[2] - 1]


def jacobian(array):
    f = lambda x: np.array([[x[0], 0, x[1], 0, 1, 0], [0, x[0], 0, x[1], 0, 1]])
    return np.apply_along_axis(f, 1, array)


def get_roi(target, window_size):
    return np.array([
        int(np.ceil(target[0])) - window_size[0] // 2,
        int(np.ceil(target[1])) - window_size[1] // 2,
        window_size[0],
        window_size[1]
    ])


def show_image(image, window_size, region_of_interest):
    top_left = tuple(region_of_interest[:2])
    bottom_right = tuple([int(top_left[i] + window_size[i]) for i in range(2)])
    cv2.rectangle(image, top_left, bottom_right, 255, 2)
    plt.imshow(image, cmap='gray')
    plt.show()


def lukas_kanade_tracking(image_list, target, target_point, region_of_interest):
    num_iterations = 50
    for idx, img in enumerate(image_list):
        next_image = cv2.imread(img, 0)
        img_copy = next_image.copy()
        dpMag = []
        params = np.zeros(6, dtype="float32")

        for i in range(num_iterations):
            warped_img = affine_2d_transform(next_image, params)
            candidate = cut_patch(warped_img, region_of_interest)

            # gx, gy = np.gradient(img_copy)

            gx = cv2.Sobel(next_image, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(next_image, cv2.CV_32F, 0, 1, ksize=3)

            gx_w = affine_2d_transform(gx, params)
            gy_w = affine_2d_transform(gy, params)

            gx_w = cut_patch(gx_w, region_of_interest)
            gy_w = cut_patch(gy_w, region_of_interest)

            X, Y = np.meshgrid(range(candidate.shape[0]), range(candidate.shape[1]))
            coords_2d = np.array([X.flatten(), Y.flatten()]).transpose()
            grad_image = np.array([gx_w.flatten(), gy_w.flatten()]).transpose()

            jacob = jacobian(coords_2d)
            steepest_descent = np.empty(shape=(grad_image.shape[0], jacob.shape[2]), dtype='float64')
            for i in range(grad_image.shape[0]):
                steepest_descent[i] = np.dot(grad_image[i], jacob[i])

            hessian = np.dot(steepest_descent.transpose(), steepest_descent)
            error_image = np.subtract(target, candidate, dtype='float64')

            cost = np.sum(steepest_descent * np.tile(error_image.flatten(), (len(params), 1)).T, axis=0)
            dp = np.dot(np.linalg.inv(hessian), cost.T)
            eps = 1e-2
            norm = np.linalg.norm(dp)
            dpMag.append(norm)

            if norm < eps:
                print('stopping early')
                break
            else:
                params += dp.T

        # transform target point
        affine_transform = np.array([[1 + params[0], params[2], params[4]], [params[1], 1 + params[3], params[5]]])
        new_target_point = affine_transform.dot(np.append(target_point, 1))
        new_region_of_interest = get_roi(new_target_point, window_size)
        print("New target point: {}".format(new_target_point))
        # update target, roi and target point
        target = cut_patch(next_image, new_region_of_interest)
        region_of_interest = new_region_of_interest
        target_point = new_target_point
        if idx % 5 == 0:
            print("image {}".format(idx))
        # show image
        show_image(img_copy, window_size, region_of_interest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--roi', nargs='+', type=int, default=[260, 140, 120, 120])
    parser.add_argument('--dataset', type=str, default='Coke/img/')
    args = parser.parse_args()

    image_list = sorted(glob.glob(os.path.join(args.dataset, '*.jpg')))
    img_cur = cv2.imread(image_list.pop(0), 0)

    target_point = np.array([320, 200])
    window_size = (120, 120)
    region_of_interest = args.roi or get_roi(target_point, window_size)
    target = cut_patch(img_cur, region_of_interest)

    lukas_kanade_tracking(image_list, target, target_point, region_of_interest)
