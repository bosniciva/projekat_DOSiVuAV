import cv2
import numpy as np

data = np.load('calib.npz')
mtx, dist = data['mtx'], data['dist']


def undistort(img):
    h, w = img.shape[:2]
    newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    return cv2.undistort(img, mtx, dist, None, newCameraMtx)


def threshold_binary(img, s_thresh=(190, 255), sx_thresh=(60, 100)):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    sobelx = np.absolute(cv2.Sobel(l_channel, cv2.CV_64F, 1, 0))
    scaled = np.uint8(255 * sobelx / max(np.max(sobelx), 1e-6))

    sx_binary = cv2.inRange(scaled, sx_thresh[0], sx_thresh[1])
    s_binary = cv2.inRange(s_channel, s_thresh[0], s_thresh[1])

    return cv2.bitwise_or(sx_binary, s_binary)


if __name__ == "__main__":
    img = cv2.imread('test_images/test1.jpg')
    undist = undistort(img)
    binary = threshold_binary(undist)

    cv2.imwrite('docs/undistort_road_compare.png', np.hstack([img, undist]))
    cv2.imwrite('docs/binary_combo.png', binary)

    cv2.imshow('Original - Undistorted', np.hstack([img, undist]))
    cv2.imshow('Binary', binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()