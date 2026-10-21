import cv2
import numpy as np

data = np.load('calib.npz')
mtx, dist = data['mtx'], data['dist']


def undistort(img):
    h, w = img.shape[:2]
    newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
    return cv2.undistort(img, mtx, dist, None, newCameraMtx)


def threshold_binary(img, s_thresh=170, sx_thresh=120, l_thresh=200):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    blur = cv2.GaussianBlur(l_channel, (5, 5), 0)
    sobelX = cv2.Sobel(blur, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
    absX = cv2.convertScaleAbs(sobelX)
    _, sx_binary = cv2.threshold(absX, sx_thresh, 255, cv2.THRESH_BINARY)

    _, s_binary = cv2.threshold(s_channel, s_thresh, 255, cv2.THRESH_BINARY)
    _, l_binary = cv2.threshold(l_channel, l_thresh, 255, cv2.THRESH_BINARY)

    return cv2.bitwise_or(sx_binary, cv2.bitwise_or(s_binary, l_binary))


if __name__ == "__main__":
    img = cv2.imread('test_images/test1.jpg')
    undist = undistort(img)
    binary = threshold_binary(undist)

    cv2.imwrite('docs/undistort_road_compare.png', np.hstack([img, undist]))
    cv2.imwrite('docs/binary_combo.png', binary)

    hls = cv2.cvtColor(undist, cv2.COLOR_BGR2HLS)
    blur = cv2.GaussianBlur(hls[:, :, 1], (5, 5), 0)
    absX = cv2.convertScaleAbs(cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3))

    cv2.imshow('Original - Undistorted', np.hstack([img, undist]))
    cv2.imshow('Binary', binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()