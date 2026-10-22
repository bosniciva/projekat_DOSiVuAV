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


h, w = dist.shape[:2]

src = np.float32([[603, 460], [701, 460], [1045, 690], [287, 690]])

dst = np.float32([[320, 0], [960, 0], [960, 720], [320, 720]])

M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)

warped = cv2.warpPerspective(dist, M, (w, h), flags=cv2.INTER_LINEAR)

def warp(img):
    h, w = img.shape[:2]
    return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)

if __name__ == "__main__":
    img = cv2.imread('test_images/straight_lines2.jpg')
    undist = undistort(img)
    h, w = undist.shape[:2]

    src = np.float32([[598, 460], [706, 460], [1045, 690], [287, 690]])
    dst = np.float32([[320, 0], [960, 0], [960, 720], [320, 720]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    marked = undist.copy()
    cv2.polylines(marked, [src.astype(np.int32)], True, (0, 0, 255), 3)

    warped = cv2.warpPerspective(undist, M, (w, h), flags=cv2.INTER_LINEAR)
    cv2.polylines(warped, [dst.astype(np.int32)], True, (0, 0, 255), 3)

    binary_warped = cv2.warpPerspective(threshold_binary(undist), M, (w, h))

    cv2.imwrite('docs/warped_straight_lines2.png', np.hstack([marked, warped]))
    cv2.imwrite('docs/binary_warped2.png', binary_warped)
    cv2.imshow('Trapezoid | Bird-eye', np.hstack([marked, warped]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()