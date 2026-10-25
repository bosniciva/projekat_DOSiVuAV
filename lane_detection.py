import cv2
import numpy as np

data = np.load('calib.npz')
mtx, dist = data['mtx'], data['dist']

def undistort(img):
    h, w = img.shape[:2]
    newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
    return cv2.undistort(img, mtx, dist, None, newCameraMtx)

def threshold_binary(img, l_thresh=200, s_min=130, l_min=40):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    l_channel = hls[:, :, 1]

    yellow = cv2.inRange(hls, (15, l_min, s_min), (35, 255, 255))
    _, white = cv2.threshold(l_channel, l_thresh, 255, cv2.THRESH_BINARY)

    return cv2.bitwise_or(yellow, white)

if __name__ == "__main__":
    img = cv2.imread('test_images/test1.jpg')
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

    binary = binary_warped // 255
    h, w = binary.shape

    histogram = np.sum(binary[h // 2:, :], axis=0)
    midpoint = w // 2
    leftx_current = np.argmax(histogram[:midpoint])
    rightx_current = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 9
    margin = 100
    minpix = 50
    window_height = h // nwindows

    nonzeroy, nonzerox = binary.nonzero()
    left_inds, right_inds = [], []
    out_img = np.dstack((binary, binary, binary)) * 255

    for window in range(nwindows):
        y_low = h - (window + 1) * window_height
        y_high = h - window * window_height

        xl_low, xl_high = leftx_current - margin, leftx_current + margin
        xr_low, xr_high = rightx_current - margin, rightx_current + margin

        cv2.rectangle(out_img, (xl_low, y_low), (xl_high, y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (xr_low, y_low), (xr_high, y_high), (0, 255, 0), 2)

        good_left = ((nonzeroy >= y_low) & (nonzeroy < y_high) & (nonzerox >= xl_low) & (nonzerox < xl_high)).nonzero()[0]
        good_right = ((nonzeroy >= y_low) & (nonzeroy < y_high) & (nonzerox >= xr_low) & (nonzerox < xr_high)).nonzero()[0]

        left_inds.append(good_left)
        right_inds.append(good_right)

        if len(good_left) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left]))
        if len(good_right) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right]))

    left_inds = np.concatenate(left_inds)
    right_inds = np.concatenate(right_inds)

    leftx, lefty = nonzerox[left_inds], nonzeroy[left_inds]
    rightx, righty = nonzerox[right_inds], nonzeroy[right_inds]

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, h - 1, h)
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    out_img[lefty, leftx] = [0, 0, 255]
    out_img[righty, rightx] = [255, 0, 0]
    for fitx in (left_fitx, right_fitx):
        pts = np.int32(np.column_stack((fitx, ploty)))
        cv2.polylines(out_img, [pts], False, (0, 255, 255), 3)

    ym_per_pix = 30 / 720
    xm_per_pix = 3.7 / 640

    left_fit_m = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_m = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

    y_eval = (h - 1) * ym_per_pix
    left_curverad = ((1 + (2 * left_fit_m[0] * y_eval + left_fit_m[1])**2)**1.5) / abs(2 * left_fit_m[0])
    right_curverad = ((1 + (2 * right_fit_m[0] * y_eval + right_fit_m[1])**2)**1.5) / abs(2 * right_fit_m[0])
    nl, nr = len(leftx), len(rightx)
    curvature = (nl * left_curverad + nr * right_curverad) / (nl + nr)
    
    lane_center = (left_fitx[-1] + right_fitx[-1]) / 2
    offset = (lane_center - w / 2) * xm_per_pix

    print(f"Left radius:  {left_curverad:.0f} m")
    print(f"Right radius: {right_curverad:.0f} m")
    print(f"Curvature:    {curvature:.0f} m")
    print(f"Offset:       {offset:.2f} m")

    lane_warp = np.zeros((h, w, 3), dtype=np.uint8)
    pts_left = np.int32(np.column_stack((left_fitx, ploty)))
    pts_right = np.int32(np.column_stack((right_fitx, ploty)))[::-1]
    cv2.fillPoly(lane_warp, [np.vstack((pts_left, pts_right))], (0, 255, 0))

    lane_unwarp = cv2.warpPerspective(lane_warp, Minv, (w, h))
    result = cv2.addWeighted(undist, 1.0, lane_unwarp, 0.3, 0)

    cv2.imwrite('docs/example_output1.png', result)
    cv2.imshow('Rezultat', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()