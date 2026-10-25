import cv2
import numpy as np
from collections import deque
from lane_detection import undistort, threshold_binary

cap = cv2.VideoCapture('test_videos/project_video03.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('output_video03.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

use_undistort = (w == 1280 and h == 720)
params = {} if w == 1280 else dict(s_min=60, l_min=60)

if use_undistort:
    src = np.float32([[0.467*w, 0.639*h], [0.552*w, 0.639*h], [0.816*w, 0.958*h], [0.224*w, 0.958*h]])
else:
    src = np.float32([[0.453*w, 0.625*h], [0.554*w, 0.625*h], [0.860*w, 0.972*h], [0.151*w, 0.972*h]])
dst = np.float32([[0.25*w, 0], [0.75*w, 0], [0.75*w, h], [0.25*w, h]])
M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)

lane_px = 0.5 * w
ym_per_pix = 30 / h
xm_per_pix = 3.7 / lane_px
ploty = np.linspace(0, h - 1, h)

N = 5
left_hist = deque(maxlen=N)
right_hist = deque(maxlen=N)

rejected = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    undist = undistort(frame) if use_undistort else frame
    mask = threshold_binary(undist, **params)
    mask[:int(0.6 * h)] = 0
    binary = cv2.warpPerspective(mask, M, (w, h)) // 255

    histogram = np.sum(binary[h // 2:, :], axis=0)
    midpoint = w // 2
    leftx_current = np.argmax(histogram[:midpoint])
    rightx_current = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 9
    margin = w // 12
    minpix = 40
    window_height = h // nwindows

    nonzeroy, nonzerox = binary.nonzero()
    left_inds, right_inds = [], []

    for window in range(nwindows):
        y_low = h - (window + 1) * window_height
        y_high = h - window * window_height

        good_left = ((nonzeroy >= y_low) & (nonzeroy < y_high) & (nonzerox >= leftx_current - margin) & (nonzerox < leftx_current + margin)).nonzero()[0]
        good_right = ((nonzeroy >= y_low) & (nonzeroy < y_high) & (nonzerox >= rightx_current - margin) & (nonzerox < rightx_current + margin)).nonzero()[0]

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

    accepted = False
    if len(leftx) > 100 and len(rightx) > 100 and np.ptp(lefty) > 0.5 * h and np.ptp(righty) > 0.5 * h:
        fit_l = np.polyfit(lefty, leftx, 2)
        fit_r = np.polyfit(righty, rightx, 2)
        xl = fit_l[0] * ploty**2 + fit_l[1] * ploty + fit_l[2]
        xr = fit_r[0] * ploty**2 + fit_r[1] * ploty + fit_r[2]

        width_bottom = xr[-1] - xl[-1]
        width_mid = xr[h // 2] - xl[h // 2]
        width_top = xr[0] - xl[0]
        widths = (width_bottom, width_mid, width_top)
        if all(0.8 * lane_px < x < 1.2 * lane_px for x in widths) and max(widths) - min(widths) < 0.15 * lane_px:
            left_hist.append(fit_l)
            right_hist.append(fit_r)
            accepted = True

    rejected = 0 if accepted else rejected + 1
    if rejected > 10:
        left_hist.clear()
        right_hist.clear()

    if len(left_hist) == 0:
        out.write(undist)
        continue

    weights = np.arange(1, len(left_hist) + 1)
    left_fit = np.average(left_hist, axis=0, weights=weights)
    right_fit = np.average(right_hist, axis=0, weights=weights)
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    left_fit_m = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
    right_fit_m = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)
    y_eval = (h - 1) * ym_per_pix
    left_curverad = ((1 + (2 * left_fit_m[0] * y_eval + left_fit_m[1])**2)**1.5) / max(abs(2 * left_fit_m[0]), 1e-9)
    right_curverad = ((1 + (2 * right_fit_m[0] * y_eval + right_fit_m[1])**2)**1.5) / max(abs(2 * right_fit_m[0]), 1e-9)
    curvature = (left_curverad + right_curverad) / 2

    lane_center = (left_fitx[-1] + right_fitx[-1]) / 2
    offset = (lane_center - w / 2) * xm_per_pix

    y_bottom_w = cv2.perspectiveTransform(np.float32([[[w / 2, h - 1]]]), M)[0, 0, 1]
    H2 = int(y_bottom_w) + 2
    y_draw = np.arange(0, H2)
    thickness = w // 64

    lines_warp = np.zeros((H2, w, 3), dtype=np.uint8)
    for fit in (left_fit, right_fit):
        x_draw = fit[0] * y_draw**2 + fit[1] * y_draw + fit[2]
        pts = np.int32(np.column_stack((x_draw, y_draw)))
        cv2.polylines(lines_warp, [pts], False, (0, 0, 255), thickness)
    lines_unwarp = cv2.warpPerspective(lines_warp, Minv, (w, h))

    result = undist.copy()
    m = lines_unwarp[:, :, 2] > 0
    result[m] = (0.35 * undist[m] + 0.65 * np.array([0, 0, 255])).astype(np.uint8)

    out.write(result)
    cv2.imshow('Video', result)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()