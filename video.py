import cv2
import numpy as np
from collections import deque
from lane_detection import undistort, threshold_binary

cap = cv2.VideoCapture('test_videos/challenge03.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('output_challenge03.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

use_undistort = (w == 1280 and h == 720)

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

N = 10
left_hist = deque(maxlen=N)
right_hist = deque(maxlen=N)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    undist = undistort(frame) if use_undistort else frame
    binary = cv2.warpPerspective(threshold_binary(undist), M, (w, h)) // 255

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

        good_left = ((nonzeroy >= y_low) & (nonzeroy < y_high) &
                     (nonzerox >= leftx_current - margin) & (nonzerox < leftx_current + margin)).nonzero()[0]
        good_right = ((nonzeroy >= y_low) & (nonzeroy < y_high) &
                      (nonzerox >= rightx_current - margin) & (nonzerox < rightx_current + margin)).nonzero()[0]

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

    if len(leftx) > 100 and len(rightx) > 100:
        fit_l = np.polyfit(lefty, leftx, 2)
        fit_r = np.polyfit(righty, rightx, 2)
        xl = fit_l[0] * ploty**2 + fit_l[1] * ploty + fit_l[2]
        xr = fit_r[0] * ploty**2 + fit_r[1] * ploty + fit_r[2]

        width_bottom = xr[-1] - xl[-1]
        width_top = xr[0] - xl[0]
        if 0.8 * lane_px < width_bottom < 1.2 * lane_px and 0.8 * lane_px < width_top < 1.2 * lane_px:
            left_hist.append(fit_l)
            right_hist.append(fit_r)

    if len(left_hist) == 0:
        out.write(undist)
        continue

    left_fit = np.mean(left_hist, axis=0)
    right_fit = np.mean(right_hist, axis=0)
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

    lines_warp = np.zeros((h, w, 3), dtype=np.uint8)
    for fitx in (left_fitx, right_fitx):
        pts = np.int32(np.column_stack((fitx, ploty)))
        cv2.polylines(lines_warp, [pts], False, (0, 0, 255), 14)
    lines_unwarp = cv2.warpPerspective(lines_warp, Minv, (w, h))
    result = cv2.addWeighted(undist, 1.0, lines_unwarp, 0.8, 0)

    side = "levo" if offset > 0 else "desno"
    cv2.putText(result, f"Radijus krivine: {curvature:.0f} m", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(result, f"Vozilo je {abs(offset):.2f} m {side} od centra", (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    out.write(result)
    cv2.imshow('Video', result)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()