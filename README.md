**Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./docs/undistort_output.png "Undistorted"
[image2]: ./docs/undistort_road.png "Road Transformed"
[image3]: ./docs/binary_combo.png "Binary Combo"
[image4]: ./docs/warped_straight_lines.png "Warp"
[image5]: ./docs/warped_straight_lines2.png "Warp"
[image6]: ./docs/color_fit_test1.png "Fit Visual 1"
[image7]: ./docs/color_fit_test2.png "Fit Visual 2"
[image8]: ./docs/example_output1.png "Output 1"
[image9]: ./docs/example_output2.png "Output 2"
[image10]: ./docs/example_output3.png "Output 3"
[video1]: ./output_video01.mp4 "Video 1"
[video2]: ./output_video02.mp4 "Video 2"
[video3]: ./output_video03.mp4 "Video 3"

---

### Writeup / README

#### 1. Provide a Writeup that includes all the rubric points and how you addressed each one.

Camera calibration was done with checkerboard pictures in calibration.py, saved in calib.npz.
Processing images from test images, undistort() and threshold_binary() functions were done in lane_detection.py
Video processing (perspective, sliding window, fit, smoothing, drawing) was done in video.py.  

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

Firstly cv2.findChessboardCorners looks for the inner corners of the board. The whole board has 9x6 interior corners, so I'm looking for that layout first. In the three images (calibration1.jpg, calibration4.jpg and calibration5.jpg) part of the board is not visible, so 9x6 was not found. That's why for each image I try smaller layouts (9x6, 8x6, 9x5, 7x6, 8x5, 6x6, 9x4) and take the first one found. Result was that 17 images are recognized as 9x6, calibration1 as 9x5, calibration4 as 6x6 and calibration5 as 7x6, which means that all 20 images were used in the calibration.
Second cv2.cornerSubPix determines the position of corners more precisely, with a stopping norm of 30 iterations or a change of less than 0.001.
I create objectPoints according to the layout size found in that image: (0,0,0), (1,0,0), (2,0,0),... The metric scale doesn't matter here because I'm only looking for camera parameters.
For each successful image, I save a pair: ideal points and found points in the image. From those pairs, cv2.calibrateCamera calculates: camera matrix mtx (focal lengths fx, fy and image center cx, cy) and distortion coefficients.

The advantage of 20 images being used instead of 17 is that these are the images where the board is at the edge of the frame, where the distortion is the strongest. 

I save the result in calib.npz and load it in the rest of the project.

I check the quality of the calibration with the reprojection error, using cv2.projectPoints. It calculates where the angles would be with the obtained parameters, I compare it with the actually found angles and average across the images. Total error: 0.124. It is a small value, which means that the angles projected from the obtained parameters match well with those found by the detector.


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

First, distortion correction was done on checkerboard picture:

![Undistorted Output][image1]


The undistort() function in lane_detection.py uses getOptimalNewCameraMatrix with parameter alpha=0, then cv2.undistort. With alpha=1, all pixels remain, but black zones appear at the edges of the image. Those zones in the next step create false edges, so I set alpha=0 because it cuts to the correct part of the image, which is then slightly enlarged:

![Road Transformed][image2]


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

Threshold_binary() function in lane_detection.py. The image is transferred to the HLS space, and the lines are searched for by color:
1. Yellow line: cv2.inRange(hls, (15, l_min, s_min), (35, 255, 255)). The shade of yellow is 15 to 35 (in OpenCV the shade goes from 0 to 180), with a minimum saturation of s_min and a minimum brightness of l_min.
2. White line: cv2.threshold on light channel (L) with threshold 200.
3. Union of yellow and white masks (cv2.bitwise_or).

For videos pixels above 0.6 * h (above the horizon) are set to zero before the perspective transformation, so that the sky and trees do not enter the bird's eye view.

What I tried and rejected:
1. Grayscale threshold + Canny: grayscale threshold does not distinguish line from light concrete and falls on shadows, and Canny gives edges of everything.
2. Sobel by x: captures the edges of the line, not its body, and also the edges of fences, trees and cars. In the test image, almost every edge was out of the way, so it only added noise.
3. Threshold only on saturation (S > 170): worked well close to the camera, but had two problems. First one was that in HLS space, the saturation is numerically unstable for near-black pixels, so the entire shadow entered the mask. That's why the lower limit of brightness l_min was added. Second one was that the far part of the yellow line is paler, so a fixed threshold of 170 only captured the near part of the line, and the polynomial extrapolated the rest and deviated from the line.

![Binary Combo][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Perspective transformation is in lane_detection.py (images) and video.py (video), using cv2.getPerspectiveTransform and cv2.warpPerspective. 

I determined the source points on the images of the straight road (straight_lines1.jpg, straight_lines2.jpg) so that the trapezoid lies on the inner edges of the lines. I verified that the lines in the top view are vertical and parallel in both images, which means that the transformation is not only applied to one image. The first attempt with the wrong numbers gave tilted lines, so I measured the points on the corrected image.

I also calculate the inverse matrix (Minv), so it can be returned to the original image.

![Wrap][image4]

![Wrap][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code is in lane_detection.py (images) and video.py (video). I use histogram and sliding windows:
1. Histogram: I sum the white pixels by column in the bottom half of the binary image. The two vertices are the initial positions of the left and right lines. I use the bottom half because that's where the line is most visible.
2. Sliding windows: from each initial position I stack 9 windows upwards, width ±100 px. In each window I collect white pixels. If there are more than minpix, I center the next window on their average x, so the windows follow the line even when it curves. If there are not enough, the window stays in place. In the space between dashes, the window waits, then continues the line on the next dash.
3. Polynomial: np.polyfit(y, x, 2) gives x = A·y² + B·y + C for each line. I fit x as a function of y because the lines are nearly vertical, so y(x) would have a nearly vertical tilt and be unstable.

Additional protections in the video (video.py), because the same problems occur frame by frame:
1. The fit is accepted only if both lines have more than 100 pixels and those pixels cover at least half the height of the image,
2. The spacing between the bottom, middle and top lines must be around 640px and almost constant (change below 15%),
3. Smoothing: weighted average of the last 5 accepted fits,
4. If nothing is accepted for 10 frames in a row, the history is cleared and "Lane not found" is displayed, instead of drawing the outdated bar.

![Fit Visual 1][image6]
![Fit Visual 2][image7]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Fitting is done in meters. Assumptions are:
xm_per_pix = 3.7 / 640: the width of the lane is 3.7 m (US standard), and in the bird's eye view it is 640 px,
ym_per_pix = 30 / 720: about 30 m road is visible in the view from above.

The radius of curvature for x = A·y² + B·y + C is:
R = (1 + (2·A·y + B)²)^1.5 / |2·A|

The second derivative tells how much the curve bends, the smaller A, the flatter the curve, and larger the R. The expression (2Ay + B) is the tilt of the line and corrects for the line being tilted. I calculate it at the point closest to the vehicle. For the final number, I take the average of the left and right lines.

Vehicle position is calculated with the middle of the bar at the bottom of the image, it's the average of the x-coordinates of the left and right curves. The difference from the center of the image, multiplied by xm_per_pix, gives the offset.

The results:
Image 1:
Left radius:  1059m
Right radius: 1453m
Curvature:    1103m
Offset:       0.16m


Image 2:
Left radius:  676m
Right radius: 2014m
Curvature:    850m
Offset:       0.39m

Image 3:
Left radius:  1600m
Right radius: 798m
Curvature:    1399m
Offset:       0.12m


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I draw the bar in the top view (simple: the area between two curves, cv2.fillPoly), then with the inverse matrix Minv (cv2.warpPerspective) I return it to the camera perspective and transparently place it over the corrected image (cv2.addWeighted).

![Output 1][image8]

![Output 2][image9]

![Output 3][image10]


### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Video in video.py processes each frame with the same steps (undistort, threshold, perspective, windowing, fit, checks, smoothing, drawing). The lines are shown as red, transparent lines over the lines.

Video 1:
![Video 1][video1]

Video 2:
![Video 2][video2]

Video 3:
![Video 3][video3]


### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Some of the problems:
1. Perspective: the keystone numbers for the standard frame didn't work after alpha=0, so the lines were skewed. Solution: measure on a corrected image and check on two straight road images.
2. Sobel was making noise, and the color is more stable.
3. Saturation is unstable for dark pixels: in the shadow, all the asphalt entered the mask. The solution: the lower limit of brightness, and yellow by hue, not just by saturation.
4. Pale yellow line in the distance: the fixed threshold was only capturing the near part, so the curve deviated. Solution: Yellow in hue with a lower saturation threshold. Price: on light concrete, the same threshold floods the mask, so the thresholds are parameters per image type.
5. Smoothing delay: the average of the last 10 fits is delayed by about 4.5 frames, when the line moves laterally to about 2.3px per frame, that's about a 10px shift. Solution: 5 fits with linear weights.
6. Mask collapse at the concrete - shadow transition. Solution: check that the pixels cover at least half the height.

Where the pipeline goes wrong:
1. Challenging footage (forest road with sharp curves, double yellow lines, strong shadows etc.): most frames are rejected by the pipeline. Threshold L > 200 marks sunny asphalt as a "white line", so the mask becomes flooded. In sharp curves, the lines in the top view have an S-shape that the quadratic polynomial cannot represent, the trapeze for a straight road does not apply there. Pipeline then prints "Lane not found", which is more correct than drawing an wrong line.
2. Weak or faded mark, snow, rain, night, very strong shadow with trees.
3. A vehicle in front obscuring the lane, or changing lanes.
4. The result depends on the assumptions: lane width 3.7 m, 30 m of visible road, camera in the middle of the vehicle.

What would improve robustness:
1. Thresholds that adapt to illumination (local contrast) instead of fixed values. I've tried a top-hat on the white line and it makes the crooked white line stand out better on a challenging shot, but that alone isn't enough.
2. Search around the previous fit instead of the histogram in each frame.
3. A trapezoid that adjusts (or bends the road model) for sharp curves, and a shorter viewing range in the curve.
4. If only one line is visible, estimate the others from the known width of the tape.
5. A Kalman filter or similar model instead of a simple weighted average.
