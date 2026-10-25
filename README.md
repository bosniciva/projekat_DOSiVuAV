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

[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

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

TODO: Add your text here!!!

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

TODO: Add your text here!!!

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

TODO: Add your text here!!!

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

TODO: Add your text here!!!

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

TODO: Add your text here!!!

