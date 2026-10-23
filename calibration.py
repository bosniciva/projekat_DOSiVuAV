import glob
import cv2
import numpy as np

criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)

board_sizes = [(9, 6), (8, 6), (9, 5), (7, 6), (8, 5), (6, 6), (9, 4)]

objectPointsArray = []
imgPointsArray = []

for path in sorted(glob.glob('camera_cal/calibration*.jpg')):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners, size = False, None, None
    for cols, rows in board_sizes:
        ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
        if ret:
            size = (cols, rows)
            break

    if ret:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objectPoints = np.zeros((size[0] * size[1], 3), np.float32)
        objectPoints[:, :2] = np.mgrid[0:size[0], 0:size[1]].T.reshape(-1, 2)
        objectPointsArray.append(objectPoints)
        imgPointsArray.append(corners)

        cv2.drawChessboardCorners(img, size, corners, ret)
        print(f"Pronađeno {size[0]}x{size[1]} uglova: {path}")
    else:
        print("Uglovi nisu pronađeni:", path)

    cv2.imshow('chess board', img)
    cv2.waitKey(20)

cv2.destroyAllWindows()

if len(objectPointsArray) < 3:
    raise RuntimeError(f"Kalibracija nije moguća: pronađeno samo {len(objectPointsArray)} validnih slika (potrebno bar 3).")

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPointsArray, imgPointsArray, gray.shape[::-1], None, None)
np.savez('calib.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

error = 0

for i in range(len(objectPointsArray)):
    imgPoints, _ = cv2.projectPoints(objectPointsArray[i], rvecs[i], tvecs[i], mtx, dist)
    error += cv2.norm(imgPointsArray[i], imgPoints, cv2.NORM_L2) / len(imgPoints)

print("Total error:", error / len(objectPointsArray))

img = cv2.imread('camera_cal/calibration1.jpg')
undist = cv2.undistort(img, mtx, dist, None, mtx)
cv2.imwrite('docs/undistort_output.png', np.hstack([img, undist]))