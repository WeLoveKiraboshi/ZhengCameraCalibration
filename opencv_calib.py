import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

square_size = 3      # [cm]
pattern_size = (6, 8)  #

reference_img = 9 #

pattern_points = np.zeros( (np.prod(pattern_size), 3), np.float32 ) #チェスボード（X,Y,Z）座標の指定 (Z=0)
pattern_points[:,:2] = np.indices(pattern_size).T.reshape(-1, 2)
pattern_points *= square_size
objpoints = []
imgpoints = []



def opencv_calib():
    #im_list = [file for file in os.listdir('./imgs') if file.endswith('.jpg')]
    im_list = ['img%d.jpg' % i for i in range(reference_img)]
    for filepath in im_list:
        img = cv2.imread(os.path.join('./imgs', filepath))
        height = img.shape[0]
        width = img.shape[1]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corner = cv2.findChessboardCorners(gray, pattern_size)
        if ret == True:
            print("detected coner!")
            print(str(len(objpoints) + 1) + "/" + str(reference_img))
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
            cv2.cornerSubPix(gray, corner, (5, 5), (-1, -1), term)
            imgpoints.append(corner.reshape(-1, 2))  # appendメソッド：リストの最後に因数のオブジェクトを追加
            objpoints.append(pattern_points)

        cv2.imshow('image', img)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    print("calculating camera parameter...")
    # intrinsic parameter
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # save result
    #np.save("mtx", mtx)
    #np.save("dist", dist.ravel())  # distortion param
    print("RMS = ", ret)
    print("mtx = \n", mtx)
    print("dist = ", dist.ravel())
    print("rvecs = ", rvecs)
    print("tvecs = ", tvecs)


if __name__ == '__main__':
    opencv_calib()

