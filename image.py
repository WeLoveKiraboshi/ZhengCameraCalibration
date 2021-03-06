'''
Define image class which read the image, extract chessboard corners find the homography
'''
import cv2
import numpy as np

'''
def normalize_trans(points):
	"""TODO: Compute a transformation which translates and scale the inputted points such that
		their center is at the origin and their average distance to the origin is sqrt(2) using Equation.(21)

	Args:
		points (np.ndarray): points to normalize, shape (n, 2)
	Return:
		np.ndarray: similarity transformation for normalizing these points, shape (3, 3)
	"""
	u = np.mean(points[:, 0]) #(points[0][0] + points[points.shape[0]-1][0]) / 2
	v = np.mean(points[:, 1]) #(points[0][1] + points[points.shape[0] - 1][1]) / 2
	#print('u = {}, v = {}'.format(u, v))
	origin = np.asarray([u, v])
	#origin of points is initially defined as 1st point
	mean_dis = np.mean(np.sqrt(np.sum(np.power(points - origin, 2), 1)))
	s = np.sqrt(2) / mean_dis

	Tp = np.zeros((3,3), dtype=np.float32)
	Tp[0][0] = s
	Tp[1][1] = s
	Tp[2][2] = 1
	Tp[0][2] = -s * u
	Tp[1][2] = -s * v

	#check

	return Tp
'''
def normalize_trans(points):

	center = np.mean(points, axis=0) # shape (2,)
	dist = np.linalg.norm(points - center, axis=1) # distance from each point to origin, shape (n,)
	s = np.sqrt(2.0) / dist.mean()
	return np.array([
	[s, 0, -s * center[0]],
	[0, s, -s * center[1]],
	[0, 0, 1]
	])


def homogenize(points):
	"""Convert points to homogeneous coordinate

	Args:
		points (np.ndarray): shape (n, 2)
	Return:
		np.ndarray: points in homogeneous coordinate (with 1 padded), shape (n, 3)
	"""
	re = np.ones((points.shape[0], 3))  # shape (n, 3)
	re[:, :2] = points
	return re

id = 1

class Image:
	"""Provide operations on image necessary for calibration"""
	refine_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

	def __init__(self, impath, square_size=0.03, debug=False):
		"""
		Args:
			impath (str): path to image file
			square_size (float): size in meter of a square on the chessboard
		"""
		self.im = cv2.imread(impath)
		self.square_size = square_size
		self.debug = debug
		self.rows = 8  # number of rows in the grid pattern to look for on the chessboard
		self.cols = 6  # number of columns in the grid pattern to look for on the chessboard
		self.im_pts = self.locate_landmark()  # pixel coordinate of chessboard's corners
		self.plane_pts = self.get_landmark_world_coordinate()  # world coordinate of chessboard's corners
		self.H = self.find_homography()
		global id
		self.id = id
		id += 1

	def locate_landmark(self, draw_corners=False):
		"""Identify corners on the chessboard such that they form a grid defined by inputted parameters

		Args:
			draw_corners (bool): to draw corners or not
		Return:
			np.ndarray: pixel coordinate of chessboard's corners, shape (self.rows * self.cols, 2)
		"""
		# convert color image to gray scale
		gray = cv2.cvtColor(self.im, cv2.COLOR_BGR2GRAY)
		# find the chessboard corners
		ret, corners = cv2.findChessboardCorners(gray, (self.rows, self.cols), None)
		# if found, refine these corners' pixel coordinate & store them
		if ret:
			corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), Image.refine_criteria)

			# self.im_pts = corners.squeeze()
			# print('self.im_pts.shape: ', self.im_pts.shape)
			# for i in range(self.im_pts.shape[0]):
			# 	print('pts [{}]: '.format(i), self.im_pts[i])

			if draw_corners:
				cv2.drawChessboardCorners(self.im, (self.rows, self.cols), corners, ret)
				cv2.imshow('im', self.im)
				cv2.waitKey(0)
				cv2.destroyWindow('im')

		return corners.squeeze()



	def get_landmark_world_coordinate(self):
		#print(self.im_pts.shape)
		"""TODO: Compute 3D coordinate for each chessboard's corner. Assumption:

				* world origin is located at the 1st corner
				* x-axis is from corner 0 to corner 1 till corner 7,
				* y-axis is from corner 0 to corner 8 till corner 40,
				* distance between 2 adjacent corners is self.square_size
		Returns:
			np.ndarray: 3D coordinate of chessboard's corners, shape (self.rows * self.cols, 2)
		"""
		coordinateslist = [] #= np.zeros((self.rows * self.cols, 2))
		for i in range(self.im_pts.shape[0]):
			col = int(i % self.rows)
			row = int(i / self.rows)
			x = col * self.square_size
			y = row * self.square_size
			coordinateslist.append([x, y])
			#print('x = {}, y = {}'.format(x, y))


		return np.asarray(coordinateslist)




	def find_homography(self):
		"""TODO: Find the homography H that maps plane_pts to im_pts using Equation.(8)

		Return:
			np.ndarray: homography, shape (3, 3)
		"""
		# get the normalize transformation
		T_norm_im = normalize_trans(self.im_pts)
		T_norm_plane = normalize_trans(self.plane_pts)

		# normalize image points and plane points
		norm_im_pts = (T_norm_im @ homogenize(self.im_pts).T).T  # shape (n, 3)
		norm_plane_pts = (T_norm_plane @ homogenize(self.plane_pts).T).T  # shape (n, 3)

		# TODO: construct linear equation to find normalized H using norm_im_pts and norm_plane_pts
		Q_list = []

		for i in range(self.rows * self.cols):
			Q_list.append(np.concatenate([norm_plane_pts[i].T, np.zeros((3)), -norm_im_pts[i][0]*norm_plane_pts[i].T], 0))
			Q_list.append(np.concatenate([np.zeros((3)), norm_plane_pts[i].T, -norm_im_pts[i][1]*norm_plane_pts[i].T], 0))
		Q = np.asarray(Q_list)


		# TODO: find normalized H as the singular vector Q associated with the smallest singular value
		u, s, vh = np.linalg.svd(Q)
		H_norm = vh[-1, :].reshape(3, 3)
		#print('check for find Homography(normalized) = {}'.format(np.linalg.norm(Q @ H_norm.reshape(-1, 1))))
		#H_norm = np.asarray([H_norm_[0:3], H_norm_[3:6], H_norm_[6:9]])


		# TODO: de-normalize H_norm to get H
		H = np.linalg.inv(T_norm_im) @ H_norm @ T_norm_plane


		return H

	def construct_v(self):
		"""
			Find the left-hand side of Equation.(8) in Zhang's paper

			Return:
			np.ndarray: shape (2, 6)
		"""

		h11 = self.H[0][0]
		h12 = self.H[0][1]
		h13 = self.H[0][2]
		h21 = self.H[1][0]
		h22 = self.H[1][1]
		h23 = self.H[1][2]
		h31 = self.H[2][0]
		h32 = self.H[2][1]
		h33 = self.H[2][2]
		v12 = np.array([h11 * h12, h11 * h22 + h21 * h12, h21 * h22, h31 * h12 + h11 * h32, h31 * h22 + h21 * h32, h31 * h32]).reshape(1, -1)
		v11 = np.array([h11 * h11, h11 * h21 + h21 * h11, h21 * h21, h31 * h11 + h11 * h31, h31 * h21 + h21 * h31, h31 * h31]).reshape(1, -1)
		v22 = np.array([h12 * h12, h12 * h22 + h22 * h12, h22 * h22, h32 * h12 + h12 * h32, h32 * h22 + h22 * h32, h32 * h32]).reshape(1, -1)
		V = np.vstack([v12, v11 - v22])

		return V

	def find_extrinsic(self, K):
		"""TODO: Find camera pose w.r.t the world frame defined by the chessboard using the homography and camera intrinsic
			matrix using Equation.(18)

		Arg:
			K (np.ndarray): camera intrinsic matrix, shape (3, 3)
		Returns:
			tuple[np.ndarray]: Rotation matrix (R) - shape (3, 3), translation vector (t) - shape (3,)
		"""
		h1 = self.H[:, 0]
		h2 = self.H[:, 1]
		h3 = self.H[:, 2]
		invK = np.linalg.inv(K)
		v = 1 / ((np.linalg.norm(invK @ h1)+np.linalg.norm(invK @ h2))/2)
		r1 = v * invK @ h1
		r2 = v * invK @ h2
		r3 = np.cross(r1, r2)
		R = np.zeros((3, 3))
		R[:, 0] = r1.reshape(3)
		R[:, 1] = r2.reshape(3)
		R[:, 2] = r3.reshape(3)
		#u, s, vh = np.linalg.svd(R)
		#R = u @ vh.T
		#print('r1 = {}, r2 = {}, r3 = {}'.format(r1.shape, r2.shape, r3.shape))
		t = v * invK @ h3
		self.T = t.reshape(-1, 1)
		self.R = R
		self.K = K
		return self.R, self.T

	def calc_reprojection_error(self, display=False):
		r1r2t = np.zeros((3, 3))
		r1r2t[:, 0] = self.R[:, 0]
		r1r2t[:, 1] = self.R[:, 1]
		r1r2t[:, 2] = self.T.reshape(3)
		diff = np.zeros((self.rows * self.cols, 3))

		for i in range(self.rows * self.cols):
			world_coordinate_pos = np.array([self.plane_pts[i][0],self.plane_pts[i][1],1])
			image_coordinate_pos_original = np.array([self.im_pts[i][0], self.im_pts[i][1], 1])
			image_coordinate_pos = r1r2t@world_coordinate_pos
			image_coordinate_pos = self.K @ image_coordinate_pos
			image_coordinate_pos = image_coordinate_pos/image_coordinate_pos[2]
			diff[i, :] = image_coordinate_pos - image_coordinate_pos_original
			if display:
				cv2.circle(self.im, (int(image_coordinate_pos_original[0]), int(image_coordinate_pos_original[1])), 3, (255, 0, 0), -1)  # B
				cv2.circle(self.im, (int(image_coordinate_pos[0]), int(image_coordinate_pos[1])), 5, (0, 0, 255), 1) # R
		ABS = np.mean(np.abs(diff))
		RMSE = np.mean(np.sqrt(np.power(diff, 2)))
		print('Image[{}] : ReproError  abs = {},  rmse = {}'.format(self.id, ABS, RMSE))

		if display:
			cv2.imshow('im', self.im)
			cv2.waitKey(1000)
			cv2.destroyWindow('im')
			cv2.imwrite("reprojected_{}.png".format(self.id), self.im)

		return ABS, RMSE


	def construct_v_answer(self):
		def v_ij(H, i, j):
			hi = H[:, i] # shape (3,)
			hj = H[:, j] # shape (3,)
			return np.array([hi[0] * hj[0], hi[0] * hj[1] + hi[1] * hj[0], hi[1] * hj[1],
				hi[2] * hj[0] + hi[0] * hj[2], hi[2] * hj[1] + hi[1] * hj[2], hi[2] * hj[2]]).reshape(1, 6)

		row_0 = v_ij(self.H, 0, 1) # shape (1, 6)
		row_1 = v_ij(self.H, 0, 0) - v_ij(self.H, 1, 1) # shape (1, 6)
		return np.vstack((row_0, row_1))

