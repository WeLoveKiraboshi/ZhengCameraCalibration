import numpy as np
import os

from image import Image
from visualize import visualize_extrinsic
from optimize import CameraOptimization

####
# COVIS Lab1 python script written by Yuki Saito  (2021.Feb.)
# TEL : (+81)80-2161-0882
# MAIL: yusa19971015@keio.jp
# website: http://www.hvrl.ics.keio.ac.jp/saito_y/site/
# Keio University , School of Science for Open and Environment Systems,
#   the department of information and computer science 
# (Ecole central de Nantes, M1 JEMARO students) 
###



if __name__ == '__main__':
	n_imgs = 9
	# get list of image names
	#im_list = [file for file in os.listdir('./imgs') if file.endswith('.jpg')]
	im_list = ['img%d.jpg'%i for i in range(n_imgs)]

	# for each image, instantiate an Image object to calculate the Homography that map points from plane to image
	images = [Image(os.path.join('./imgs', im_file), 0.03, False) for im_file in im_list]

	# TODO: construct V to solve for b by stacking the output of im.construct_v() (Equation.(17))
	# image is class component defined in image.py
	V = np.empty((0, 6))
	for image in images:
		V = np.concatenate((V, image.construct_v()), axis=0)
	print('v shape = {}'.format(V.shape))

	# TODO: find b using the SVD trick
	u, s, vh = np.linalg.svd(V)
	b = vh[-1, :].reshape(6, 1)
	print('b shape = {}'.format(b.shape))
	print('norm(V @ b) =', np.linalg.norm(V @ b))  # check if the dot product between V and b is zero
	b11, b12, b22, b13, b23, b33 = b.tolist()
	print('b.shape: ', b.shape)
	print('b11: ', b11)
	print('b12: ', b12)
	print('b22: ', b22)
	print('b13: ', b13)
	print('b23: ', b23)
	print('b33: ', b33)

	# TODO: find components of intrinsic matrix from Equation.(12)
	v0 = (b12[0] * b13[0] - b11[0] * b23[0]) / (b11[0] * b22[0] - np.power(b12[0], 2))
	lamda = b33[0] - (
				np.power(b13[0], 2) + (b12[0] * b13[0] - b11[0] * b23[0]) / (b11[0] * b22[0] - np.power(b12[0], 2)) * (
					b12[0] * b13[0] - b11[0] * b23[0])) / b11[0]
	alpha = np.sqrt(lamda / b11[0])
	beta = np.sqrt((lamda * b11[0]) / (b11[0] * b22[0] - np.power(b12[0], 2)))
	c = - b12[0] * np.power(alpha, 2) * beta / lamda
	u0 = c * v0 / beta - b13[0] * np.power(alpha, 2) / lamda
	print('----\nCamera intrinsic parameters:')
	print('\talpha: ', alpha)
	print('\tbeta: ', beta)
	print('\tlamda: ', lamda)
	print('\tc: ', c)
	print('\tu0: ', u0)
	print('\tv0: ', v0)
	cam_intrinsic = np.array([
		[alpha, c, u0],
		[0, beta, v0],
		[0, 0, 1]
	])

	# get camera pose
	for im in images:
		R, t = im.find_extrinsic(cam_intrinsic)
		print('R = \n', R)
		print('t = ', t)

	sum_abs = 0
	sum_rmse = 0
	for im in images:
		abs, rmse = im.calc_reprojection_error(display=False)
		sum_abs = abs
		sum_rmse = rmse
		pass
	print("_______________________________________________")
	print('Average : ReproError  abs = {},  rmse = {}'.format(sum_abs / n_imgs, sum_rmse / n_imgs))
	print("_______________________________________________")



	optim = CameraOptimization(images)
	optim.refine_params()

	sum_abs = 0
	sum_rmse = 0
	for im in images:
		abs, rmse = im.calc_reprojection_error(display=True, save=True)
		sum_abs = abs
		sum_rmse = rmse
		pass
	print("_______________________________________________")
	print('Average : ReproError  abs = {},  rmse = {}'.format(sum_abs / n_imgs, sum_rmse / n_imgs))
	print("_______________________________________________")

	for im in images:
		R, t = im.find_extrinsic(cam_intrinsic)
		print('R = \n', R)
		print('t = ', t)

	#visualize_extrinsic(images)
	#visualize_extrinsic(images)



