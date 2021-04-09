import numpy as np
import os

from image import Image
import os
import numpy as np
import cv2
from scipy import optimize
from utils import _convert_R_mat_to_vec, _convert_R_vec_to_mat



class CameraOptimization:
    def __init__(self, images):
        self.images = images
        self.num_imgs = len(images)
        self.K = self.images[0].K
        self.Rt = []
        for im in self.images:
            Rt = np.concatenate([im.R, im.T], 1) #Rt is the mat(3, 4)
            self.Rt.append(Rt)


    @staticmethod
    def compute_residuals(x, images):
        """
        :param x: np array - variables to optimize
        0:5 (K -> u0, v0, alpha, beta, gamma)
        Then 3 entries of R ,followed by 3 entries for t for each img
        If radial dist is True, then followed by k1, k2 for each img
        x = [alpha, gamma, u0, beta, v0,  r1_1, r2_1, r3_1, t1_1, t2_1, t3_1, r1_2, r2_2, r3_2, t1_2, t2_2, t3_2......]
        :param img_hc: List of lists of all actual img pts
        :param world_hc: np array of rows of world pts
        :param radial_dist: bool, whether to take radial distortion into account
        :return:
        """
        num_corners = images[0].rows * images[0].cols
        num_imgs = len(images)
        K = np.zeros((3, 3))
        #  Build K
        K[0][0] = x[0]
        K[0][1] = x[1]
        K[0][2] = x[2]
        K[1][1] = x[3]
        K[1][2] = x[4]
        K[2][2] = 1

        residual_sum = 0
        img_hc_list = []
        proj_crd_list = []

        for i in range(num_imgs):
            Rt_vec = x[5 + i * 6: 5 + (i + 1) * 6]
            R = _convert_R_vec_to_mat(Rt_vec[0:3])
            Rt = np.hstack((R, Rt_vec[3:].reshape(3, 1)))
            P = np.matmul(K, Rt)  # shape = 3, 4
            # Compute projections per image, per corner
            # world_hc = <num_corners> rows of x, y, z, w
            # image = list of nd arrays. Each nd array has rows of [x, y, z]

            img_hc = np.concatenate([images[i].im_pts, np.ones((num_corners, 1))], 1).T  # 3, n_pts
            world_hc = np.concatenate([images[i].plane_pts, np.zeros((num_corners, 1)), np.ones((num_corners, 1))],
                                      1).T  # 4, n_pts

            proj_crd = np.matmul(P, world_hc)  # Proj_crd shape = # 3, n_pts

            proj_crd = proj_crd / proj_crd[2, :]  # normalizing last crd


            #save result and make mat which contains all info
            proj_crd_list.append(proj_crd)
            img_hc_list.append(img_hc)

            #diff = img_hc - proj_crd
            #ABS = np.mean(np.abs(diff))
            #RMSE = np.sqrt(np.mean(np.power(diff, 2)))
            #print('Image[1] : ReproError  abs = {},  rmse = {}'.format(ABS, RMSE))

        # compute residual
        img_hc = np.array(img_hc_list)
        proj_crd = np.array(proj_crd_list)
        residual = img_hc.ravel() - proj_crd.ravel()  # 3, n_pts

        return residual

    def refine_params(self):
        print("---------------------------------------")
        print("refine Camera Extrinsic and Intrinsic parameters")
        print('this may takes several seconds for computation')
        print("---------------------------------------")

        num_params = 5 + self.num_imgs * 6  # 5 params for K and (3 DOF for R and 3 DOF for t)for each image

        x_init = np.zeros(num_params)

        #####  Initialize x_init with K values ######
        # K = [[alpha, gamma, u0], [0, beta, v0], [0, 0, 1]]
        x_init[0] = self.K[0][0]
        x_init[1] = self.K[0][1]
        x_init[2] = self.K[0][2]
        x_init[3] = self.K[1][1]
        x_init[4] = self.K[1][2]

        for i in range(self.num_imgs):
            r_vec = _convert_R_mat_to_vec(self.Rt[i][:, 0:3])
            x_init[5 + i * 6: 5 + (i + 1) * 6] = np.hstack(
                (r_vec, self.Rt[i][:, -1]))  # assign R vector and t of each image to init values

        #cons = (
        #    {'type': 'eq', lambda: x: np.array([x[1] - 0])}
        #)

        sol = optimize.least_squares(CameraOptimization.compute_residuals, x_init, args=([self.images]),
                                    method='lm',
                                     xtol=1e-15, ftol=1e-15)

        ### Build K, R, t from solution

        # K = [[alpha, gamma, u0], [0, beta, v0], [0, 0, 1]]
        self.K_final = np.zeros_like(self.K)
        self.K_final[0][0] = sol.x[0]
        self.K_final[0][1] = sol.x[1]
        self.K_final[0][2] = sol.x[2]
        self.K_final[1][1] = sol.x[3]
        self.K_final[1][2] = sol.x[4]
        self.K_final[2][2] = 1

        self.Rt_final = [[] for _ in range(self.num_imgs)]

        for i in range(self.num_imgs):
            Rt_i = sol.x[5 + i * 6: 5 + (i + 1) * 6]
            R_i = _convert_R_vec_to_mat(Rt_i[0:3])
            self.Rt_final[i] = np.hstack((R_i, Rt_i[3:].reshape(3, 1)))
            self.images[i].K = self.K_final
            self.images[i].R = R_i
            self.images[i].T = Rt_i[3:].reshape(3, 1)

        print("-------------------------------------")

        print("Inital_K: {}".format(self.K))
        print("final_K: {}".format(self.K_final))

        print("------")

        print("Inital_Rt: {}".format(self.Rt[0]))
        print("final_Rt: {}".format(self.Rt_final[0]))

        print("-------------------------------------")

        print(" Optimize ------------  Done!")

        print("-----------------------------------------------------------------------------------------")



