import numpy as np
import cv2

'''
def _convert_R_vec_to_mat(R_vec):
    """
    Function to convert R vector computed using Rodriguez formula back to a mtrix
    R_vec = [wx, wy, wz]
    :return:
    """
    R_mat = np.array([[R_vec[0], R_vec[1], R_vec[2]],[R_vec[3], R_vec[4], R_vec[5]],[R_vec[6], R_vec[7], R_vec[8]]]).reshape(3,3)
    return R_mat


def _convert_R_mat_to_vec(R_mat):
    """
    Use rodiriguez formula to convert R matrix to R vector with 3 DoF
    :return 3*3mat:
    """
    R_vec = np.array([R_mat[0][0], R_mat[0][1], R_mat[0][2], R_mat[1][0], R_mat[1][1], R_mat[1][2], R_mat[2][0], R_mat[2][1], R_mat[2][2]]).reshape(9, )

    return R_vec

'''
def _convert_R_vec_to_mat(R_vec):
    """
    Function to convert R vector computed using Rodriguez formula back to a mtrix
    R_vec = [wx, wy, wz]
    :return:
    """

    phi = np.linalg.norm(R_vec)
    Wx = np.zeros((3,3))

    Wx[0][1] = -1*R_vec[2]
    Wx[0][2] = R_vec[1]

    Wx[1][0] = R_vec[2]
    Wx[1][2] = -1*R_vec[0]

    Wx[2][0] = -1*R_vec[1]
    Wx[2][1] = R_vec[0]

    R_mat = np.eye(3) + (np.sin(phi)/phi) * Wx + ((1-np.cos(phi))/phi**2)*np.dot(Wx, Wx)

    # print("-------- R vec to mat-----------")
    # print(cv2.Rodrigues(R_vec[:, np.newaxis])[0])
    # print(R_mat)
    # print("-------- R vec to mat-----------")

    return R_mat



def _convert_R_mat_to_vec(R_mat):
    """
    Use rodiriguez formula to convert R matrix to R vector with 3 DoF
    :return:
    """
    phi = np.arccos((np.trace(R_mat) - 1) / 2)

    R_vec = np.array([R_mat[2][1] - R_mat[1][2], R_mat[0][2] - R_mat[2][0], R_mat[1][0] - R_mat[0][1]])

    R_vec = R_vec * (phi / (2 * np.sin(phi)))

    # print("-------- R mat to vec-----------")
    # print(cv2.Rodrigues(R_mat)[0])
    # print("-------- R mat to vec-----------")

    return R_vec



