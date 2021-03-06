import numpy as np

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as p3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import numpy as np

from matplotlib import rc
from matplotlib.ticker import MultipleLocator

def to_homogeneous(A):
    """Convert a stack of inhomogeneous vectors to a homogeneous
       representation.
    """
    A = np.atleast_2d(A)

    N = A.shape[0]
    A_hom = np.hstack((A, np.ones((N,1))))

    return A_hom


def make_axis_publishable(ax, major_x, major_y, major_z):
    # [t.set_va('center') for t in ax.get_yticklabels()]
    # [t.set_ha('left') for t in ax.get_yticklabels()]
    # [t.set_va('center') for t in ax.get_xticklabels()]
    # [t.set_ha('right') for t in ax.get_xticklabels()]
    # [t.set_va('center') for t in ax.get_zticklabels()]
    # [t.set_ha('left') for t in ax.get_zticklabels()]

    ax.grid(False)
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_edgecolor('black')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.xaxis._axinfo['tick']['inward_factor'] = 0
    ax.xaxis._axinfo['tick']['outward_factor'] = 0.4
    ax.yaxis._axinfo['tick']['inward_factor'] = 0
    ax.yaxis._axinfo['tick']['outward_factor'] = 0.4
    ax.zaxis._axinfo['tick']['inward_factor'] = 0
    ax.zaxis._axinfo['tick']['outward_factor'] = 0.4
    ax.zaxis._axinfo['tick']['outward_factor'] = 0.4

    ax.xaxis.set_major_locator(MultipleLocator(major_x))
    ax.yaxis.set_major_locator(MultipleLocator(major_y))
    ax.zaxis.set_major_locator(MultipleLocator(major_z))

def visualize_extrinsic(images):
	fig = plt.figure()
	ax = fig.add_subplot('111', projection='3d')
	make_axis_publishable(ax, 10, 10, 10)

	ax.set_title('World-Centric Extrinsics')
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')

	ax.set_xlim(-5, 5)
	ax.set_ylim(-5, 5)
	ax.set_zlim(-4, 4)

	# From StackOverflow: https://stackoverflow.com/questions/39408794/python-3d-pyramid
	v = np.array([[-0.5, -0.5, 1], [0.5, -0.5, 1], [0.5, 0.5, 1], [-0.5, 0.5, 1], [0, 0, 0]])
	v = to_homogeneous(v)

	n = 256
	x = np.linspace(-4, 4, n)
	y = np.linspace(-4, 4, n)
	X, Y = np.meshgrid(x, y)
	Z = X * 0
	ax.plot_surface(X, Y, Z, cmap="plasma_r")

	for im in images:
		E = np.eye(4)
		E[0:3, 0:3] = im.R
		E[0:3, 3] = im.T.reshape(3)

		E_inv =np.linalg.inv(E)
		E_inv = E_inv[:3]
		v_new = np.dot(v, E_inv.T)
		#v_new = np.dot(E[0:3, :], v.T).T
		#v_new = v_new.T
		print(E)


		verts = [[v_new[0], v_new[1], v_new[4]], [v_new[0], v_new[3], v_new[4]],
				 [v_new[2], v_new[1], v_new[4]], [v_new[2], v_new[3], v_new[4]],
				 [v_new[0], v_new[1], v_new[2], v_new[3]]]

		ax.add_collection3d(Poly3DCollection(verts, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))

	ax.invert_xaxis()

	plt.show()
