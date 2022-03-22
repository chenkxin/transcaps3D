import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from pytransform3d.rotations import *

ax = plot_basis(R=np.eye(3), ax_s=2)
axis = 1
angle = np.pi / 2

p = np.array([1.0, 1.0, 1.0])
euler = [0, 0, 0]
euler[axis] = angle
R = matrix_from_euler_xyz(euler)
plot_basis(ax, R, p)


plt.show()
