# %reload_ext autoreload
# %autoreload 2
# from IPython.core.display import display, HTML
# display(HTML("<style>.container { width:90% !important; }</style>"))

# %matplotlib widget
# import matplotlib.pyplot as plt
from utils_SL import SimpleScene
from utils_SL_bbox import get_corners_of_bb3d
import sys
sys.path.insert(0, '/home/ruizhu/Documents/Projects/semanticInverse/train')

import numpy as np
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

basis = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
coeffs = [0.5, 0.5, 0.5] # half length
centroid = np.array([coeffs[0], coeffs[1], coeffs[2]]).reshape((1, 3))

bbox = get_corners_of_bb3d(basis, coeffs, centroid)

cam_dict = {'origin': np.array([0.5, 0.5, 0.5]), 'cam_axes': np.array([[-1., 0., -1.], [0., 1., 0.], [1., 0., -1]]), 'fov_x': 90., 'fov_y': 90., 'width': 240, 'height': 320}
scene = SimpleScene(cam_dict)
# x = np.array([0., 1., 0.]).reshape((1, 3))
# cam.transform_and_proj(x)

# ax_3d = scene.vis_3d(bbox)
# ax_2d = scene.vis_2d_bbox_proj_simple(bbox, if_show=False)
scene.vis_2d_bbox_proj(bbox, if_show=False)
