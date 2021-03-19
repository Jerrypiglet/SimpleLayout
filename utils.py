import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils_SL_vis import vis_cube_plt, set_axes_equal, vis_axis, vis_axis_xyz

class SimpleScene():

    def __init__(self, cam_dict):
        self.cam_params = self.form_camera(cam_dict)
        self.K = self.cam_params['K']
        self.T = np.array([[0., 0., 1.], [0., -1., 0.], [1., 0., 0.]]) # cam coords -> project cam coords
        self.H, self.W = cam_dict['height'], cam_dict['width']
        self.bbox = None
        self.xyz_min = np.zeros(3,) + np.inf
        self.xyz_max = np.zeros(3,) - np.inf
    
    def form_camera(self, cam_dict):
        '''
        axes: np.array([[-1., 0., -1.], [0., 1., 0.], [1., 0., -1]])
        fov_x, fov_y: in degrees

        '''
        origin = cam_dict['origin'].reshape((3,1))
        if cam_dict['cam_axes'] is None:
            lookat_pnt = cam_dict['lookat'].reshape((3,1))
            toward = cam_dict['toward'].reshape((3,1))  # x-axis
            toward /= np.linalg.norm(toward)
            up = cam_dict[6:9]  # y-axis
            up /= np.linalg.norm(up)
            right = np.cross(toward, up)  # z-axis
            right /= np.linalg.norm(right)
        else:
            toward, up, right = np.split(cam_dict['cam_axes'].T, 3, 1) # x_cam, y_cam, z_cam
            toward = toward / np.linalg.norm(toward)
            up = up / np.linalg.norm(up)
            right = right / np.linalg.norm(right)
            assert abs(np.dot(toward.flatten(), up.flatten())) < 1e-5
            assert abs(np.dot(toward.flatten(), right.flatten())) < 1e-5
            assert abs(np.dot(right.flatten(), up.flatten())) < 1e-5
            cam_axes = np.hstack([toward, up, right]).T
            R = cam_axes.T  # columns respectively corresponds to toward, up, right vectors.
            t = origin

        fov_x = cam_dict['fov_x'] / 180. * np.pi
        fov_y = cam_dict['fov_y'] / 180. * np.pi
        width = cam_dict['width']
        height = cam_dict['height']

        f_x = width / (2 * np.tan(fov_x/2.))
        f_y = height / (2 * np.tan(fov_y/2.))

        K = np.array([[f_x, 0., width/2.-1.], [0., f_y, height/2.-1.], [0., 0., 1.]])

        cam_params = {'K': K, 'R': R, 'origin': origin, 'cam_axes': cam_axes}
        return cam_params

    def transform(self, x):
        assert len(x.shape)==2 and x.shape[1]==3
        x = x.reshape((-1, 3))
        return (self.cam_params['cam_axes'] @ (x.T - self.cam_params['origin'])).T

    def transform_and_proj(self, x):
        x_c = self.transform(x)
        x_c_T = (self.T @ (x_c.T)).T
        x_c_proj = (self.K @ x_c_T.T).T
        x_c_proj = x_c_proj[:, :2] / (x_c_proj[:, 2:3]+1e-6)
        invalid_ids = np.where(x_c_T[:, 2]<=0)[0].tolist()
        return x_c_proj, invalid_ids
        
    def vis_3d(self, bbox):
        fig = plt.figure(figsize=(5, 5))
        ax_3d = fig.add_subplot(111, projection='3d')
        ax_3d = fig.gca(projection='3d')
        ax_3d.set_proj_type('ortho')
        ax_3d.set_aspect("auto")

        [cam_xaxis, cam_yaxis, cam_zaxis] = np.split(self.cam_params['cam_axes'].T, 3, 1)
        vis_cube_plt(ax_3d, bbox, linewidth=2, if_vertex_idx_text=True)
        vis_axis_xyz(ax_3d, cam_xaxis.flatten(), cam_yaxis.flatten(), cam_zaxis.flatten(), self.cam_params['origin'].flatten(), suffix='_c', make_bold=[0])
        vis_axis(ax_3d, make_bold=[1])

        self.xyz_min = np.minimum(self.xyz_min, np.amin(bbox, 0))
        self.xyz_max = np.maximum(self.xyz_max, np.amax(bbox, 0))
        self.xyz_min = np.minimum(self.xyz_min, self.cam_params['origin'].reshape((3,))-1.)
        self.xyz_max = np.maximum(self.xyz_max, self.cam_params['origin'].reshape((3,))+1.)
        ax_3d.view_init(elev=121, azim=-111)
        ax_3d.set_box_aspect([1,1,1])
        new_limits = np.hstack([self.xyz_min.reshape((3, 1)), self.xyz_max.reshape((3, 1))])
        set_axes_equal(ax_3d, limits=new_limits) # IMPORTANT - this is also required

        return ax_3d

    def vis_2d_bbox_proj(self, bbox, if_show=True):
        bbox_proj, bbox_invalid_ids = self.transform_and_proj(bbox)
        fig = plt.figure()
        for idx, idx_list in enumerate([[0, 1, 2, 3, 0], [4, 5, 6, 7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]):
            for i in range(len(idx_list)-1):
                x1 = bbox_proj[idx_list[i]]
                x2 = bbox_proj[idx_list[i+1]]
                plt.plot([x1[0], x2[0]], [x1[1], x2[1]], color='k', linewidth=2, linestyle='--')
        for idx, x2d in enumerate(bbox_proj):
            plt.text(x2d[0], x2d[1], str(idx))
        plt.xlim([0., self.W-1])
        plt.ylim([self.H-1, 0])
        if if_show:
            plt.show()
        else:
            return plt.gca()




