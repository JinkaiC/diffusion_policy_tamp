import os
import numpy as np

from diffusion_policy.dataset.utils.constants import *
from diffusion_policy.dataset.utils.transformation import xyz_rot_to_mat, mat_to_xyz_rot

class Projector:
    def __init__(self):
        self.cam_to_base = EXTRINSIC_MATRIX

    def project_tcp_to_camera_coord(
        self, tcp, rotation_rep="quaternion", rotation_rep_convention=None
    ):
        return mat_to_xyz_rot(
            np.linalg.inv(self.cam_to_base)
            @ xyz_rot_to_mat(
                tcp,
                rotation_rep=rotation_rep,
                rotation_rep_convention=rotation_rep_convention,
            ),
            rotation_rep=rotation_rep,
            rotation_rep_convention=rotation_rep_convention,
        )

    def project_tcp_to_base_coord(
        self, tcp, rotation_rep="quaternion", rotation_rep_convention=None
    ):
        return mat_to_xyz_rot(
            self.cam_to_base
            @ xyz_rot_to_mat(
                tcp,
                rotation_rep=rotation_rep,
                rotation_rep_convention=rotation_rep_convention,
            ),
            rotation_rep=rotation_rep,
            rotation_rep_convention=rotation_rep_convention,
        )
