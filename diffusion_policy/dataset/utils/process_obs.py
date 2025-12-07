"""
Functions for processing robot sensor data.

This module provides utilities for:
- Transforming robot sensor data between coordinate frames (base, camera, hand)
- Generating and filtering point clouds from RGB-D camera images
- Applying workspace boundaries to filter point cloud data
- Converting end-effector poses between reference frames

"""

import numpy as np
from diffusion_policy.dataset.utils.projector import Projector

# Initialize global objects
projector = Projector()

def convert_tcp_data_to_camera(data: np.ndarray) -> np.ndarray:
    """
    Convert data from TCP to camera coordinate system.

    Args:
        data: TCP data with position and orientation

    Returns:
        Converted data in camera coordinate system
    """
    cam_tcp = projector.project_tcp_to_camera_coord(data[:7])
    return np.concatenate([cam_tcp, data[7:]])