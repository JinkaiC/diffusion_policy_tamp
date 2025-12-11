# """
# Evaluation Agent.
# """

import time
import numpy as np
import cv2
import rospy
from collections import deque
from diffusion_policy.device.sim_camera import SimCamera
from diffusion_policy.device.franka_sim import FrankaGenesisEnvWrapper
from termcolor import cprint
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import JointState
import os
from diffusion_policy.dataset.utils.process_obs import (
    convert_tcp_data_to_camera,
)
from diffusion_policy.model.common.rotation_transformer import RotationTransformer

quat2rot6d_transformer = RotationTransformer(from_rep='quaternion', to_rep="rotation_6d")

def transform_ee_pose_frame(ee_pose: np.ndarray, frame: str) -> np.ndarray:
    """Transform end effector pose to the specified reference frame."""
    if frame == "camera":
        return convert_tcp_data_to_camera(ee_pose)
    elif frame == "base":
        return ee_pose
    else:
        raise ValueError(f"Unsupported frame type: {frame}. Use 'camera' or 'base'.")


class Agent:
    def __init__(self, obs_num, gripper, frame="base", **kwargs):
        self.obs_num = obs_num  # Number of frames to buffer
        self.frame = frame
        
        # Initialize deques for buffering
        self.rgb_buffer = deque(maxlen=obs_num)
        self.pose_buffer = deque(maxlen=obs_num)
        
        # Initialize camera and robot for simulation
        self.camera = SimCamera()
        self.arm = FrankaGenesisEnvWrapper(control_mode="joint", gripper="panda" if gripper else None, gripper_init_state="open")

        self.gripper = gripper   
        
        # Warmup camera and fill buffer
        self._warmup_camera()
        self._fill_initial_buffer()

    def _warmup_camera(self):
        """Get initial frames to stabilize camera"""
        for _ in range(30):
            self.camera.get_rgbd_image()

    def _fill_initial_buffer(self):
        """Pre-fill buffer with obs_num frames"""
        for _ in range(self.obs_num):
            self._add_single_observation()
    
    def reset_buffer(self):
        """Reset observation buffers and refill with current observations"""
        self.rgb_buffer.clear()
        self.pose_buffer.clear()
        self._fill_initial_buffer()

    def _add_single_observation(self):
        """Capture and process one frame+pose"""
        # Get raw camera data
        colors, _ = self.camera.get_rgbd_image()
        
        # Process image
        processed_img = self._process_image(colors)
        # from matplotlib import pyplot as plt
        # plt.imshow(processed_img)
        # plt.axis("off")
        # plt.show()
        # Get robot state
        arm_ee_pose = self.arm.get_tcp_position()
        # 7-dim: x, y, z, qw, qx, qy, qz
        transformed_ee_pose = transform_ee_pose_frame(arm_ee_pose, self.frame)
        xyz, quat = transformed_ee_pose[:3], transformed_ee_pose[3:7]
        rot6d = quat2rot6d_transformer.forward(np.array(quat))
        # 9-dim: x, y, z, rot6d
        pose = np.concatenate([xyz, rot6d])
        # from ipdb import set_trace; set_trace()
        # if self.sim: #! Check SIm!!!!
            # pose = self.arm.get_tcp_position()
            # quat = pose[3:]
            # rot = R.from_quat(quat)
            # rotation_matrix = rot.as_matrix()
            # z = rotation_matrix[:, 2]
            # pose[:3]-= 0.1* z
            # pose = pose[:3]
        if self.gripper:
            gripper_width = self.arm.get_gripper_position()
            pose = np.hstack([pose, gripper_width])

        # Add to buffers
        self.rgb_buffer.append(processed_img)
        self.pose_buffer.append(pose)
        return colors

    def _process_image(self, colors):
        """Image processing pipeline"""
        height, width = colors.shape[:2]
        if width > height:
            start_x = (width - height) // 2  # (W - target_width) // 2
            end_x = start_x + height

            # extract central area
            center_crop_color_image = colors[:, start_x:end_x]
            
        else:
            start_y = (height - width) // 2  # (W - target_width) // 2
            end_y = start_y + width
            
            # extract central area
            center_crop_color_image = colors[start_y:end_y, :]
            
        # Then resize to 224x224 for policy input
        center_crop_color_image = cv2.resize(center_crop_color_image, (224, 224), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(center_crop_color_image, cv2.COLOR_BGR2RGB)
    
    def get_observation(self):
        """Returns dictionary with n_obs_steps of buffered data"""
        # Always capture new observation before returning
        raw_img = self._add_single_observation()
        
        return {
            'img': np.stack(self.rgb_buffer),  # Shape: (n_obs_steps, 224, 224, 3)
            'state': np.stack(self.pose_buffer),  # Shape: (n_obs_steps, 10)
            "raw_img": raw_img.copy(),
        }

    def set_tcp_pose(
        self, pose, blocking=False
    ):
        self.arm.move_ee(pose)
        
    def set_tcp_gripper(
        self, gripper, blocking=False
    ):
        self.arm.move_gripper(gripper > 0.5)
    
    def sleep(self):
        rospy.sleep(0.2)
        
if __name__ == "__main__":
    agent = Agent(obs_num=1, gripper="panda")
    while True:
        agent.get_observation()