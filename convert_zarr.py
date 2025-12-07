import os
os.environ["DISPLAY"] = ":1"  # Set DISPLAY environment variable for Open3D
import cv2
import zarr
import click
import numpy as np
from tqdm import tqdm
from termcolor import cprint
from typing import Tuple, List, Optional, Dict, Union
import h5py

from diffusion_policy.dataset.utils.constants import *
from diffusion_policy.dataset.utils.process_obs import (
    convert_tcp_data_to_camera,
)
from diffusion_policy.dataset.utils.convert_util import (
    handle_existing_data,
)
from diffusion_policy.model.common.rotation_transformer import RotationTransformer

quat2rot6d_transformer = RotationTransformer(from_rep='quaternion', to_rep="rotation_6d")
        
# Compression settings (adjust clevel for size/speed tradeoff)
DEFAULT_COMPRESSOR = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
DEFAULT_CHUNK_SIZE = 100

class DatasetConfig:
    """Configuration class for dataset conversion parameters."""

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        frame: str,
        val_ratio: float = 0.0,
        gripper: bool = False,
    ):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.frame = frame
        self.val_ratio = val_ratio
        self.gripper = gripper
        
def initialize_data_containers() -> Dict[str, List]:
    """Initialize data containers for storing different types of data."""
    return {
        "img": [],
        "state": [],
        "action": [],
        "time": [],
        "episode_ends": [],
    }


def load_demo_files(demo_dir: str, sim: bool, mod: int) -> List[str]:
    """Load demonstration files from a directory, sorted by numeric value."""
    if not os.path.exists(demo_dir):
        raise FileNotFoundError(f"Demonstration directory not found: {demo_dir}")

    # Extract only files that contain digits and sort numerically
    if sim:
        demo_files = [x for x in os.listdir(demo_dir) if x.endswith(".h5") and int(os.path.splitext(x)[0]) % mod == 0]
        demo_files = sorted(
            demo_files, key=lambda f: int(os.path.splitext(f)[0])
        )
    else:
        demo_files = sorted(
            os.listdir(demo_dir), key=lambda f: int("".join(filter(str.isdigit, f)) or 0)
        )

    return demo_files

def extract_demo_data(step_path: str, gripper: bool, sim: bool) -> Tuple:
    """Extract and preprocess data from a demonstration step."""
    # Check if the required keys are present in the demonstration data
    try:
        if not sim:
            demo = np.load(step_path, allow_pickle=True)
            if gripper:
                gripper_joint_positions = demo.get("gripper_joint_positions", None)
                if gripper_joint_positions is not None:
                    try:
                        gripper_joint_positions = float(gripper_joint_positions[0])
                    except:
                        gripper_joint_positions = float(gripper_joint_positions)
                elif len(demo.get('arm_joint_positions', [])) > 7: # use joint 7 as gripper for sim
                    gripper_joint_positions = float(demo.get('arm_joint_positions', [])[7])
            else:
                gripper_joint_positions = None
            return (
                demo.get("arm_ee_pose", None),
                demo.get("arm_ee_control", None), 
                cv2.cvtColor(demo.get("camera_1_color_image", None), cv2.COLOR_BGR2RGB), # OpenCV BGR --> RGB
                demo.get("time", None),
                gripper_joint_positions,
            )

        else:
            with h5py.File(step_path, 'r') as f:
                arm_ee_pose = np.array(f['ee_states'])
                arm_ee_control = np.array(f['ee_control'])
                if gripper:
                    gripper_joint_positions = float(f.get("gripper_control", None)[0])
                else:
                    gripper_joint_positions = None
                timestamp = f.get("timestamp", [int(os.path.basename(step_path).split('.h5')[0])])[0]
            img_path = step_path.replace(".h5", ".jpg")
            if not os.path.exists(img_path):
                img_path = step_path.replace(".h5", ".png")
            rgb_image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            return (
                arm_ee_pose,
                arm_ee_control,
                rgb_image,
                timestamp,
                gripper_joint_positions,
            )
            
    except KeyError as e:
        raise KeyError(f"Missing required key in demonstration data: {e}")
def transform_ee_pose_frame(ee_pose: np.ndarray, frame: str) -> np.ndarray:
    """Transform end effector pose to the specified reference frame."""
    if frame == "camera":
        return convert_tcp_data_to_camera(ee_pose)
    elif frame == "base":
        return ee_pose
    else:
        raise ValueError(f"Unsupported frame type: {frame}. Use 'camera' or 'base'.")


def process_demo_step(
    step_path: str,
    frame: str,
    gripper: bool = False,
    sim: bool = False,
    ignore_rotation: bool = False,
    old_rgb_crop: str = "",
) -> Tuple:
    """Process a single step of demonstration data."""
    try:
        # Extract data components
        (
            arm_ee_pose,
            arm_ee_control, 
            color_image,
            time,
            gripper_joint_positions,
        ) = extract_demo_data(step_path, gripper, sim=sim,)
            
        # Transform poses to the specified frame
        # 7-dim: x, y, z, qw, qx, qy, qz
        transformed_ee_pose = transform_ee_pose_frame(arm_ee_pose, frame)
        xyz, quat = transformed_ee_pose[:3], transformed_ee_pose[3:7]
        rot6d = quat2rot6d_transformer.forward(np.array(quat))
        # 9-dim: x, y, z, rot6d
        if ignore_rotation:
            state = xyz
        else:
            state = np.concatenate([xyz, rot6d])
        
        # Transform poses to the specified frame
        # 7-dim: x, y, z, qw, qx, qy, qz
        transformed_controled_ee_pose = transform_ee_pose_frame(arm_ee_control, frame)
        action_xyz, action_quat = transformed_controled_ee_pose[:3], transformed_controled_ee_pose[3:7]
        action_rot6d = quat2rot6d_transformer.forward(np.array(action_quat))
        # 9-dim: x, y, z, rot6d
        if ignore_rotation:
            action = action_xyz
        else:
            action = np.concatenate([action_xyz, action_rot6d])
        
        # 10-dim: x, y, z, rot6d, gripper
        if gripper:
            # no gripper state here
            if gripper_joint_positions is not None:
                state = np.concatenate([state, np.array([gripper_joint_positions])], axis=0)
            else:
                raise NotImplementedError(f"No gripper position. Skip this data: {step_path}")
            if gripper_joint_positions is not None:
                action = np.concatenate([action, np.array([gripper_joint_positions])], axis=0)
            else:
                raise NotImplementedError(f"No gripper action. Skip this data: {step_path}")
        
        if len(old_rgb_crop) > 0:
            h, w = color_image.shape[:2]
            # Crop to 640x480, positioned higher to see more details
            start_x = (w - 640) // 2
            start_y = max(0, (h - 480) // 2 - 100)  # Shift up by 100 pixels to see more details
            # Crop to a square region (480x480) from the center for better resizing
            # Adjust the vertical position to get a higher view (reduce start_y by 100 pixels)
            adjusted_start_y = max(0, start_y - 100)
            if old_rgb_crop == "stacking":
                x_offset = 100
            else:
                x_offset = 0
            center_crop_color_image = color_image[adjusted_start_y:adjusted_start_y+480, start_x+(640-480)//2+x_offset:start_x+(640-480)//2+480+x_offset]
        else:
            height, width = color_image.shape[:2]
            if width > height:
                start_x = (width - height) // 2  # (W - target_width) // 2
                end_x = start_x + height

                # extract central area
                center_crop_color_image = color_image[:, start_x:end_x]
                
            else:
                start_y = (height - width) // 2  # (W - target_width) // 2
                end_y = start_y + width
                
                # extract central area
                center_crop_color_image = color_image[start_y:end_y, :]
            
        # Then resize to 224x224 for policy input
        center_crop_color_image = cv2.resize(center_crop_color_image, (224, 224), interpolation=cv2.INTER_AREA)

        return center_crop_color_image, state, action, time

    except Exception as e:
        cprint(f"Error processing demo step {step_path}: {str(e)}", "red")
        return None, None, None, None

def get_val_mask(n_episodes: int, val_ratio: float, seed: int = 0) -> np.ndarray:
    """Generate a boolean mask for validation data selection."""
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask

    # Have at least 1 episode for validation, and at least 1 episode for training
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes - 1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask


def save_zarr_dataset(
    zarr_group: zarr.Group,
    name: str,
    data: Union[List, np.ndarray],
    chunks: Optional[Tuple] = None,
    dtype: Optional[str] = None,
    compressor: Optional[zarr.Blosc] = None,
) -> None:
    """Helper function to save dataset to zarr group with standard parameters."""
    # Skip empty datasets
    if (
        data is None
        or (isinstance(data, np.ndarray) and data.size == 0)
        or (isinstance(data, list) and len(data) == 0)
    ):
        return

    # Convert list to numpy array if needed
    if isinstance(data, list):
        try:
            data = np.stack(data, axis=0)
        except ValueError:
            cprint(f"Could not stack data for {name}, skipping", "red")
            return

    # Use data's dtype if not specified
    if dtype is None:
        dtype = data.dtype

    # Use default compressor if not specified
    if compressor is None:
        compressor = DEFAULT_COMPRESSOR

    # Set default chunks if not specified
    if chunks is None:
        chunks = (DEFAULT_CHUNK_SIZE,) + data.shape[1:]

    # Create dataset
    try:
        zarr_group.create_dataset(
            name, data=data, chunks=chunks, dtype=dtype, compressor=compressor
        )
    except Exception as e:
        cprint(f"Error saving {name} dataset: {str(e)}", "red")


def convert_to_zarr(
    data: List[str],
    data_dir: str,
    output_dir: str,
    frame: str,
    gripper: bool = False,
    debug: bool = False,
    sim: bool = False,
    mod: int = 5,
    ignore_rotation: bool = False,
    old_rgb_crop: bool = False,
) -> None:
    """Convert demonstration data to zarr format"""
    # Initialize data containers
    data_containers = initialize_data_containers()
    total_count = 0

    # Unpack data containers for easier access
    img_arrays = data_containers["img"]
    state_arrays = data_containers["state"]
    action_arrays = data_containers["action"]
    time_arrays = data_containers["time"]
    episode_ends_arrays = data_containers["episode_ends"]

    # Process each demonstration
    for demo_name in tqdm(data, desc="Processing demonstrations"):
        demo_dir = os.path.join(data_dir, demo_name)
        demo_files = load_demo_files(demo_dir, sim=sim, mod=mod)
        num_steps = len(demo_files)

        color_images = []
        states = []
        actions = []
        times = []
        # Process all steps within the demonstration
        for step_idx in range(num_steps):
            step_path = os.path.join(demo_dir, demo_files[step_idx])
            try:
                # Process current step (full data)
                (
                    color_image,
                    state,
                    action,
                    time,
                ) = process_demo_step(
                    step_path,
                    frame,
                    gripper,
                    sim=sim,
                    ignore_rotation=ignore_rotation,
                    old_rgb_crop=old_rgb_crop,
                )

                # Debug visualization if enabled
                #! Render RGB image for visualization
                if debug and step_idx == 0:
                    from matplotlib import pyplot as plt
                    plt.figure(figsize=(12, 8))
                    plt.subplot(1, 2, 1)
                    plt.imshow(color_image)
                    plt.title('RGB Image')
                    
                    plt.tight_layout()
                    plt.show()

                # Store current step data
                color_images.append(color_image)
                states.append(state)
                actions.append(action)
                times.append(time)
                
            except Exception as e:
                cprint(f"Error processing step {step_path}: {str(e)}", "red")
                cprint("Skipping this step and continuing...", "yellow")
                continue

        # Mark the end of each episode
        img_arrays.extend(color_images)
        state_arrays.extend(states)
        action_arrays.extend(actions)
        time_arrays.extend(times)
        total_count += len(states)
        episode_ends_arrays.append(total_count)
            
    # Create zarr file structure
    os.makedirs(output_dir, exist_ok=True)
    zarr_root = zarr.group(output_dir)
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")
    compressor = DEFAULT_COMPRESSOR

    # Save all datasets
    save_zarr_dataset(zarr_data, "img", img_arrays, None, "uint8", compressor)

    # Convert remaining lists to arrays and save
    try:
        if state_arrays:
            state_array = np.stack(state_arrays, axis=0)
            chunks = (DEFAULT_CHUNK_SIZE, state_array.shape[1])
            save_zarr_dataset(
                zarr_data, "state", state_array, chunks, "float32", compressor
            )
        if action_arrays:
            action_array = np.stack(action_arrays, axis=0)
            chunks = (DEFAULT_CHUNK_SIZE, action_array.shape[1])
            save_zarr_dataset(
                zarr_data, "action", action_array, chunks, "float32", compressor
            )
        if time_arrays:
            time_array = np.array(time_arrays)
            save_zarr_dataset(
                zarr_data, "time", time_array, None, "float32", compressor
            )
        if episode_ends_arrays:
            episode_ends_array = np.array(episode_ends_arrays)
            save_zarr_dataset(
                zarr_meta,
                "episode_ends",
                episode_ends_array,
                (DEFAULT_CHUNK_SIZE,),
                "int64",
                compressor,
            )
    except Exception as e:
        cprint(f"Error saving arrays: {str(e)}", "red")

    # Print summary of dataset
    print_dataset_summary(data_containers)
    cprint(f"Saved zarr file to {output_dir}", "green")


def print_dataset_summary(data_containers: Dict[str, List]) -> None:
    """Print summary of dataset shapes."""
    cprint("Final dataset shapes:", "cyan")
    cprint(f"Last episode ends: {data_containers['episode_ends'][-1]}", "cyan")
    assert data_containers['episode_ends'][-1] == np.array(data_containers['state']).shape[0], "length not consistent!"
    # Helper function to print shape if data exists
    def print_shape(name, data):
        if isinstance(data, list) and len(data) > 0:
            try:
                shape = np.array(data).shape
                cprint(f"  {name}: {shape}", "cyan")
            except:
                cprint(f"  {name}: {len(data)} items (varied shapes)", "cyan")
        elif isinstance(data, np.ndarray) and data.size > 0:
            cprint(f"  {name}: {data.shape}", "cyan")

    # Print shapes for all data types
    for key, data in data_containers.items():
        print_shape(key, data)


@click.command()
@click.option(
    "--data_dir",
    type=str,
    required=True,
    help="Path to the directory containing the data.",
)
@click.option(
    "--output_dir",
    type=str,
    required=False,
    default="",
    help="Path to save the data in zarr format.",
)
@click.option(
    "--frame", type=str, default="base", help="Frame to transform the point cloud to."
)
@click.option("--val_ratio", type=float, default=0.0, help="Validation dataset ratio.")
@click.option(
    "--force_overwrite", type=bool, default=True, help="Force overwrite existing data."
)
@click.option(
    "--gripper", type=bool, default=False, help="Whether to use gripper states."
)
@click.option(
    "--debug", type=bool, default=False, help="Whether to debug."
)
@click.option(
    "--nots", type=bool, default=False, help="Whether to debug."
)
@click.option(
    "--sim", type=bool, default=False, help="Whether is sim generated data."
)
@click.option(
    "--ignore_rotation", type=bool, default=False, help="Whether is sim generated data."
)
@click.option(
    "--mod", type=int, default=5, help="Sim Sample frequency."
)
@click.option(
    "--max_demo_num", type=int, default=200, help="Sample frequency."
)
@click.option(
    "--old_rgb_crop", type=str, default="",
)
def main(
    data_dir: str,
    output_dir: str,
    frame: str,
    val_ratio: float,
    force_overwrite: bool,
    gripper: bool,
    debug: bool,
    nots: bool,
    sim: bool,
    mod: int,
    max_demo_num: int,
    ignore_rotation: bool,
    old_rgb_crop: str,
):
    
        
    """Convert demonstration data to zarr format."""
    try:

        # List and sort demonstration directories
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        data = []
        if nots:
            demo_names = os.listdir(data_dir)
            demo_names.sort(key=lambda f: int("".join(filter(str.isdigit, f)) or 0))
            data = demo_names
        else:
            for timestamp in sorted(os.listdir(data_dir)):
                demo_names = os.listdir(os.path.join(data_dir, timestamp))
                demo_names.sort(key=lambda f: int("".join(filter(str.isdigit, f)) or 0))
                data.extend([os.path.join(timestamp, demo_name) for demo_name in demo_names])
                
        cprint(f"Original data demo num {len(data)} --> Current {min(len(data), max_demo_num)}", "red")
        data = data[:max_demo_num]
        if len(output_dir) == 0:
            splitted_data_dir = data_dir.rstrip(os.sep).split(os.sep)
            sim_str = "sim_" if sim else ""
            splitted_data_dir[-1] = splitted_data_dir[-1] + f"_zarr_dp_{sim_str}demonum{len(data)}"
            output_dir = os.sep.join(splitted_data_dir)
            
        if not data:
            raise ValueError(f"No demonstration directories found in {data_dir}")

        # Handle existing data at the output location
        handle_existing_data(output_dir, force_overwrite)

        # Split data into training and validation sets
        val_mask = get_val_mask(len(data), val_ratio)
        train_data = np.array(data)[~val_mask].tolist()
        val_data = np.array(data)[val_mask].tolist()

        cprint(f"Processing {len(train_data)} training demonstrations", "blue")

        # Process training data
        convert_to_zarr(
            train_data,
            data_dir,
            os.path.join(output_dir, "train"),
            frame,
            gripper,
            debug,
            sim,
            mod,
            ignore_rotation,
            old_rgb_crop,
        )

        # Process validation data if needed
        if val_ratio > 0:
            cprint(f"Processing {len(val_data)} validation demonstrations", "blue")
            convert_to_zarr(
                val_data,
                data_dir,
                os.path.join(output_dir, "val"),
                frame,
                gripper,
                debug,
                sim,
                mod,
                ignore_rotation,
                old_rgb_crop,
            )

        cprint("Dataset conversion completed successfully!", "green")

    except Exception as e:
        cprint(f"Error in dataset conversion: {str(e)}", "red")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    main()
