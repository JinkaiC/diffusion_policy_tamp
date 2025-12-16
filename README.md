# TwinAligner Diffusion Policy

This is the policy implementation of Diffusion Policy in TwinAligner.

## Dependencies

- Ubuntu 20.04
- CUDA 11.8
- ROS noetic
    
## Installation

```bash
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-11.8

git clone --recurse-submodules ssh://git@gitlab.hwfan.cn:11452/hwfan/twinaligner-policy-2d.git
cd twinaligner-policy-2d
conda env create -f conda_environment.yaml
pip install huggingface_hub==0.25.0
ln -sf /usr/lib/x86_64-linux-gnu/libffi.so.7 $CONDA_PREFIX/lib/libffi.so.7
pip install -e .
```

### Franka Control

```bash
conda activate robodiff
unset ROS_DISTRO && source /opt/ros/noetic/local_setup.bash
pip install -e dependencies/frankapy
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install -e dependencies/curobo --no-build-isolation --verbose
cd dependencies/frankapy && ./bash_scripts/make_catkin.sh && cd ../..
pip install protobuf==3.20 click==8.1.8
```
## Convert ZARR

```bash
conda activate robodiff
bash scripts/generate_data.sh -i ../REAL-ROBO/expert_dataset/recorded_data/banana_plate --gripper True
# --nots
# --debug
# --sim
```

## Training

```bash
conda activate robodiff
bash scripts/train.sh -i /DATA/disk1/hwfan/twinaligner_data/banana_plate_zarr_dp_real50/train/ --task pnp
```

## Deployment

### Real (Franka)

!!! NOTE: Please change the settings in `dependencies/frankapy/bash_scripts/start_control_pc.sh` to your own.

```bash
conda activate robodiff
# Step 1: start franka daemon processes
bash scripts/start_franka.sh
# Step 2: start eval process
bash scripts/eval.sh -i $checkpoint_path -g
```

## Acknowledgement

Maintained by Jinzhou Li ([@kingchou007](https://github.com/kingchou007)) and Hongwei Fan ([@hwfan](https://github.com/hwfan)).
