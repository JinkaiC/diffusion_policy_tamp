#!/bin/bash
# Parse parameters
export HYDRA_FULL_ERROR=1 
export OMP_NUM_THREADS=24

gpu_info=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
min_memory=999999
selected_gpu=""
while IFS=, read -r index memory; do
    index=$(echo $index | xargs)
    memory=$(echo $memory | xargs)
    if [ $memory -lt $min_memory ]; then
        min_memory=$memory
        selected_gpu=$index
    fi
done <<< "$gpu_info"

if [ -z "$selected_gpu" ]; then
    echo "Error: No available GPU found."
    exit 1
fi

export CUDA_VISIBLE_DEVICES=$selected_gpu

echo "Selected GPU $selected_gpu with $min_memory MB memory used."

gripper=""
sim=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --input | -i)
            data_dir="$2"
            shift 2
            ;;
        --gripper | -g)
            gripper="-g"
            shift 1
            ;;
        --sim | -s)
            sim="-s"
            shift 1
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

unset ROS_DISTRO
source /opt/ros/noetic/local_setup.bash
if [[ "$sim" != "-s" ]]; then
    cd dependencies/frankapy
    source catkin_ws/devel/setup.bash
    cd ../..
fi
python eval_real_robot_continue.py -i $data_dir $gripper $sim