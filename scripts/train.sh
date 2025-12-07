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

while [[ $# -gt 0 ]]; do
    case $1 in
        --input | -i)
            data_dir="$2"
            shift 2
            ;;
        --task | -t)
            task="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# training stacking task
python train.py \
    --config-name=train_diffusion_unet_real_image_workspace \
    task=$task \
    task.dataset_path=$data_dir

