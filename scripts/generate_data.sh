export HYDRA_FULL_ERROR=1 
export OMP_NUM_THREADS=24
debug=False
nots=False
sim=False
max_demo_num=200
ir=False
old_rgb_crop=""
mod=5

# Parse parameters
while [[ $# -gt 0 ]]; do
    case $1 in
        --input | -i)
            data_dir="$2"
            shift 2
            ;;
        --gripper)
            gripper="$2"
            shift 2
            ;;
        --ignore_rotation | -ir)
            ir=True
            shift 1
            ;;
        --debug | -d)
            debug=True
            shift 1
            ;;
        --nots | -nots)
            nots=True
            shift 1
            ;;
        --sim | -s)
            sim=True
            shift 1
            ;;
        --mod)
            mod=$2
            shift 2
            ;;
        --max_demo_num | -m)
            max_demo_num="$2"
            shift 2
            ;;
        --old_rgb_crop | -orc)
            old_rgb_crop="--old_rgb_crop $2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

python convert_zarr.py \
    --data_dir $data_dir \
    --gripper $gripper \
    --debug $debug \
    --nots $nots \
    --sim $sim \
    --max_demo_num $max_demo_num \
    --mod $mod \
    --ignore_rotation $ir \
    $old_rgb_crop