# Zarr 转换脚本使用指南

## 概述
本指南介绍如何使用转换脚本将原始的 .h5 逐帧数据和图像转换为 zarr 格式，用于 diffusion-policy 训练。

## 现有脚本

### 1. `utils/h5_to_zarr_ee_states.py`（推荐）
**用途**: 将 `ee_states` 和 `gripper_control` 合并为单个 action array，符合 diffusion-policy 标准格式

**特点**:
- 读取 end-effector states (7-dim: 位置 3 + 四元数 4)
- 读取 gripper_control (1-dim: 布尔值或控制信号)
- **合并为单个 action 向量 (8-dim)** - diffusion-policy 标准格式
- 自动转换为 float32
- 支持图像缩放和归一化
- 使用 Blosc 压缩（zstd）
- 分块对齐（time_chunk=161）

**使用示例**:
```bash
python3 utils/h5_to_zarr_ee_states.py \
    --record_dir datasets/records/banana_plate/20251201_154925_530/00000 \
    --out_zarr outputs/banana_plate_ee_states.zarr \
    --resize 96 96 \
    --normalize 0-1 \
    --layout array \
    --time_chunk 161 \
    --overwrite \
    --write_meta \
    --compressor_name zstd \
    --clevel 5
```

**输出结构** (diffusion-policy 标准格式):
```
outputs/banana_plate_ee_states.zarr/
├── data/
│   ├── img       # (T, 96, 96, 3) float32, T=帧数
│   └── action    # (T, 8) float32 - 合并的 [ee_states(7) + gripper_control(1)]
└── meta/
    └── episode_ends  # [T], 累积 episode 边界
```

**Action 向量结构**:
- 前 7 个分量: ee_states (end-effector 位置和姿态)
  - 索引 0-2: 位置 (x, y, z)
  - 索引 3-6: 四元数 (qx, qy, qz, qw)
- 最后 1 个分量: gripper_control (夹爪控制信号)

**命令行参数**:
- `--record_dir`: 包含 .h5 文件的目录
- `--out_zarr`: 输出 zarr 路径
- `--resize W H`: 缩放图像到 W×H (例如 96 96)
- `--normalize {0-1,-1-1,none}`: 图像归一化方式 (默认: 0-1)
- `--layout {array,group}`: 
  - `array`: 合并所有 episode 为单一数组（推荐用于 diffusion-policy）
  - `group`: 保持按 episode 分组
- `--time_chunk`: 时间维度分块大小（默认: 161）
- `--write_meta`: 是否写入 `meta/episode_ends`
- `--compressor_name {zstd,lz4}`: 压缩算法
- `--clevel`: 压缩等级 (1-22, 默认: 5)

**Diffusion-policy 兼容性**:
✓ 标准格式: `data/` 下只有 `img` 和 `action`
✓ Action 合并: ee_states 和 gripper_control 合并为单个向量
✓ 数据类型: 所有 action 为 float32
✓ 结构简洁: 与 pusht_cchi_v7_replay.zarr 等标准数据集格式一致

### 2. `utils/h5_to_zarr_explicit.py`（备选）
**用途**: 将 `joint_control` 和 `gripper_control` 打包为 action

**特点**:
- 读取 joint control (关节空间控制)
- 适合需要关节空间动作的场景
- 支持自定义 action_keys（逗号分隔）

**使用示例**:
```bash
python3 utils/h5_to_zarr_explicit.py \
    --record_dir datasets/records/banana_plate/20251201_154925_530/00000 \
    --out_zarr outputs/banana_plate_joint.zarr \
    --action_keys joint_control,gripper_control \
    --resize 96 96 --normalize 0-1 --layout array --time_chunk 161 \
    --overwrite --write_meta
```

## 批量转换

对多个 episode 进行转换（假设记录目录结构为 `records/<timestamp>/<episode_id>/`）：

```bash
# 若记录按 timestamp 子目录组织
python3 utils/h5_to_zarr_ee_states.py \
    --record_dir datasets/records/banana_plate/ \
    --out_zarr outputs/banana_plate_ee_states_all.zarr \
    --resize 96 96 --normalize 0-1 --layout array \
    --time_chunk 161 --overwrite --write_meta

# 若直接在 record_dir 下包含多个 episode 目录
python3 utils/h5_to_zarr_ee_states.py \
    --record_dir datasets/records/banana_plate/ \
    --no_timestamps \
    --out_zarr outputs/banana_plate_ee_states_all.zarr \
    --resize 96 96 --normalize 0-1 --layout array \
    --time_chunk 161 --overwrite --write_meta
```

## 验证输出

使用 Python 检查生成的 zarr：

```python
import zarr
root = zarr.open('outputs/banana_plate_ee_states.zarr', mode='r')
print('Keys:', list(root.keys()))
print('img shape:', root['data']['img'].shape)
print('ee_states shape:', root['data']['ee_states'].shape)
print('gripper_control shape:', root['data']['gripper_control'].shape)
print('episode_ends:', root['meta']['episode_ends'][:])
```

## Diffusion-Policy 训练集成

生成的 zarr 可直接用于 diffusion-policy：

```python
import zarr

# 加载 zarr
root = zarr.open('outputs/banana_plate_ee_states.zarr', mode='r')
images = root['data']['img'][:]  # (T, 96, 96, 3)
ee_states = root['data']['ee_states'][:]  # (T, 7)
gripper = root['data']['gripper_control'][:]  # (T, 1)
actions = np.concatenate([ee_states, gripper], axis=1)  # (T, 8)

# 或作为 diffusion-policy 的动作空间：
# - observation: images
# - action: [ee_states (7-dim), gripper_control (1-dim)]
```

## 常见问题

**Q: 脚本跳过了许多 .h5 文件**  
A: 这是正常的。脚本要求每一帧必须有：
   1. 外部图像（`.jpg`, `.png` 等）或 .h5 内的图像数据
   2. `ee_states` 数据集
   3. `gripper_control` 数据集
   
   若某帧缺失任何一项，会被跳过并打印警告。

**Q: 输出的 ee_states 维度与预期不符**  
A: 脚本会自动 pad 不同长度的向量到最大长度（通常为 7）。若需要降维或变换，可在训练前处理。

**Q: Zarr 文件很大**  
A: 使用 `--clevel` 调整压缩等级（1-22，值越大压缩率越高但速度越慢）。例如 `--clevel 9` 可获得更高压缩率。

**Q: 如何跳过缩放和归一化？**  
A: 使用 `--normalize none` 并省略 `--resize` 参数。

## 脚本修改历史

- **v1** (h5_to_zarr_explicit.py): 读取 `joint_control` 和 `gripper_control`
- **v2** (h5_to_zarr_ee_states.py): 读取 `ee_states` 和 `gripper_control`，支持分离的数据集（用于稳定学习）
