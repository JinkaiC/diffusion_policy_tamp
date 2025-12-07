# Zarr 格式标准化总结

## 更新内容

### 时间
- 日期：2025年12月
- 脚本：`utils/h5_to_zarr_ee_states.py`
- 目标：确保与 diffusion-policy 标准格式完全兼容

### 关键变更

#### 1. Action 向量合并
**之前（分离格式）**:
```
data/
├── img              (T, 96, 96, 3) float32
├── ee_states        (T, 7) float32
└── gripper_control  (T, 1) float32
```

**之后（标准格式）**:
```
data/
├── img       (T, 96, 96, 3) float32
└── action    (T, 8) float32  ← 合并格式
```

#### 2. Action 向量内容
新的 `data/action` 数组包含：
- **索引 0-6**: ee_states (7-dim)
  - 0-2: End-effector 位置 (x, y, z)
  - 3-6: 四元数姿态 (qx, qy, qz, qw)
- **索引 7**: gripper_control (1-dim)
  - 夹爪控制信号

**合并公式**:
```python
action[t] = np.concatenate([ee_states[t], gripper_control[t]])
```

#### 3. 格式兼容性
- ✓ 符合 diffusion-policy 标准（单一 `data/action` 数组）
- ✓ 与其他标准数据集（如 pusht_cchi_v7_replay.zarr）格式一致
- ✓ 简化数据加载和模型输入处理
- ✓ 保持完整的控制信息（无数据丢失）

### 实现细节

#### 脚本修改位置
1. **第 85-95 行**: 移除了分离的 `ee_states_group` 和 `gripper_control_group` 创建
2. **第 165-185 行**: 合并两个列表为单一 `actions_list`
   ```python
   action = np.concatenate([ee_states, gripper_control], axis=0)
   actions_list.append(action)
   ```
3. **第 195-200 行**: 单一 action 向量的填充
4. **第 260-270 行**: 创建单一 `data/action` 数组而非分离的三个数组

#### 关键优势
- **简化**: 减少 zarr 中的数据集数量
- **标准**: 符合 diffusion-policy 框架的期望格式
- **兼容**: 与其他学习框架和预训练模型兼容
- **效率**: 单一 action 数组提升数据加载效率

### 验证结果

**生成的 zarr 文件**：`outputs/banana_plate_ee_states.zarr`

```
Top-level keys: [meta, data]
Data keys: [action, img]
  img shape: (387, 96, 96, 3), dtype: float32
  action shape: (387, 8), dtype: float32
Meta keys: [episode_ends]
  episode_ends: [387]
```

**格式对比**：
| 特性 | 新格式 (banana_plate_ee_states) | 标准格式 (pusht_cchi_v7_replay) |
|------|------|------|
| img shape | (387, 96, 96, 3) | (25650, 96, 96, 3) |
| action shape | (387, 8) | (25650, 2) |
| action dtype | float32 | float32 |
| 结构 | {img, action} | {img, action, ...} |
| ✓ 兼容性 | 是 | 是 |

### 后续使用

#### 数据加载示例
```python
import zarr

z = zarr.open('outputs/banana_plate_ee_states.zarr', mode='r')
images = z['data']['img'][:]       # (T, 96, 96, 3)
actions = z['data']['action'][:]   # (T, 8)

# 解析 action 向量
ee_states = actions[:, :7]         # (T, 7)
gripper_control = actions[:, 7:]   # (T, 1)
```

#### Diffusion-policy 集成
该格式现已完全兼容 diffusion-policy 框架：
- 模型可直接读取 `data/action` 作为输入
- 无需额外的数据处理或格式转换
- 支持标准的批加载和数据管道

### 文档更新

- ✓ `ZARR_CONVERSION_GUIDE.md`: 更新了输出结构和 action 向量说明
- ✓ `utils/h5_to_zarr_ee_states.py`: 更新了函数 docstring
- ✓ 创建本文档 (`FORMAT_STANDARDIZATION_SUMMARY.md`): 总结格式变更

### 备注

- 原始的 `utils/h5_to_zarr_explicit.py` 保持不变，支持其他 action 组合方式
- 若需要其他 action 配置，可参考 `h5_to_zarr_explicit.py` 进行定制
- 生成的 zarr 可直接用于 diffusion-policy 训练
