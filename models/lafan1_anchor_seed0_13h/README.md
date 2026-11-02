# 已训练的 LAFAN1 位姿反馈策略

| 项目 | 内容 |
|---|---|
| 模型 | `ckpt_0144050.pt` |
| 机器人 | Unitree G1，29 个关节 |
| 数据 | LAFAN1 G1，40 条动作，50 Hz |
| 训练 | seed 0，4096 环境，13 小时时间上限 |
| 已完成 | 144,050 次环境迭代，57,604 次训练更新 |
| 位姿反馈 | `anchor_obs=True`，目标输入 689 维 |
| 整段评测 | 平均归一化回报 0.9360，提前终止 0/40 |
| 权重大小 | 215,679,331 字节，约 205.69 MiB |
| 原训练代码 | `CuboiLeo/SuperMimic-Reproduced@ba6105349069bf040f1fdad60873d13138d505a2` |

这是本次 LAFAN1 位姿反馈实验训练得到的完整 checkpoint，包含策略、世界模型、critic、优化器状态、配置和迭代信息。可直接用于评测、轨迹录制，也可按原配置续训。它是后续太极学习和卸力实验使用的基础模型。

## 文件

- `ckpt_0144050.pt`：完整训练 checkpoint，由 Git LFS 管理。
- `config.json`：该次训练的配置。
- `benchmark.json`：40 条 LAFAN1 动作的整段评测，按完整参考动作帧数归一化。
- `SHA256SUMS`：上述三个文件的完整性校验值。

在仓库根目录下载和校验：

```bash
git lfs pull
(cd models/lafan1_anchor_seed0_13h && sha256sum -c SHA256SUMS)
```

安装环境、准备 LAFAN1 NPZ 后，在仓库根目录评测：

```bash
mkdir -p logs/pretrained_anchor_eval
MUJOCO_GL=disable env -u PYTHONPATH uv run --locked python scripts/eval_benchmark.py \
  --ckpt models/lafan1_anchor_seed0_13h/ckpt_0144050.pt \
  --npz-dir data/npz/lafan1 --num-envs 40 \
  --out logs/pretrained_anchor_eval/benchmark.json
```

视频录制命令见[仓库使用说明](../../README.md#34-使用仓库提供的已训练策略)。
