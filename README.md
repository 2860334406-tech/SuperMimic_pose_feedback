# SuperMimic 位姿反馈实验

本仓库用于 Unitree G1 的动作跟踪实验。代码基于 [CuboiLeo/SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced)，保留其训练、评测和视频渲染流程。本次使用仓库已有的 `anchor_obs` 位姿反馈选项训练策略，并修正了整段动作评测和更换动作库续训时的处理。

## 本次使用的功能

- `--env.anchor-obs`：在目标观测中加入参考动作与机器人之间的相对位置和朝向。
- `scripts/data/csv_to_npz_batch.py`：将 G1 动作 CSV 转为训练用 NPZ。
- `sm-train`：训练或从 checkpoint 续训策略。
- `scripts/eval_benchmark.py`：逐条评测整段动作。
- `scripts/media/record_rollout.py`、`render_rollout.py`：保存仿真轨迹并渲染跟踪视频。

## 本地代码修正

- `src/supermimic/eval/benchmark.py`：整段动作的归一化回报使用动作完整帧数作为分母。
- `src/supermimic/training/loop.py`：从 checkpoint 切换到新的动作库时，重置与旧动作库对应的采样统计。

## 运行示例

```bash
uv sync --locked --group dev

# 数据准备完成后，启用位姿反馈训练
MUJOCO_GL=disable env -u PYTHONPATH uv run --locked sm-train \
  --run-name lafan1_anchor --data.npz-dir data/npz/lafan1 \
  --env.anchor-obs --env.num-envs 4096

# 使用保存的 checkpoint 评测
MUJOCO_GL=disable env -u PYTHONPATH uv run --locked python scripts/eval_benchmark.py \
  --ckpt logs/lafan1_anchor/ckpt_latest.pt \
  --npz-dir data/npz/lafan1 --num-envs 40
```

训练数据、checkpoint 和实验视频单独保存，不放在代码仓库中。原项目许可证见 [LICENSE](LICENSE)。
