# SuperMimic 位姿反馈训练与使用说明

本项目用于在 MuJoCo 仿真中训练 Unitree G1 跟踪参考动作，包含数据转换、策略训练、独立评测、漂移分析和视频生成。

代码基于 [CuboiLeo/SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced)，起始版本为 `ba6105349069bf040f1fdad60873d13138d505a2`。本次实验使用其中已有的 `anchor_obs` 相对位姿反馈，并修正了整段动作评测的归一化分母和切换动作库续训时的采样统计处理。

第一次使用，按下面的顺序操作：

**安装环境 → 下载 G1 动作 → CSV 转 NPZ → 检查环境 → 启用位姿反馈训练 → 评测 checkpoint → 记录轨迹 → 渲染视频。**

已有兼容的 checkpoint 和动作 NPZ 时，可以直接从“独立评测”和“生成视频”开始。数据、模型权重和本次实验的视频单独保存，需要自行准备；克隆代码不会自动获得训练好的策略。

## 目录

- [1. 仓库里的文件分别做什么](#1-仓库里的文件分别做什么)
- [2. 位姿反馈与训练流程](#2-位姿反馈与训练流程)
- [3. 安装运行环境](#3-安装运行环境)
- [4. 准备动作数据](#4-准备动作数据)
- [5. 检查环境与短训练](#5-检查环境与短训练)
- [6. 正式训练与后台运行](#6-正式训练与后台运行)
- [7. 查看训练状态和曲线](#7-查看训练状态和曲线)
- [8. 从 checkpoint 继续训练](#8-从-checkpoint-继续训练)
- [9. 独立评测与漂移分析](#9-独立评测与漂移分析)
- [10. 生成视频与交互查看](#10-生成视频与交互查看)
- [11. 测试新的动作](#11-测试新的动作)
- [12. 常见问题](#12-常见问题)
- [13. 保存代码和实验结果](#13-保存代码和实验结果)

## 1. 仓库里的文件分别做什么

### 1.1 根目录

| 文件或目录 | 用途 | 什么时候查看 |
|---|---|---|
| `README.md` | 本使用说明书 | 初次使用、查命令和目录用途 |
| `src/supermimic/` | 核心 Python 代码：环境、策略、世界模型、训练和评测 | 阅读方法或修改算法 |
| `scripts/` | 可在终端调用的数据、评测、诊断和视频工具 | 日常运行实验 |
| `tests/` | 数据、运动学、仿真和训练链路的自动测试 | 装好环境后检查功能 |
| `docs/` | 原项目的论文规格、实现计划、评审和实验记录，具体文件见下表 | 查实现背景；其中的历史实验数字属于原项目 |
| `media/` | 原项目随代码提供的 BONES-SEED 演示媒体 | 了解原项目展示；本次生成的视频存到各自的 `logs/<run>/demo/` |
| `pyproject.toml` | Python 要求、依赖、软件源、开发依赖及 `sm-train` 命令入口 | 安装依赖或查命令入口 |
| `uv.lock` | 锁定具体依赖版本及来源 | 用 `uv sync --locked` 重建环境 |
| `.python-version` | 指定 Python 3.12 | 供 uv 选择 Python |
| `.gitignore` | 设置 Git 忽略的环境、数据、模型和日志文件 | 查为什么某些文件不出现在 Git 状态中 |
| `LICENSE` | 原代码的 MIT 许可证和版权信息 | 使用和分发代码时保留 |

运行后还会产生以下目录，它们默认不提交到 Git：

```text
SuperMimic_pose_feedback/
├── .venv/                         # uv 创建的 Python 虚拟环境
├── data/
│   ├── raw/lafan1/g1/             # 下载的原始 CSV
│   ├── npz/lafan1/                # 转换后的 50 Hz 动作和 manifest.csv
│   └── library/                   # 可选：大动作库的紧凑打包文件
└── logs/
    └── lafan1_anchor_seed0_13h/    # 一次训练使用一个独立目录
        ├── config.json            # 实际训练配置
        ├── console.log            # 后台训练终端输出
        ├── train.pid              # 启动命令保存的后台进程号
        ├── tb/                    # TensorBoard 标量曲线
        ├── ckpt_0004000.pt         # 周期性保存的模型
        ├── ckpt_latest.pt          # 最近一次保存的模型
        ├── benchmark_ckpt_latest.json
        ├── drift_ckpt_latest.json
        └── demo/
            ├── rollouts.npz       # 策略与参考动作的轨迹
            └── videos/            # 渲染得到的 MP4
```

### 1.2 `scripts/`：可以直接运行的工具

| 文件 | 作用 |
|---|---|
| [`data/csv_to_npz_batch.py`](scripts/data/csv_to_npz_batch.py) | 批量读取 G1 动作 CSV，重采样、计算速度和身体位姿，生成 NPZ 与 `manifest.csv` |
| [`data/filter_bones_seed.py`](scripts/data/filter_bones_seed.py) | 对 BONES-SEED 做动作名称和可行性筛选，输出筛选清单；LAFAN1 入门流程无需运行 |
| [`data/build_library.py`](scripts/data/build_library.py) | 将 NPZ 打包成紧凑动作库，适用于大量动作；普通 LAFAN1 训练可以直接读取 NPZ |
| [`eval_benchmark.py`](scripts/eval_benchmark.py) | 加载 checkpoint，对指定动作逐条运行，输出归一化回报、提前终止数量及 JSON |
| [`diag_drift.py`](scripts/diag_drift.py) | 计算水平位置和朝向漂移、超出阈值的时间，以及朝向对齐前后的速度奖励 |
| [`play.py`](scripts/play.py) | 加载策略，在桌面 MuJoCo 窗口或浏览器查看动作跟踪 |
| [`media/record_rollout.py`](scripts/media/record_rollout.py) | 在仿真中运行策略，记录机器人与参考动作的位姿，保存轨迹 NPZ |
| [`media/render_rollout.py`](scripts/media/render_rollout.py) | 将已记录的轨迹渲染成 MP4，可指定动作、时间段、相机和字幕 |
| [`eval_push.py`](scripts/eval_push.py) | 在合成站立动作上测试速度扰动；直接改变根部速度，参数单位为 m/s，不是以 N 为单位的外力测试 |
| [`tb_tail.py`](scripts/tb_tail.py) | 在终端读取 TensorBoard 最新标量，不必打开网页 |
| [`merge_benchmark_shards.py`](scripts/merge_benchmark_shards.py) | 合并分批或多卡评测得到的 JSON |
| [`bench_sim_speed.py`](scripts/bench_sim_speed.py) | 测量仿真采样速度 |
| [`bench_imagination.py`](scripts/bench_imagination.py) | 测量世界模型想象轨迹及反向传播的耗时 |
| [`pull_and_eval.sh`](scripts/pull_and_eval.sh) | 原项目的远程 checkpoint 同步及循环评测脚本，内含 `~/SuperMimic` 和 BONES-SEED 子集路径，使用前需要改成自己的路径 |

常用命令参数可以通过 `--help` 查看，例如：

```bash
MUJOCO_GL=disable env -u PYTHONPATH uv run --locked sm-train --help
MUJOCO_GL=disable env -u PYTHONPATH uv run --locked python scripts/eval_benchmark.py --help
MUJOCO_GL=egl env -u PYTHONPATH uv run --locked python scripts/media/render_rollout.py --help
```

### 1.3 `src/supermimic/`：核心代码

| 路径 | 作用 |
|---|---|
| `config.py` | 集中定义训练配置，包括数据、环境、随机种子、训练预算和断点路径 |
| `scripts/train.py` | `sm-train` 的入口，解析终端参数并启动 `Trainer` |
| `training/loop.py` | 主训练循环：仿真采样、拟合世界模型、更新策略、记录日志、保存与恢复 checkpoint |
| `training/schedule.py` | 学习率调度 |
| `training/distributed.py` | 多进程训练中的参数广播和梯度同步 |
| `data/motion_convert.py` | CSV 解析、坐标和四元数转换、重采样、速度计算、MuJoCo 正向运动学 |
| `data/recorder.py` | 将仿真中的状态转移写入经验回放缓存 |
| `data/replay_buffer.py` | 缓存与采样训练片段、世界模型窗口和想象起点 |
| `envs/tracking_env_cfg.py` | G1 仿真环境配置：机器人、观测、奖励、终止条件和位姿反馈开关 |
| `envs/actions.py` | 将策略输出缩放为关节位置目标，并在训练采样时加入动作噪声 |
| `envs/state_io.py` | 仿真状态的读取和写入 |
| `envs/sim_context.py` | 摩擦、关节偏置、质心偏移等仿真上下文 |
| `envs/mdp/motion_library.py` | 加载动作库，按动作和帧索引提供参考状态、身体位姿及目标特征 |
| `envs/mdp/commands.py` | 管理当前参考动作、时间推进、重置和动作采样 |
| `envs/mdp/observations.py` | 提供未来参考帧的目标观测 |
| `envs/mdp/rewards.py` | 计算位置、朝向、速度等跟踪奖励及惩罚 |
| `envs/mdp/terminations.py` | 判断动作结束、身体高度和倾斜等终止条件 |
| `policy/nets.py` | 动作目标编码器、策略网络和价值网络 critic |
| `world_model/state_spec.py` | 世界模型的状态布局、规范化和状态误差计算 |
| `world_model/dynamics.py` | 动力学网络及网络构件 |
| `world_model/world_model.py` | 世界模型整体：状态预测和训练损失 |
| `world_model/heads.py` | 通过预测状态计算奖励、观测、终止相关特征；包含想象训练使用的相对位姿计算 |
| `world_model/wm_trainer.py` | 从回放缓存取连续窗口，更新世界模型 |
| `imagination/rollout.py` | 让策略在世界模型中生成短期预测轨迹 |
| `imagination/objective.py` | 策略优化目标与价值学习目标 |
| `imagination/policy_trainer.py` | 通过想象轨迹的梯度更新策略，并训练 critic |
| `imagination/compiled.py` | 对想象计算进行编译加速 |
| `kinematics/g1_fk.py` | 可微的 G1 正向运动学 |
| `kinematics/rotations.py` | 旋转、四元数等数学运算 |
| `eval/benchmark.py` | 加载策略、运行整段评测、汇总和保存结果 |
| 各目录中的 `__init__.py` | Python 包初始化与接口导出；例如 `envs/__init__.py` 提供 `make_env` |

阅读训练实现，可以从 `config.py → scripts/train.py → training/loop.py` 开始，再顺着调用关系查看环境、世界模型和策略训练器。

### 1.4 `tests/`、`docs/` 和 `media/`

| 文件 | 作用 |
|---|---|
| `tests/test_motion_convert.py` | 检查动作转换、关节顺序、速度和正向运动学 |
| `tests/test_g1_fk.py` | 检查可微运动学与 MuJoCo 的一致性及梯度 |
| `tests/test_env_smoke.py` | 检查环境创建、步进、观测、回放缓存和状态读写 |
| `tests/test_analytic_heads.py` | 检查世界模型中的奖励、观测、终止和相对位姿计算与仿真环境的一致性 |
| `tests/test_world_model.py` | 检查世界模型状态处理及小样本拟合 |
| `tests/test_imagination.py` | 检查想象目标、短训练和 checkpoint 恢复 |
| `tests/test_motion_library_compact.py` | 检查紧凑动作库与普通动作库的结果一致性 |
| `docs/PAPER_SPEC.md` | 原项目整理的论文参数、公式和实现要求 |
| `docs/REPRODUCTION_PLAN.md` | 原项目的实施步骤与设计选择 |
| `docs/REVIEW_LOG.md` | 原项目的实现方案评审记录 |
| `docs/EXPERIMENTS.md` | 原项目的历史实验配置、结果和问题记录 |
| `media/g1_boneseed_highlights.gif` | 原项目的 BONES-SEED 动作演示动图 |
| `media/g1_boneseed_showcase.mp4` | 原项目的 BONES-SEED 动作演示视频 |

## 2. 位姿反馈与训练流程

策略根据机器人当前的本体观测历史和参考动作，输出 29 个关节的控制动作。环境再将动作缩放、加偏置，得到关节位置目标，由仿真中的执行器控制机器人。

默认配置中，本体观测历史为 930 维，未来 10 帧参考动作特征为 680 维。启用 `--env.anchor-obs` 后，目标观测增加 9 维：参考躯干相对当前机器人躯干的位置 3 维，以及相对朝向的 6 维表示，目标观测变为 689 维。6 维朝向表示不等于 6 个旋转角。

位姿反馈同时用于仿真采样和世界模型中的想象训练。`training/loop.py` 根据开关自动设置策略目标维度，无需手工修改网络输入维度。

训练循环为：

```text
策略在 MuJoCo 中执行动作
    ↓
状态、实际执行动作、后继状态等进入回放缓存
    ↓
世界模型学习预测状态变化
    ↓
策略在世界模型中生成短期想象轨迹
    ↓
通过轨迹回报的梯度更新策略，并更新 critic
    ↓
回到 MuJoCo 继续采样
```

本仓库 `sm-train` 使用上述世界模型训练流程。正式评测和轨迹录制则将训练好的策略放回 MuJoCo 中执行。

## 3. 安装运行环境

下面的训练命令在具有 NVIDIA GPU 的 Linux 机器上运行，可以是远程服务器。若使用远程 GPU，先登录服务器，再执行安装、训练和评测命令；本地电脑用于查看网页、日志和下载视频。

项目要求 Python `>=3.12,<3.13`，具体依赖由 `uv.lock` 锁定。当前锁定配置使用 CUDA 13.0 版 PyTorch，因此服务器驱动需要支持相应运行环境。先查看设备：

```bash
nvidia-smi
```

### 3.1 克隆项目

在准备存放代码和训练结果的磁盘目录下执行：

```bash
git clone https://github.com/2860334406-tech/SuperMimic_pose_feedback.git
cd SuperMimic_pose_feedback
```

后文所有相对路径都以这个仓库根目录为起点。已有项目时直接进入原目录，不必重复克隆。

### 3.2 安装 uv 和 Python

未安装 uv 时，按 [uv 官方安装说明](https://docs.astral.sh/uv/getting-started/installation/) 执行：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv --version
uv python install 3.12
```

安装项目环境：

```bash
env -u PYTHONPATH uv sync --locked --group dev
```

这会创建 `.venv/` 并安装运行与测试依赖。后续用 `uv run --locked` 会自动使用该环境。`env -u PYTHONPATH` 用来避免其他 Python 环境（例如 ROS）影响当前命令。

如果系统盘容量较小，在安装 Python 和依赖前，将 uv 缓存放到数据盘。以下路径只是示例，需要换成实际磁盘路径，并在后续终端中保持相同设置：

```bash
export UV_CACHE_DIR=/personal_fast_storage/.uv-cache
export UV_PYTHON_INSTALL_DIR=/personal_fast_storage/.uv-python
```

### 3.3 检查 GPU 计算

```bash
env -u PYTHONPATH uv run --locked python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
assert torch.cuda.is_available(), "PyTorch 未检测到 CUDA GPU"
print("GPU:", torch.cuda.get_device_name(0))
print("算力:", torch.cuda.get_device_capability(0))
x = torch.arange(10, device="cuda")
print("计算结果:", x.sum().item())
PY
```

最后应输出 `计算结果: 45`。

## 4. 准备动作数据

### 4.1 下载 LAFAN1 的 G1 重定向数据

```bash
env -u PYTHONPATH uv run --locked hf download \
  lvhaidong/LAFAN1_Retargeting_Dataset \
  --repo-type dataset \
  --revision ce1572906efe6157840e8474d5a0d7aa87481e74 \
  --include 'g1/*.csv' \
  --local-dir data/raw/lafan1
```

这里下载固定版本的 40 条 G1 动作。原始文件位于 `data/raw/lafan1/g1/`。数据来源为 [LAFAN1 Retargeting Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)。

### 4.2 转成 50 Hz NPZ

```bash
MUJOCO_GL=disable env -u PYTHONPATH uv run --locked python \
  scripts/data/csv_to_npz_batch.py \
  --input-dir data/raw/lafan1/g1 \
  --output-dir data/npz/lafan1 \
  --input-fps 30 --output-fps 50 --workers 8
```

完成后会生成 40 个动作 NPZ 和 `manifest.csv`，总时长约 2.451 小时。检查输出中的失败数是否为 0。这个时长是参考动作数据的时长，不是训练耗时。

### 4.3 自己的动作需要什么格式

普通 G1 CSV 无表头，每行是一帧，共 36 列：

| 列（从 0 开始） | 内容 | 单位或顺序 |
|---|---|---|
| `0:3` | 根部位置 | 米 |
| `3:7` | 根部四元数 | **xyzw** |
| `7:36` | 29 个关节角度 | 弧度，顺序见 `data/motion_convert.py` 中的 `G1_JOINT_NAMES` |

输入帧率由 `--input-fps` 单独指定。转换后 NPZ 的身体四元数使用 **wxyz**，两种顺序不能混用。NPZ 包含 `fps`、`joint_pos`、`joint_vel`、`body_pos_w`、`body_quat_w`、身体速度等字段，不能将任意姿态数组改名为 `.npz` 后直接使用。

CSV 必须已经重定向到本项目的 G1 关节配置。原始人体视频需要先经过人体运动恢复和机器人重定向，才能进入这一步。

## 5. 检查环境与短训练

数据转换完成后运行：

```bash
MUJOCO_GL=disable env -u PYTHONPATH uv run --locked pytest tests -q
```

`passed` 表示通过，`skipped` 表示该测试没有执行。部分测试依赖 CUDA 以及 `data/npz/lafan1/walk1_subject1.npz`，因此只有跳过项不能证明整条训练链路可用。首次运行可能包含 GPU 内核编译，耗时会比后续更长。

正式长训练前，可用小规模配置检查位姿反馈训练与保存流程：

```bash
MUJOCO_GL=disable env -u PYTHONPATH uv run --locked sm-train \
  --run-name anchor_smoke \
  --data.npz-dir data/npz/lafan1 --data.clips walk1_subject1 \
  --env.anchor-obs --env.num-envs 128 --env.replay-capacity 200 \
  --wm-trainer.batch-size 256 --policy-trainer.batch-size 128 \
  --schedule.total-iterations 200 --schedule.checkpoint-interval 200
```

此命令用于检查运行链路。完成后应有 `logs/anchor_smoke/config.json` 和 `logs/anchor_smoke/ckpt_latest.pt`。正式实验使用新的运行名称，并采用下一节的训练配置。

## 6. 正式训练与后台运行

下面以单卡、LAFAN1 全部动作、随机种子 0、位姿反馈开启、最长训练 13 小时为例。运行名称 `lafan1_anchor_seed0_13h` 会用于后续所有评测与视频命令。

```bash
mkdir -p logs/lafan1_anchor_seed0_13h

nohup env -u PYTHONPATH MUJOCO_GL=disable PYTHONUNBUFFERED=1 \
  uv run --locked sm-train \
  --run-name lafan1_anchor_seed0_13h \
  --data.npz-dir data/npz/lafan1 \
  --seed 0 --env.anchor-obs \
  --env.num-envs 4096 \
  --wm-trainer.batch-size 2048 \
  --policy-trainer.batch-size 4096 \
  --schedule.total-iterations 320000 \
  --schedule.max-hours 13 \
  --schedule.checkpoint-interval 4000 \
  > logs/lafan1_anchor_seed0_13h/console.log 2>&1 < /dev/null &

echo $! > logs/lafan1_anchor_seed0_13h/train.pid
```

每次新实验使用新的 `--run-name`，并同步修改日志路径。相同名称会写入同一目录。

| 参数 | 含义 |
|---|---|
| `--env.anchor-obs` | 开启相对位姿反馈；默认关闭 |
| `--env.num-envs 4096` | 并行运行 4096 个仿真环境 |
| `--wm-trainer.batch-size 2048` | 每批世界模型训练窗口数量 |
| `--policy-trainer.batch-size 4096` | 每批策略想象轨迹的起点数量 |
| `--schedule.total-iterations 320000` | 环境迭代总上限；一次迭代让所有并行环境各前进一步 |
| `--schedule.max-hours 13` | 本次训练主循环的时间上限，按日志间隔检查；环境初始化不在该计时内 |
| `--schedule.checkpoint-interval 4000` | 每 4000 次迭代保存模型 |
| `--seed 0` | 随机种子 |

默认世界模型窗口长度为 8，策略想象长度为 32，更新数据比 UTD 为 0.4。UTD 表示每次环境迭代平均触发 0.4 次训练更新。

训练在迭代或时间上限先满足时停止。`nohup` 让任务在 SSH 断开后继续运行，前提是服务器实例保持运行。较小显存的设备需要调整并行环境数、两类 batch 和回放缓存容量，并记录调整后的配置。

## 7. 查看训练状态和曲线

### 7.1 日志和进程

```bash
tail -n 10 logs/lafan1_anchor_seed0_13h/console.log
ps -p "$(cat logs/lafan1_anchor_seed0_13h/train.pid)" -o pid,etime,stat,cmd
```

持续跟随日志：

```bash
tail -f logs/lafan1_anchor_seed0_13h/console.log
```

在这个查看窗口按 `Ctrl+C` 只会退出 `tail`。进程消失后，结合日志末尾和模型文件确认是否正常结束：

```bash
tail -n 30 logs/lafan1_anchor_seed0_13h/console.log
ls -lh logs/lafan1_anchor_seed0_13h/ckpt*.pt
```

`max_hours reached, stopping` 表示触发时间上限。程序正常退出时还会写入 `ckpt_latest.pt`；若最后的迭代数不是 4000 的倍数，不一定存在对应编号的 checkpoint。

### 7.2 终端日志中的指标

| 字段 | 中文含义 |
|---|---|
| `it` | 当前累计环境迭代数 |
| `h` | 本次训练主循环已运行的小时数 |
| `it/s` | 每秒环境迭代数 |
| `updates` | 累计训练更新次数 |
| `train_ret` | 训练期间已结束 episode 的平均归一化回报 |
| `wm nse1 / nse8` | 世界模型单步与 8 步预测误差指标 |
| `imag_ret` | 世界模型中想象轨迹的累计回报 |
| `surv` | 想象轨迹末端的平均存活权重 |
| `t sim/wm/pi` | 最近一个日志区间内，仿真、世界模型更新、策略更新的耗时 |

训练 episode 默认最长 10 秒，并有噪声和随机化；整段评测则从动作首帧开始运行完整动作。`train_ret` 和正式评测回报对应不同过程，`imag_ret` 也不能直接与二者比较。

### 7.3 TensorBoard

在训练机器的新终端中运行：

```bash
env -u PYTHONPATH uv run --locked tensorboard \
  --logdir logs/lafan1_anchor_seed0_13h/tb \
  --host 127.0.0.1 --port 6006
```

本机训练时打开 `http://127.0.0.1:6006`。远程训练时，在本地电脑另开终端建立转发，替换下面的服务器信息：

```bash
ssh -N -L 16006:127.0.0.1:6006 -p SSH端口 用户名@服务器地址
```

保持转发连接，在本地浏览器打开 `http://127.0.0.1:16006`。常用曲线有 `train/episode_norm_return`、`wm/online_nse_1step`、`wm/online_nse_8step`、`policy/imag_return_undisc` 和 `time/hours`。

仅查看最近的标量也可以使用：

```bash
env -u PYTHONPATH uv run --locked python scripts/tb_tail.py \
  --run-dir logs/lafan1_anchor_seed0_13h --last 3
```

## 8. 从 checkpoint 继续训练

`--resume` 会恢复网络、优化器、迭代数和适用的采样统计。它不会自动将当前终端传入的所有训练设置替换成 checkpoint 配置，续训时应按原 `config.json` 保持网络结构、位姿反馈开关和其他主要配置一致。

例如在已有模型基础上再运行最多 3 小时，写入新目录：

```bash
MUJOCO_GL=disable env -u PYTHONPATH uv run --locked sm-train \
  --run-name lafan1_anchor_seed0_resume \
  --resume logs/lafan1_anchor_seed0_13h/ckpt_latest.pt \
  --data.npz-dir data/npz/lafan1 \
  --seed 0 --env.anchor-obs --env.num-envs 4096 \
  --wm-trainer.batch-size 2048 --policy-trainer.batch-size 4096 \
  --schedule.total-iterations 320000 --schedule.max-hours 3
```

需要后台运行时，使用第 6 节的 `nohup` 包装方式。

- `total-iterations` 是累计总目标，不是新增迭代数；必须大于 checkpoint 内已经完成的迭代数才能继续更新。
- 新进程的 `time/hours` 从零开始，统计累计训练时间时需要相加。
- 回放缓存没有保存在 checkpoint 中，恢复后会重新采样并预热。
- 当前加载逻辑通过数据目录、动作列表、筛选清单和采样数组形状判断是否沿用采样统计；更换动作集应使用新目录，不要只覆盖原路径下的数据。
- 关闭位姿反馈时目标为 680 维，开启后为 689 维，两者权重不能通过简单切换开关直接续训。

## 9. 独立评测与漂移分析

### 9.1 评测全部 40 条 LAFAN1 动作

```bash
MUJOCO_GL=disable env -u PYTHONPATH uv run --locked python scripts/eval_benchmark.py \
  --ckpt logs/lafan1_anchor_seed0_13h/ckpt_latest.pt \
  --npz-dir data/npz/lafan1 --num-envs 40
```

脚本从 checkpoint 读取策略结构和位姿反馈开关。每条动作从首帧开始，关闭观测噪声、动作噪声、重置噪声和域随机化，运行到动作结束或触发终止条件。

结果保存在模型同目录的 `benchmark_ckpt_latest.json`。换成编号模型，例如 `ckpt_0004000.pt`，会得到 `benchmark_ckpt_0004000.json`。也可通过 `--out` 指定文件位置，其父目录需先存在。

| JSON 字段 | 含义 |
|---|---|
| `summary.clips` | 已评测动作数量 |
| `summary.failed` | 触发失败终止条件的动作数量 |
| `summary.mean_norm_return` | 各动作归一化回报的算术平均 |
| `summary.step_weighted_norm_return` | 按各动作实际执行步数加权的归一化回报 |
| `clips[].frames` | 参考动作完整帧数 |
| `clips[].steps` | 实际执行步数 |
| `clips[].norm_return` | 该动作的整段归一化回报 |
| `clips[].failed / fail_step` | 是否失败及失败步数，未失败时 `fail_step=-1` |
| `clips[].terms` | 各奖励项在实际执行步数上的均值 |

当前整段回报计算为：

```text
单动作归一化回报 = 实际获得的奖励总和 / (0.1 × 该参考动作完整帧数)
```

提前终止的动作也使用完整帧数作分母。`0/40` 失败表示所有动作运行到结尾，跟踪精度还需要结合回报、位置朝向误差和视频判断。

### 9.2 检查位置和朝向漂移

```bash
MUJOCO_GL=disable env -u PYTHONPATH uv run --locked python scripts/diag_drift.py \
  --ckpt logs/lafan1_anchor_seed0_13h/ckpt_latest.pt \
  --npz-dir data/npz/lafan1
```

默认结果保存在模型同目录的 `drift_ckpt_latest.json`，包括平均与末端水平位置误差、偏航误差、首次超过 0.3 米和 15° 的时间等。若只诊断部分动作，可添加 `--clips walk4_subject1 run2_subject1`。

## 10. 生成视频与交互查看

视频分两步生成：先用策略在物理仿真中执行并记录轨迹，再读取轨迹渲染画面。已经有轨迹时，可以改变截取区间或相机重复渲染。

### 10.1 录制策略轨迹

```bash
MUJOCO_GL=disable env -u PYTHONPATH uv run --locked python scripts/media/record_rollout.py \
  --ckpt logs/lafan1_anchor_seed0_13h/ckpt_latest.pt \
  --npz-dir data/npz/lafan1 \
  --clips walk4_subject1 run2_subject1 \
  --out logs/lafan1_anchor_seed0_13h/demo/rollouts.npz
```

动作名称是 NPZ 文件名去掉 `.npz`。每条动作从首帧运行，记录到动作结束或提前终止。轨迹文件包含 `robot_qpos`、`ref_qpos`、`names`、`lengths`、`norm_return` 等字段。当前录制脚本还有约 20,000 步的保护上限，录制特别长的动作时应检查输出步数。

### 10.2 渲染 20 秒视频

Linux 服务器先确认有 ffmpeg：

```bash
ffmpeg -version
```

Ubuntu 未安装时：

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

无桌面的 GPU 服务器使用 EGL 渲染：

```bash
MUJOCO_GL=egl env -u PYTHONPATH uv run --locked python scripts/media/render_rollout.py \
  --traj logs/lafan1_anchor_seed0_13h/demo/rollouts.npz \
  --out-dir logs/lafan1_anchor_seed0_13h/demo/videos \
  --clips walk4_subject1 run2_subject1 \
  --segments walk4_subject1:25:45 run2_subject1:5:25 \
  --fps 25 --width 960 --height 540
```

这会输出 `walk4_subject1.mp4` 和 `run2_subject1.mp4`。时间段为 `动作名:起始秒:结束秒`；上例各截取约 20 秒。截取范围应落在实际记录的动作长度内。相同目录中同名视频会被覆盖，保存不同片段时使用不同的 `--out-dir`。

| 画面内容 | 含义 |
|---|---|
| 实体机器人 | 训练好的策略在仿真中实际执行的动作 |
| 蓝色半透明机器人 | 参考动作 |
| 左上动作名称 | 当前动作片段的名称 |
| `normalized return` | 录制脚本保存的整条已执行轨迹回报，不是当前画面或截取 20 秒的回报 |
| `t = ... s` | 当前画面在原记录动作中的时间 |

当前录制脚本的回报分母使用**实际执行步数**，正式 benchmark 使用**完整动作帧数**。动作提前终止时，两者会不同，统计实验成绩时使用第 9 节的 JSON。

### 10.3 交互查看

在有桌面显示和可用 GPU 的运行机器上：

```bash
MUJOCO_GL=glfw env -u PYTHONPATH uv run --locked python scripts/play.py \
  --ckpt logs/lafan1_anchor_seed0_13h/ckpt_latest.pt \
  --npz-dir data/npz/lafan1 --clips walk4_subject1 \
  --num-envs 1 --viewer native
```

无桌面服务器可以改用 `--viewer viser`，配合 `MUJOCO_GL=disable`，根据终端显示的浏览器服务端口进行 SSH 转发后查看。需要保存可交付视频时，使用前两步的轨迹录制与离线渲染流程。

## 11. 测试新的动作

假设新动作已经转换为 G1 CSV，放在 `data/raw/custom/`，实际帧率为 30 Hz：

```bash
MUJOCO_GL=disable env -u PYTHONPATH uv run --locked python scripts/data/csv_to_npz_batch.py \
  --input-dir data/raw/custom --output-dir data/npz/custom \
  --input-fps 30 --output-fps 50 --workers 4

MUJOCO_GL=disable env -u PYTHONPATH uv run --locked python scripts/eval_benchmark.py \
  --ckpt logs/lafan1_anchor_seed0_13h/ckpt_latest.pt \
  --npz-dir data/npz/custom --num-envs 4 \
  --out logs/lafan1_anchor_seed0_13h/benchmark_custom.json
```

`--num-envs` 可以大于动作数，评测会取二者较小值。确认效果后，用第 10 节的命令替换数据目录和动作名称生成视频。

测试时使用固定 checkpoint，不更新策略。若要检验未见动作的跟踪能力，应确认这些动作未用于该 checkpoint 的训练。可跟踪的范围要通过测试确定，不是提供任意动作文件就一定能准确完成。

## 12. 常见问题

| 现象 | 处理方式 |
|---|---|
| `uv: command not found` | 执行 `source "$HOME/.local/bin/env"`，或重新打开终端 |
| Python 版本不匹配 | 安装 Python 3.12，并在仓库目录执行 `uv sync --locked --group dev` |
| CUDA 不可用或出现 GPU 架构错误 | 先检查 `nvidia-smi`，再执行第 3.3 节的实际 GPU 计算检查，核对驱动与锁定环境 |
| CUDA 显存不足 | 减少 `--env.num-envs`、`--wm-trainer.batch-size`、`--policy-trainer.batch-size`，必要时减少回放容量；使用新运行名记录配置 |
| 训练或评测在图形接口导入时失败 | 使用 `MUJOCO_GL=disable` |
| 视频渲染 EGL 初始化失败 | 检查服务器的 EGL/OpenGL 库与 GPU 图形能力；有桌面的机器可使用 `MUJOCO_GL=glfw` |
| 渲染找不到 ffmpeg | 安装 ffmpeg，并确认 `ffmpeg -version` 可执行 |
| `no clips under ...` 或 `clips not found` | 检查当前工作目录、NPZ 路径和不带扩展名的动作名称 |
| checkpoint 加载出现维度不匹配 | 检查网络结构和 `anchor_obs`；评测脚本会自动读取 checkpoint 配置，续训命令需要显式匹配 |
| 测试显示大量 `s` | 查看跳过原因，补齐 CUDA 和 LAFAN1 NPZ 后再运行 |
| 首次运行较长时间没有新日志 | 可能正在编译 GPU 内核；结合进程、GPU 使用情况和后续日志判断 |
| TensorBoard 提示缺少 TensorFlow、以 reduced feature set 运行 | 本项目使用的标量曲线仍可查看 |
| 进程已经消失 | 看日志末尾是否报错、达到时间上限或完成迭代，再检查 `ckpt_latest.pt` |
| 视频里的两个机器人不重叠 | 查看位置、朝向漂移及关节动作误差；蓝色参考与实体轨迹本来就是用来展示差异的 |

## 13. 保存代码和实验结果

代码与实验输出分别保存：

- **GitHub**：源代码、运行脚本、说明文档、`pyproject.toml` 和 `uv.lock`。
- **实验归档**：`config.json`、使用的 Git commit、checkpoint、评测 JSON、TensorBoard 日志、曲线、视频以及数据来源和转换参数。

`.gitignore` 默认忽略根目录下的 `data/`、`logs/`，以及 `.venv/`、`*.pt`、`*.npz` 等。GitHub 上能看到代码，不代表权重和数据也已备份。

每次实验可以记录当前代码版本：

```bash
git rev-parse HEAD
git status --short
```

若工作区还有未提交的代码修改，应一并保存这些改动，才能完整重现当次实验。修改代码后先查看差异，再按文件提交；在当前电脑的双远程配置中，`origin` 是社区仓库，`mygithub` 是自己的仓库。从本仓库新克隆的副本则默认使用 `origin`，以 `git remote -v` 的实际结果为准。

本 README 对应的两处代码修正：

1. `eval/benchmark.py`：每条动作按完整参考帧数归一化，避免提前终止后仅按已运行步数计算整段成绩。
2. `training/loop.py`：续训时根据动作库配置和采样数组形状决定是否恢复旧采样统计，更换动作库时重新初始化对应统计。

代码许可证见 [LICENSE](LICENSE)。
