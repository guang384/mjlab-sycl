# mjlab-sycl

[English](README.md) | 简体中文

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Release](https://img.shields.io/github/v/release/guang384/mjlab-sycl)](https://github.com/guang384/mjlab-sycl/releases)
[![CI](https://github.com/guang384/mjlab-sycl/actions/workflows/ci.yml/badge.svg)](https://github.com/guang384/mjlab-sycl/actions/workflows/ci.yml)

**在 Intel 核显/独显上跑 mjlab（MuJoCo Warp）的 PPO 训练——不需要 NVIDIA 显卡。**

[mjlab](https://github.com/mujocolab/mjlab) 的 `select_gpus` 只认 CUDA，在只有
Intel 显卡的机器上第 0 个迭代之前就会挂掉。`mjlab-sycl` 是一个伴生包：把**现有的**
mjlab 技术栈原样搬到 Intel iGPU/Arc 上训练，**不改你项目的任何源码**——装进 mjlab
项目的 venv、跑一条 overlay 命令，所有已注册的任务（例如
[microduck_rl](https://github.com/pollen-robotics/microduck_rl)）即可直接训练。

包含内容：

- **内置 warp 1.12.0 SYCL 后端**（补丁文件 + `warpsycl.dll`，严格增量——CUDA/CPU
  路径不动），由 `mjlab-sycl-install` 一键铺设
- **运行时补丁**：把 mjlab/mujoco_warp 的物理计算路由到 `sycl` 设备
- **无 barrier 的平铺核 + 原生 SYCL 核**：替换 mujoco_warp 最热的几个 kernel
- **train/bench/viewer/check 系列入口**：绕过 mjlab 的 CUDA-only GPU 选择，附带
  验证门和一键环境自检

## 快速上手（在你的 mjlab 项目 venv 中执行）

```powershell
# 1. 把本包装进 venv（editable 安装，改代码即生效；
#    如果 pip 全局配置了 target 重定向，先清掉 PIP_CONFIG_FILE/PIP_TARGET，
#    或者像下面这样加 --isolated）
<project>\.venv\Scripts\python.exe -m pip install --isolated --no-deps -e <path-to-mjlab-sycl>

# 2. 铺设 warp SYCL 后端（加 --warmup <TASK> 可顺带预编译内核）
<project>\.venv\Scripts\mjlab-sycl-install.exe --warmup <TASK>

# 3. 环境预检（只读，8 项检查，每项带修复指引）
<project>\.venv\Scripts\mjlab-sycl-check.exe

# torch 必须是 +xpu 轮子——直接 `pip install torch` 装的是 CPU 版，
# PPO 会静默慢 3 倍且毫无报错。只有当上面的检查 FAIL 才需要执行：
<project>\.venv\Scripts\python.exe -m pip install "torch==2.9.1+xpu" --index-url https://download.pytorch.org/whl/xpu
```

安装已发布版本（装进你的 **mjlab 项目 venv**，`--no-deps` 是因为依赖版本由项目
自己的 lockfile 决定）：

```powershell
<project>\.venv\Scripts\python.exe -m pip install --isolated --no-deps "git+https://github.com/guang384/mjlab-sycl@v0.3.0"
<project>\.venv\Scripts\mjlab-sycl-install.exe --warmup <TASK>
<project>\.venv\Scripts\mjlab-sycl-check.exe
```

（`v0.3.0` 换成目标版本号即可；发布页的 `.whl` 安装方式相同。本包不在 PyPI 发布。）

然后训练任意已注册任务。`<TASK>` 是**你的** mjlab 项目注册表里的任务名（任务包通过
`register_mjlab_task` 注册），用 `mjlab-sycl-train --list-tasks` 查看全部：

```powershell
# 先冒烟（64 环境、5 个迭代）
<project>\.venv\Scripts\mjlab-sycl-train.exe <TASK> --num-envs 64 --max-iterations 5

# 正式训练（4096 环境）
<project>\.venv\Scripts\mjlab-sycl-train.exe <TASK> --num-envs 4096 --max-iterations 1000

# 例如 microduck_rl 的行走任务：
#   mjlab-sycl-train.exe Mjlab-Velocity-Flat-MicroDuck --num-envs 4096 --max-iterations 1000
```

新项目一键安装：`scripts/setup_project.ps1 -Repo <项目路径>`。

**首次运行（一次性）**：内核模块首次使用时 JIT 编译（本机实测约 3.5 分钟，之后走
缓存）——入口会提前打印提示；第 2 步的 `--warmup` 把这个成本挪到安装期。从全新
克隆到第一个训练步约 **10 分钟**（大头是 `uv sync` 和一次性 JIT）。

## 常见陷阱

| 陷阱 | 现象 | 解决 |
|---|---|---|
| 装成 torch CPU 版 | 不报错，但 PPO 更新慢 ~3 倍 | 装 `torch==2.9.1+xpu`（PyTorch XPU 专用源，见上） |
| 训练后台开着视频/其它 GPU 程序 | 吞吐低 10–45% | 关掉它们；train/bench 启动时会检测并警告 |
| `uv sync` 后训练报 overlay 不同步 | 入口中止并给出修复命令 | 重跑 `mjlab-sycl-install`（uv 重装 warp 会抹掉 overlay） |
| 首次运行"卡住"几分钟 | 内核 JIT 编译中 | 正常现象；用 `--warmup` 预编译可免 |
| 运行时报 WinError 127 | sycl8.dll 版本太旧 | `pip install "intel-sycl-rt==2025.3.3" "dpcpp-cpp-rt==2025.3.3"`（~50MB，无需安装 oneAPI 工具链） |

## 性能

Intel Arc 130T（Lunar Lake 核显）、microduck velocity 任务、4096 环境：
端到端 **~19–22k env-steps/s（约 5.5–6.5 秒/迭代）**，同栈 warp-cpu 设备约 258
（会话/GPU 状态带来 ±20% 方差——比较性能请用同一会话内的配对测试）。所有实测
数据、误差条与每项优化的结论见 [`docs/performance.md`](docs/performance.md)。

## 可视化工具

| 工具 | 用途 |
|---|---|
| `python -m mjlab_sycl.train_viewer` | 训练过程实时开窗显示（镜像 0 号环境，约 8 次刷新/秒） |
| `python -m mjlab_sycl.kview` | K 个训练环境的 K 只鸭并排显示 |
| `python -m mjlab_sycl.cpu_replay` | 在 CPU MuJoCo 上约 50 Hz 平滑回放 checkpoint |
| `python -m mjlab_sycl.play` | 真仿真回放 checkpoint |

## 文档索引

- [README-SYCL-TRAINING.md](README-SYCL-TRAINING.md)（英文）：完整手册——需求、
  环境变量、验证门、踩坑记录、已知限制
- [docs/performance.md](docs/performance.md)：实测性能档案（唯一数据来源）
- [CHANGELOG.md](CHANGELOG.md)：版本历史

## 反馈

Bug、其它 Intel 显卡的实测数据、功能建议：[Issues](https://github.com/guang384/mjlab-sycl/issues)
/ [Discussions](https://github.com/guang384/mjlab-sycl/discussions)。
