# VFI Hard Miner

面向游戏插帧模型的离线困难样本挖掘工具。它扫描一个游戏中的连续帧，用当前模型重建中间帧，筛选“数据有效、局部错误明显、模型理论上可解”的样本，并把相邻或重叠的困难 triplet 合并为连续原始帧片段。

## 当前基线

- 输入为 RGB `[0,1]` 的 `img0/img1`，固定插值时刻 `t=0.5`。
- 网络返回低分辨率 backward `flow_t0/flow_t1` 和 sigmoid 后的 `mask0/mask1`。
- flow 上采样及像素尺度修正、backward warp 和两级 mask 融合默认在加速设备上以 FP32 完成（`runtime.reconstruction: "auto"`，启动时探测 `grid_sample` 能力）；设 `"cpu"` 回退 CPU FP32 参考实现（校准基线，与旧版本逐操作一致）。加速设备路径与 CPU 参考仅有 fp32 级舍入差异。
- GT 决定“是否做错”；原生分辨率 RGB/亮度差、Sobel 边缘差、局部连通区域、多尺度窗口和小面积高峰共同生成 `p_wrong`。
- `p_wrong` 保留原始 GT 误差；贴边、端点近似静态且文字/图标式高边缘密度的区域会得到较低 `priority_weight`，另算 `mining_p_wrong` 参与入选，中央人物/武器结构和非 UI 候选仍保留名额。
- 可选 teacher 和 best-of-warp 分支只估计 `p_solvable`，teacher 做错不会把源 triplet 判为无效。
- 主挖掘和最终诊断均可使用 CPU、CUDA 或多个独立 `torch_npu` worker；NPU 模式不使用 DDP/HCCL。
- 最终输出包含连续原始帧、诊断拼图、完整 JSONL manifest、分段记录和可恢复 SQLite 状态。默认每个合并 segment 独占一个叶目录并保留原文件名；`finalize` 会按配置的命名规则和 stride 重扫落盘结果。训练器必须把每个叶目录视为一个独立视频，不能跨叶目录拼 triplet。

FLIP、CGVQM 和 DINOv2 的源码/权重目前只是离线储备，方便后续对照实验或人工复核研究。当前基线运行时不会导入它们，它们也不参与 `p_wrong`、`p_solvable` 或 `accept/review/reject` 判定；缺少这些模型不会触发另一套静默评分逻辑。

## 运行前提

目标平台是 Ascend A3/910B。driver、firmware、CANN 9.0、PyTorch 和 `torch_npu` 必须由目标服务器预装并保持版本匹配，本项目不会安装或替换这些系统组件。

项目级依赖见 `requirements.lock`（numpy、Pillow、scipy）。scipy 用于评分阶段连通区域等热点的 C 级向量化，目标机可直接 `python -m pip install scipy==1.13.1`（Python 3.13 环境请改装 1.14.1+）。

先复制并修改配置：

```bash
cp configs/example.json configs/my_game.json
```

至少需要完成以下修改：

- `data.root` 指向一个游戏的连续帧根目录。
- `model.factory` 指向实际适配器的 `module:function`。
- `model.checkpoint` 指向已经存在的当前模型权重。
- `runtime.run_dir` 使用本次实验独立目录。
- NPU 生产配置使用 `backend: "npu"`、设备 `0..7`、`workers: 8` 和 `precision: "float32"`。吞吐调优项：`runtime.postproc_workers`（CPU 评分线程数，0 为自动）、`runtime.postproc_buffer_mb`（每 worker 在途重建结果的内存预算，默认 1024 MB）、`runtime.decode_cache_mb`（uint8 帧缓存上限）、`runtime.reconstruction`（重建设备，默认 `"auto"`）和 `model.batch_size`。模型 batch 与重建/CPU 后处理微批次相互独立，大 batch 不会再把多个完整的全分辨率结果无限堆入 Future 队列。

工具不会猜测 checkpoint key、网络输出顺序或 `mask0` 方向；接入契约见 `docs/model_adapter.md`。  
配置字段的详细说明见 `configs/example.jsonc`（JSONC 格式，VS Code 可渲染内联注释，运行时不能直接加载）：

```bash
cp configs/example.json configs/my_game.json
# 参考 configs/example.jsonc 了解每个字段的含义后再修改
```

## 添加新模型

`src/vfi_hard_miner/models/` 目录提供基于文件名的短名称机制：将模型文件放入该目录，即可在配置中直接用文件名引用，无需修改其他任何代码。

**快速上手**

```bash
# 1. 复制模板，以目标模型命名
cp src/vfi_hard_miner/models/_template.py src/vfi_hard_miner/models/my_model.py

# 2. 填入网络结构（__init__、forward）和工厂函数（create_model）

# 3. 在配置中引用
#    "model": { "factory": "my_model", "checkpoint": "ckpts/my_model.pth", ... }
```

原有的完整路径写法保持向后兼容：

```json
"model": { "factory": "my_project.adapter:create_model", ... }
```

**内置模型**

| 名称 | 文件 | 说明 |
|------|------|------|
| `unet` | `models/unet.py` | 轻量 UNet，两帧拼接输入，1/4 分辨率双向 flow + 混合 mask 输出 |

查看当前全部可用模型：

```python
from vfi_hard_miner.models import list_models
print(list_models())   # ['unet', ...]
```

**两种调用方式**

通过 `ModelAdapter`（pipeline 正常路径，推荐）：

```python
adapter = ModelAdapter.from_config(config.model, device=device)
outputs = adapter(img0, img1)   # 返回 ModelOutputs dataclass
```

直接调用模型（自定义评估脚本等）：

```python
import torch
from vfi_hard_miner.models.unet import create_model

model = create_model(checkpoint="ckpts/unet.pth", device=torch.device("cuda"))
outputs = model.infer(img0, img1)   # 自动 inference_mode，返回 dict
```

`infer()` 与 `forward()` 的区别：`infer()` 由模型类直接暴露，内置 `torch.inference_mode()` 包裹，适合独立脚本调用；`forward()` 是 PyTorch 内部接口，由 `ModelAdapter` 调用。通过 adapter 使用时无需手动调用 `infer()`。

## 典型流程

开发或准备机安装项目后：

```bash
python -m pip install -e . --no-deps
python -m pytest
vfi-hard-miner --help
```

在目标机先用独立进程探测 8 张 NPU，并把结果保存到配置的 `runtime.run_dir`。`finalize` 会把该文件写入运行元数据：

```bash
mkdir -p runs/game_name
python scripts/probe_ascend.py --require-devices 8 --strict \
  > runs/game_name/runtime_probe.json
```

随后执行：

```bash
vfi-hard-miner index --config configs/my_game.json
vfi-hard-miner mine --config configs/my_game.json
# 仅当配置了 teacher 时执行：
vfi-hard-miner teacher --config configs/my_game.json
vfi-hard-miner finalize --config configs/my_game.json
```

也可以用 `vfi-hard-miner run --config configs/my_game.json` 串行执行 index、main、可选 teacher 和 finalize。没有实际模型时可使用项目 mock 适配器和合成数据做 smoke test，但它只能验证流水线，不能代表困难样本质量。

需要在长时间运行中提前查看已完成视频的原始困难帧时，可设置
`output.materialize_strategy: "per_video"`。该模式要求
`output.layout: "segment_relative"` 且首版不支持 teacher；`mine` 会在一个视频的
全部 chunk 完成后，将其原子物化到
`data.root/.vfi_hard_miner_staging/<execution_id>/hard_case`。中间目录只供人工查看，
`finalize` 会复用这些帧并在诊断与 manifest 完成后统一切换正式输出目录。

## Execution snapshot

`index` 首次运行会在 `runtime.run_dir/execution.snapshot.json` 写入不可变执行快照，包含：

- 配置哈希和索引内容摘要；索引记录包含帧路径、编号、文件大小和 `mtime_ns`。
- 当前模型及可选 teacher checkpoint 的绝对路径、字节数和 SHA-256。
- 可定位时的模型工厂模块源文件 SHA-256。
- `vfi_hard_miner` Python 源码树摘要。
- 由上述内容生成的 `execution_id`。

索引记录、任务和结果都会携带 `execution_id`，SQLite 状态文件也按配置哈希与 execution ID 隔离。若数据索引、checkpoint、工厂源文件或 miner 源码发生变化，工具会拒绝在已有 `run_dir` 中混用旧结果；请为新实验使用新的 `runtime.run_dir`。该快照记录文件元数据和已列出的源码入口，不等同于逐像素哈希全部帧或递归哈希用户模块的所有传递依赖。

## 8 卡最终诊断

困难片段合并后，`finalize` 会自动启动诊断阶段，无需单独的 CLI 命令：

- 仅为入选困难中心生成诊断图；`output.save_review=true` 时也包括 review 样本。
- 任务会动态切成足够多的小块，最多启动 `runtime.workers` 个独立设备进程；8 卡配置下最多一张卡一个 worker，任务不足时自动减少 worker 数。
- 每个 worker 只绑定自己的 NPU，并只加载一次当前模型；图片解码、NPU 推理+重建（默认在卡上，见 `runtime.reconstruction`）和 CPU 评分/拼图通过预取流水线重叠，多个 CPU 后处理线程（`runtime.postproc_workers`）并发消费重建结果。
- 诊断任务使用独立 SQLite 状态、后台 lease heartbeat、attempt-scoped part 和 artifact 目录。过期、失败或产物缺失的任务可在重跑时恢复，过期 attempt 不会覆盖 winning attempt。

最终诊断仍要求经过校准的 FP32 基线；Windows/CUDA 结果只能用于通用功能与数值检查，不能代替 Ascend 910B 上的算子和吞吐验证。

## 离线包状态

离线资源校验分为两层：

```bash
# 校验已登记资源、两个 lockfile 和资源路径覆盖；允许实际 VFI checkpoint 尚未到位
vfi-hard-miner verify-bundle --project-root . --resources-only

# 先生成只含 manifest 登记资源、审计元数据和两个 lockfile 的依赖包
vfi-hard-miner build-bundle --project-root . --resources-only \
  --output dist/vfi-hard-miner-resources-aarch64.tar.gz

# 生产包完整校验；同时检查必需目录及 configs/**/*.json 引用的 checkpoint
vfi-hard-miner verify-bundle --project-root .
```

`build-bundle --resources-only` 用于在用户 checkpoint 交付前先搬运固定的 third-party 资源和依赖锁文件；归档只纳入 `third_party/manifest.json` 明确登记的外部文件以及必要审计元数据，不会因工作区中残留旧 wheel、解压源码媒体或其他文件而静默带入。归档仍带逐文件 inventory，但不包含项目源码、配置或 VFI checkpoint，不能直接运行挖掘。

完整校验要求每个配置引用的 current/teacher checkpoint 都存在、位于项目内、被纳入归档范围，并登记在 `third_party/manifest.json`。因此当前 checkout 的 `ckpts/current/model.pth` 缺失会按设计阻止完整校验和完整 `build-bundle`；先放入并登记实际 checkpoint，或把所有生产配置改为真实的已登记路径。

完整准备、打包和目标机安装流程见 `docs/offline_deployment.md`。
