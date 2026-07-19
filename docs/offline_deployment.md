# Ascend A3/910B + CANN 9.0 离线部署

本文描述当前实现的生产离线流程：在联网准备环境收集 aarch64 资源和实际 VFI checkpoint，完成两级校验并生成带全文件 inventory 的归档；再在已安装匹配 CANN/PyTorch/`torch_npu` 的目标服务器断网安装和运行。

## 1. 系统组件边界

以下组件不进入项目离线包，由目标服务器预装并统一维护：

- Ascend driver 和 firmware
- CANN 9.0
- 与 CANN 匹配的 Python、PyTorch 和 `torch_npu`

项目脚本只读取和记录环境信息，不安装、升级或替换这些组件。`requirements.lock` 也刻意不包含 `torch` 和 `torch_npu`。

在目标机使用独立进程探测环境，避免协调进程提前导入 `torch_npu`：

```bash
mkdir -p runs/game_name
python scripts/probe_ascend.py --require-devices 8 --strict \
  > runs/game_name/runtime_probe.json
```

这里的目录必须和配置中的 `runtime.run_dir` 一致。`--strict` 检查 NPU 可用性，`--require-devices 8` 检查可见卡数；脚本默认还采集只读的 `npu-smi info`。最终 manifest 的运行元数据会包含这个 `runtime_probe.json`。

## 2. 联网准备环境

先在目标服务器确定 Python ABI、glibc、CANN、PyTorch 和 `torch_npu` 组合，再在具有相同 Python ABI/OS 的联网 aarch64 环境准备 wheels。若只能在 x86_64 联网机下载纯二进制 wheel，可交叉下载：

```bash
python scripts/download_aarch64_wheels.py \
  --python-version 311 \
  --requirements requirements.lock \
  --destination third_party/wheelhouse/linux-aarch64

python scripts/download_aarch64_wheels.py \
  --python-version 311 \
  --requirements requirements-build.lock \
  --destination third_party/wheelhouse/linux-aarch64
```

把 `311` 换成目标 CPython ABI。带本地编译扩展且没有匹配 wheel 的包，必须在兼容的 aarch64 构建机生成；不得把 x86_64 wheel 放进 `linux-aarch64`。
下载脚本只负责下载，不会清空目标目录或自动猜测许可证。每个准备进入交付包的 wheel 都必须用 `record_offline_resource.py` 登记；旧 ABI、旧版本或未登记 wheel 即使仍留在目录中也不会进入归档，建议在发布前移出以免人工误用。

联网准备工作区布局为：

```text
third_party/
├─ downloads/                     # 固定 revision 的源码归档
├─ src/{flip,cgvqm,dinov2}/       # 离线源码快照
├─ weights/{cgvqm,dinov2}/        # 可选研究模型权重
├─ wheelhouse/linux-aarch64/       # 目标 ABI wheels
├─ licenses/
└─ manifest.json

ckpts/
├─ current/                        # 当前 VFI 模型
└─ teacher/                        # 可选 teacher VFI 模型
```

`src/{flip,cgvqm,dinov2}` 是便于联网侧检查的解压工作副本，不自动进入离线包。归档采用闭集：`third_party/` 和 `ckpts/` 下只打入 manifest 明确登记的普通文件，另加 `third_party/README.md`、`third_party/manifest.json` 和依赖锁文件。固定 revision 的源码归档已经登记在 `downloads/`，目标机如需阅读源码可从这些归档解压。未登记的演示视频、EXR、测试图片、旧 wheel 或其他 checkpoint 不会静默进入包。

### FLIP、CGVQM、DINOv2 的当前定位

这三套资源目前只是离线储备，用于未来对照实验或复核研究。生产基线代码不会导入它们，也不会用它们计算 `p_wrong`、`p_solvable` 或改变 `accept/review/reject`。当前基线依赖的是项目自身的原生分辨率 RGB/亮度/边缘/局部结构评分；teacher 若配置，只用于估计当前错误是否可解。

因此，把这些资源放进 bundle 不表示它们已经接入运行时。后续若要启用，必须显式增加配置、依赖和经过校准的评分路径，不能因资源存在而静默改变判分。

## 3. 放入并登记实际 checkpoint

`configs/**/*.json` 中每个非空的 `model.checkpoint` 和 `teacher.checkpoint` 都会被完整校验。checkpoint 必须：

1. 位于项目根目录内；
2. 实际存在；
3. 位于归档 include root 内；
4. 以相同相对路径登记在 `third_party/manifest.json`。

例如：

```bash
cp /secure/export/current-model.pth ckpts/current/model.pth

PYTHONPATH=src python scripts/record_offline_resource.py \
  ckpts/current/model.pth \
  --name current-vfi-model \
  --kind weight \
  --source 'internal://vfi/current-model' \
  --version 'export-2026-07-18' \
  --license Proprietary
```

如果配置了 teacher，对 `ckpts/teacher/...` 执行同样操作。准备内部模型时请把 `source`、版本和许可证值替换成组织内真实且可审计的信息。

当前仓库只有 `ckpts/current/.gitkeep` 和 `ckpts/teacher/.gitkeep`，而 `configs/example.json` 引用了 `ckpts/current/model.pth`。所以当前 checkout 可以做“资源清单校验”，但完整校验和正式打包会按设计失败。必须先提供并登记该文件，或者修改 `configs/` 下所有 JSON，使它们引用真实、已登记的 bundle 内 checkpoint。

## 4. 两级离线校验

### 4.1 资源清单校验

实际 VFI checkpoint 尚未交付时，先验证非空 manifest 中已经登记的资源：

```bash
PYTHONPATH=src python -m vfi_hard_miner verify-bundle \
  --project-root . \
  --resources-only
```

`--resources-only` 与资源打包使用同一个项目校验入口：检查两个 lockfile、manifest 的安全位置和每条记录的归档范围、文件存在性、字节数及 SHA-256。它有意跳过生产源码目录完整性和 `configs/**/*.json` checkpoint 覆盖检查，因此成功表示“资源先行包具备打包条件”，不表示项目已经可以运行或生成完整生产包。

### 4.2 生产包完整校验

实际配置与 checkpoint 就绪后执行：

```bash
PYTHONPATH=src python -m vfi_hard_miner verify-bundle \
  --project-root .
```

完整校验除 manifest 哈希外，还检查必需项目根、每个已登记资源是否会进入归档，以及所有配置引用的 current/teacher checkpoint 是否存在并已登记。任何占位 checkpoint、绝对外部路径、遗漏登记或哈希不一致都会阻止打包。

## 5. 生成传输归档

### 5.1 checkpoint 到位前的资源先行包

用户 VFI checkpoint 尚未交付时，可以先生成只包含 manifest 登记资源、`third_party` 审计元数据、`requirements.lock` 和 `requirements-build.lock` 的可搬运归档：

```bash
PYTHONPATH=src python -m vfi_hard_miner build-bundle \
  --project-root . \
  --resources-only \
  --output dist/vfi-hard-miner-resources-aarch64.tar.gz
```

该命令执行与 `verify-bundle --resources-only` 相同的项目校验，并同样生成、读回验证 `BUNDLE_INVENTORY.json`。它用于提前把固定源码归档、候选模型储备、许可证和 aarch64 wheels 搬到隔离区；本地解压源码树和未登记文件不会进入包。它不包含项目 `src/`、`configs/`、`ckpts/` 或运行脚本，因此不是可执行的生产包，也不证明实际 VFI checkpoint 已就绪。

### 5.2 完整生产包

实际 checkpoint 登记且完整校验通过后，任选一个入口：

```bash
PYTHONPATH=src python -m vfi_hard_miner build-bundle \
  --project-root . \
  --output dist/vfi-hard-miner-aarch64.tar.gz
```

或：

```bash
PYTHONPATH=src python scripts/prepare_offline_bundle.py \
  --project-root . \
  --output dist/vfi-hard-miner-aarch64.tar.gz
```

输出路径必须在本次归档的所有 include root 之外；项目根下的 `dist/` 符合当前默认布局。两种归档模式都会：

- 拒绝缺失必需根和符号链接；
- 先完成对应模式的校验；
- 对 `third_party/` 与 `ckpts/` 使用 manifest 闭集，只纳入已登记资源；
- 排除常见 Python、pytest、mypy、ruff、notebook、构建和虚拟环境缓存；
- 为归档中的每个普通 payload 文件记录字节数和 SHA-256；
- 写入 `BadCaseSelect/BUNDLE_INVENTORY.json`；
- 重新打开临时归档，逐文件核对 member、大小和 SHA-256；
- 校验成功后才原子替换最终 `.tar.gz`。

资源 manifest 用于外部资源的来源、版本、许可证和固定哈希；`BUNDLE_INVENTORY.json` 则覆盖归档中的每个普通 payload 文件，两者用途不同。

## 6. 目标机断网安装

```bash
tar -xzf vfi-hard-miner-aarch64.tar.gz
cd BadCaseSelect

python -m pip install --no-index \
  --find-links third_party/wheelhouse/linux-aarch64 \
  -r requirements-build.lock

python -m pip install --no-index \
  --find-links third_party/wheelhouse/linux-aarch64 \
  -r requirements.lock

python -m pip install --no-index --no-deps --no-build-isolation -e .
```

然后在断网状态再次执行完整校验：

```bash
vfi-hard-miner verify-bundle --project-root .
python scripts/probe_ascend.py --require-devices 8 --strict
```

先完成 CPU smoke test 和单卡 NPU Eager FP32 数值核对，再扩展至 8 卡。最终诊断明确要求 `runtime.precision="float32"`；不要在基线尚未校准时启用混合精度、TorchAir 或 aclgraph。

## 7. Execution snapshot 与实验隔离

实际 checkpoint 必须在 `index` 前就位，因为 `index` 会创建 `runtime.run_dir/execution.snapshot.json`。快照包含：

- 配置哈希；
- 索引内容摘要（路径、编号、大小、`mtime_ns` 等索引字段）；
- current/teacher checkpoint 的路径、大小和 SHA-256；
- 可定位时的 current/teacher factory 源文件 SHA-256；
- miner Python 源码树摘要；
- 派生的 24 位 `execution_id`。

任务状态库名、任务 payload、index 和结果 manifest 都绑定该 execution ID。再次对同一 `run_dir` 执行 `index` 时，只要数据索引、checkpoint 或记录的源码摘要发生变化，程序就会拒绝复用旧结果。开始新实验时应使用新的 `runtime.run_dir`，并保留旧目录用于审计。

快照不会逐像素哈希全部游戏帧，也不会递归追踪用户 factory 导入的所有模块；若需要更强的数据治理，应在上游额外冻结数据集版本或内容清单。

## 8. 8 卡挖掘与诊断流水线

推荐生产配置：

```json
{
  "runtime": {
    "backend": "npu",
    "devices": [0, 1, 2, 3, 4, 5, 6, 7],
    "workers": 8,
    "precision": "float32"
  }
}
```

执行顺序：

```bash
vfi-hard-miner index --config configs/my_game.json
vfi-hard-miner mine --config configs/my_game.json
# 配置 teacher 时：
vfi-hard-miner teacher --config configs/my_game.json
vfi-hard-miner finalize --config configs/my_game.json
```

main 与可选 teacher 阶段使用独立设备 worker 和各自的 SQLite lease 状态。`finalize` 合并困难区间后，会自动为 selected hard center（以及可选 review）建立诊断任务；诊断阶段没有独立 CLI 子命令。

诊断任务按视频排序后动态切块，目标是让 8 卡配置拥有足够任务并行度；实际 worker 数为 `min(runtime.workers, 当前任务数)`。每个进程只绑定一张 NPU、加载一次当前模型，并利用解码预取和单 CPU 后处理线程重叠 NPU 推理、FP32 原图重建、局部评分和拼图生成。这里不使用 DDP、HCCL 或跨卡梯度同步。

状态恢复使用 execution-scoped SQLite、后台 lease heartbeat 和 attempt fencing。每次尝试写入独立的 part/artifact 目录；完成记录会绑定 winning task/attempt，缺失或无效产物在重跑时重新排队，旧 attempt 不会进入最终 `diagnostic_results.jsonl`。

最终输出写入：

- `data.root/<hard_case_dir>/`：连续原始帧；默认 `segment_relative` 布局让每个合并 segment 独占叶目录并保留原 basename，`finalize` 会按配置的命名规则和 stride 重扫并拒绝空效 segment，训练器必须把每个叶目录当作独立视频且不得跨目录组 triplet；
- `data.root/<visualization_dir>/`：`img0 / GT / prediction / img1` 与局部错误诊断拼图；
- `data.root/<manifest_name>`：完整结果及 `selected`、`covered_by_segment`、`contributed_to_segment` 语义；
- `runtime.run_dir/segments.json`：合并分段；
- `runtime.run_dir/diagnostic_results.jsonl`：winning diagnostic 记录。

发布阶段使用 staging、ownership marker 和 finalize lock，将帧目录、可视化目录、manifest、segments 与 current marker 作为一个可回滚发布批次处理。
