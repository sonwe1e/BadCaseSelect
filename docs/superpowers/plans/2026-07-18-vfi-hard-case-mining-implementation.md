# 游戏插帧困难样本挖掘实施计划

日期：2026-07-18

关联设计：`docs/superpowers/specs/2026-07-18-vfi-hard-case-mining-design.md`

## 1. 首版交付边界

首版交付一个可离线部署的 Python 包和命令行工具，能够：

1. 按可配置文件名规则扫描一个游戏下的全部连续帧，建立稳定索引。
2. 通过用户提供的 Python 模型工厂加载当前模型和可选 teacher。
3. 在 CPU、CUDA 或 `torch_npu` 上执行固定 H×W、固定 batch 的中点插帧推理。
4. 在 CPU FP32 中按明确的 backward-flow 契约重建 `warp0`、`warp1`、`warp_blend` 和 prediction。
5. 通过有效性、适用范围、局部错误和可解性四类证据筛选困难样本。
6. 合并同一视频内重叠或相邻的失败 triplet，输出连续原始帧、诊断图和 JSONL manifest。
7. 使用 SQLite WAL、租约和幂等 sample ID 支持 8 个独立 NPU worker 断点续跑。
8. 校验独立 `third_party/` 资源目录并生成可供断网服务器使用的压缩包。

首版不内置用户的实际网络结构，不猜测 checkpoint 加载方式，也不伪造 DINO/CGVQM 权重地址。项目通过稳定的模型工厂协议接入实际网络；未提供模型时，使用合成 mock 模型完成完整测试。

## 2. 包结构

```text
src/vfi_hard_miner/
├─ cli.py                 # index、mine、finalize、probe、verify-bundle
├─ config.py              # JSON 配置解析、默认值和严格校验
├─ schemas.py             # 帧、triplet、区域、判定和产物数据结构
├─ indexing.py            # 文件名解析、视频分组、稳定 sample ID
├─ image_io.py            # RGB [0,1] 解码和原子图像写入
├─ reconstruction.py      # flow 上采样、缩放、backward warp、mask 融合
├─ model_adapter.py       # Python 工厂协议与 CPU/CUDA/NPU 设备抽象
├─ scoring.py             # 多尺度局部残差、边缘和候选区域
├─ gates.py               # validity、in_scope 与灰区判定
├─ diagnosis.py           # 原因标签、分支误差和 p_solvable
├─ segments.py            # 区间合并和强制断点
├─ visualization.py       # 诊断拼图与热图
├─ manifest.py            # 原子 JSONL part 与最终确定性合并
├─ state.py               # SQLite WAL、任务租约和恢复
├─ worker.py              # 单设备常驻模型 worker
├─ pipeline.py            # 三阶段协调器
├─ runtime.py             # backend 探测、版本信息、spawn 入口
└─ offline.py             # third_party 哈希、锁文件和归档校验
```

辅助内容：

```text
configs/example.json
examples/mock_model.py
scripts/prepare_offline_bundle.py
scripts/probe_ascend.py
third_party/{wheelhouse,src,weights,licenses}/
tests/
```

## 3. 模型接入契约

配置使用 `module:function` 指向模型工厂。每个 worker 绑定设备后调用一次工厂；工厂返回 `torch.nn.Module` 或实现同等调用协议的对象。调用输入为 `[B,3,H,W]` 的 `img0` 和 `img1`，值域 `[0,1]`。

模型输出允许命名字典或长度为四的序列，但归一化后必须得到：

```text
flow_t0: [B,2,h,w]
flow_t1: [B,2,h,w]
mask0:   [B,1,h,w]
mask1:   [B,1,h,w]
```

配置必须明确：

- `mask0_role`：`warp0_weight` 或 `warp1_weight`；
- `align_corners`；
- `padding_mode`；
- flow 单位是否为网络输入像素；
- checkpoint 路径和可选工厂参数。

生产模式禁止根据每个样本的 GT 自动选择 mask0 方向。启动 smoke test 会验证 shape、有限值、mask 范围和重建公式。

## 4. 实施顺序

### 阶段 A：核心契约与可重复重建

- 建立包、CLI、JSON 配置和数据结构。
- 实现固定 frame-digit 或命名正则解析；跳过输出目录和隐藏目录。
- 生成 `video_id`、有序帧、triplet 和 SHA-256 稳定 ID。
- 实现 CPU FP32 flow resize/尺度修正、`grid_sample` backward warp 与两层 mask 融合。
- 为恒等 flow、已知平移、两个 mask0 方向、mask1 endpoint-copy 编写数值测试。

验收：合成测试能够逐像素验证重建语义；错误 shape、NaN、未知 mask 方向必须快速失败。

### 阶段 B：局部错误判定

- 计算 RGB、亮度、Sobel 边缘、GT-only/pred-only 边缘与局部结构残差。
- 通过多尺度平均/最大池化和阈值连通区域生成候选框。
- 统计局部最大值、高分位、top-area mean、区域能量和边界长度。
- 用端点差异、直方图突变、重复帧、时间对称性和 flow 出界/不连续比例实现保守门控。
- 通过当前模型、teacher 和 best-of-warp 的局部误差构造 `p_solvable`。
- 输出稳定的非语义错误标签；DINO、FLIP、CGVQM 采用可选 scorer 协议，不作为核心包硬依赖。

验收：小面积断线不会被全图平均掩盖；全屏切换、重复帧和大面积不可解释遮挡不会进入最终困难池。

### 阶段 C：断点流水线与 8 卡运行

- SQLite 使用 WAL、busy timeout 和短事务；任务粒度为视频块。
- `pending/running/done/failed` 保存 owner、lease、attempt 和原子 part 路径。
- 采用 `multiprocessing.get_context("spawn")`；父进程不初始化 CUDA/NPU。
- worker 内先设置设备，再加载模型；每个 worker 按视频块顺序解码三帧滑窗。
- 固定生产 batch，尾批复制 padding 并用 `valid_count` 截断。
- 阶段 A 全量主模型、阶段 B 候选 teacher、阶段 C 入选样本诊断分离执行。

验收：中断后过期 lease 可恢复；重复运行不重复产物；CPU 单 worker 与多 worker 的样本 ID 和判定一致。

### 阶段 D：产物、离线包和运维

- 按原视频合并失败区间；invalid/out-of-scope 形成强制断点。
- 原始 PNG 优先 hardlink，失败时显式回退 copy。
- 输出 `img0 | GT | prediction | img1`、热图、边缘、flow/mask 和局部放大图。
- 合并 JSONL part 时按稳定键排序，记录模型、权重、配置和环境哈希。
- 提供 CANN/torch/torch_npu/NPU 数量探针；禁止自动安装或替换系统栈。
- `third_party/manifest.json` 记录资源路径、类别、来源、许可证和 SHA-256；归档前验证全部资源。
- 离线 wheel 安装固定使用 `--no-index`，运行时禁止下载。

验收：干净目录中使用 mock 模型完成 index → mine → finalize；篡改任一第三方文件时哈希检查失败。

## 5. 配置与阈值策略

所有阈值写入 JSON 配置并进入 run hash。默认阈值只用于 smoke test，不能宣称适用于所有游戏。正式数据先输出 `accept/review/reject` 三档，再根据人工抽检校准；灰区样本写 manifest 但不复制到 `extremely_hard_case`。

模型、命名正则、网络尺寸、batch、stride、设备列表、mask 公式、阈值、teacher 和可选 scorer 都必须可配置。无法从环境可靠推断的值不提供隐式猜测。

## 6. 测试矩阵

- 单元测试：文件名解析、稳定 ID、区间合并、强制断点、SQLite lease、哈希校验。
- 数值测试：flow 缩放、平移、padding、mask0/mask1、候选框和原因标签。
- 集成测试：mock 模型和合成帧的完整单进程流水线。
- 多进程测试：两个 CPU worker 的 spawn、尾批 padding、崩溃恢复和幂等输出。
- 平台测试：CUDA 本机 smoke test；目标机 NPU eager FP32 smoke test、8 卡设备探测和无隐式 fallback 检查。
- 性能测试：单卡固定 batch 基线与 1/2/4/8 卡扩展；记录 decode、H2D、forward、D2H、CPU postprocess 和 writer。

## 7. 完成定义

首版完成必须同时满足：

1. 本地自动测试全部通过。
2. mock 数据端到端生成连续帧输出、可视化和 manifest。
3. CLI 的 `--help`、示例配置和离线部署说明完整。
4. 未提供真实模型时给出明确接入错误，不隐藏使用 mock。
5. 目标机探针能够报告 CANN 9.0 配套环境，但不修改它。
6. 所有第三方资源均可离线校验；缺失可选模型时显式降级。

