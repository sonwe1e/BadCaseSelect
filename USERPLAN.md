一、当前存在一个重要的内存预算错误

PendingReservation 已经增加了：

device_retained_bytes

并且单个 reservation 的 pipeline_bytes 正确包含了：

CPU retained
+ CPU scratch
+ reconstruction transient
+ NPU Tier-2 retained

但是 pipeline 的累计变量仍然只有：

pending_reserved_bytes
pending_retained_bytes

提交 Future 时只执行：

pending_reserved_bytes += reservation.reserved_bytes

没有累计：

reservation.device_retained_bytes

下一批准入判断却是：

pending_reserved_bytes + reservation.pipeline_bytes > buffer

这只包含“当前 reservation 的 device bytes”，没有包含之前所有 pending Future 仍驻留在 NPU 上的 Tier-2。

结果是：

实际 NPU retained memory 可能超过 postproc_buffer_mb 的预期；
多个 pending Future 时可能产生显存峰值；
日志中的 resident_estimate 也没有展示 pending Tier-2 device memory；
2 GiB buffer 并不能真正保证总 pipeline memory 被限制在 2 GiB。

应增加：

pending_device_retained_bytes = 0

准入条件改为：

pending_reserved_bytes \
+ pending_device_retained_bytes \
+ reservation.pipeline_bytes \
> postproc_buffer_bytes

提交和完成 Future 时同步加减：

pending_device_retained_bytes += reservation.device_retained_bytes
pending_device_retained_bytes -= reservation.device_retained_bytes

同时在进度日志中显示：

pending_device_retained
tier2_d2h_bytes
tier2_materialized_batches
tier2_candidate_samples

这是当前最优先需要修复的代码问题。

二、Tier-2 的设备生命周期仍需优化
1. materialize 后没有立即释放 NPU Tensor

Tier2Residue.materialize() 将 _packed 复制到 CPU 并写入 _cached，但 _packed 本身没有被清空。

因此 materialize 之后会同时持有：

NPU 上 11 通道 _packed；
CPU 上 11 通道 _cached；
CPU Tier-1；
原始三帧 float32；
scoring scratch。

应在成功传输后执行：

self._packed = None

否则候选微批次在 CPU diagnosis 期间仍会占用大量 NPU 显存。

2. 当前锁只能保护同一个 residue

Tier2Residue 内部的 _lock 只能避免多个线程同时 materialize 同一个 residue。不同 Future 对应不同 residue，因此 postproc_workers=4 时，仍可能有多个 CPU 后处理线程同时向同一张 NPU 发起 D2H。

这至少存在两个风险：

多条 Host 线程同时操作同一 NPU context/默认 stream；
D2H 与下一批 reconstruction 的时序和同步不可控。

更稳妥的结构是：

CPU scoring Future
    ↓ 只返回 candidate_indices
NPU worker 主线程
    ↓ 统一执行 selected Tier-2 D2H
CPU diagnosis Future

也就是让 CPU 线程只做 NumPy/SciPy 工作，不直接触碰 torch_npu Tensor。

三、CGVQM问题
2. 实际 backend 记录存在错误

每个 worker 的 _load_scorer() 可能因为 NPU Conv3D 不可用而回退 CPU，并返回实际 backend。

但 worker 入口忽略了这个返回值，父进程最后使用的是配置推导出的 backend label，而不是实际运行 backend。

因此可能出现：

实际：8 个 worker 全部在 CPU 运行 CGVQM
summary：backend=npu

应让每个 worker 写入：

{
  "device": 3,
  "model_backend": "npu:3",
  "scorer_backend": "cpu",
  "fallback_reason": "Conv3D unsupported"
}

父进程汇总后再记录：

npu
cpu
mixed

否则运行元数据不可信。

3. CPU fallback 可能造成严重过载

如果八个 NPU worker 的 CGVQM scorer 全部回退 CPU，那么会同时出现：

八份 R3D-18；
每个进程 torch.set_num_threads(8)；
最多 64 个 Torch CPU intra-op 线程；
每个进程仍然加载一份插帧模型；
同时执行压缩和图像 crop。

建议先在独立 probe 阶段确认 R3D Conv3D 是否支持 NPU。若必须回退 CPU，CGVQM CPU worker 数应单独配置，通常不能继续等于 8。

4. Context cache 方向正确，但存储格式效率较低

当前缓存的是：

全分辨率 prediction float32
+ 全分辨率 GT float32

1080p 每个缓存条目约为：

1920×1080×3×4×2≈47.5 MiB

默认 256 MiB 只能保存约 5 个条目，而 CGVQM 窗口是 16 帧。4K 下单条约 190 MiB，基本只能缓存一个。

更适合的缓存内容是：

prediction uint8；
GT uint8；
或者直接缓存不同候选需要的 224×224 crop；
或先计算整个视频所有候选的 context 并集，每个 context sample 只推理一次。

新增测试证明缓存不会改变数值结果，并能减少重复 reconstruction；但测试使用的是 64×64 小图，不能反映生产分辨率下的缓存命中能力。

5. 固定 crop 问题仍然存在

每个候选仍只使用中心帧的 primary box，并在整个 16 帧窗口中固定这个 crop。

高速人物、武器、弹道或粒子可能在数帧内离开 crop，导致：

temporal persistence 被低估；
CGVQM 热区落到背景；
spatial overlap 不准确；
大运动结构错误被错误降级到 Review。

这里仍建议增加 flow-tracked crop 或 16 帧候选框 union。

四、CPU和流水线
2. 并行解码只覆盖同一个 triplet 的三个 cache miss

当前 decode_workers 并行读取一个 triplet 的 img0、GT 和 img1，但必须等这三个结果全部完成，才开始下一个 triplet。

连续帧 triplet 通常高度重叠，例如：

1,2,3
2,3,4
3,4,5

在缓存正常时，每个新 triplet 通常只有一张新图片需要解码，因此四个解码线程大部分时间不会同时工作。

更有效的是批次级 look-ahead：

收集未来 N 个 triplet；
求所有唯一帧路径；
对 cache miss 批量提交解码；
按原顺序组装 triplet。
3. 输入构建仍然是逐帧处理

当前仍然会对每张图分别执行：

HWC→CHW；
uint8→float32；
单张 F.interpolate；
copy 到 batch Tensor；
尾批复制最后一个样本。

ModelAdapter 随后仍执行 CPU→NPU .to(non_blocking=True)，但代码中没有固定 pinned staging buffer、显式 stream 或双缓冲。

因此主线程依然是：

CPU 构建输入
→ H2D
→ 模型
→ reconstruction
→ Tier-1 D2H
→ 提交 CPU Future

而不是完整的 H2D、compute、D2H 多级重叠。

五、Teacher 阶段还可以继续减小传输

Teacher 的 _finish_teacher_batch() 只消费 prediction，不使用 flow、mask 或 warp。

但 Teacher 目前也启用了 7 通道 Tier-1：

flow0 + flow1 + prediction

因此可以单独提供：

pack_prediction_to_cpu()

将 Teacher D2H 从 7 通道进一步下降到 3 通道，减少约 57%。


六、2026-07-27 下一轮实施计划

本轮目标是在不改变候选区域原生分辨率检测逻辑、困难样本阈值和训练数据质量门控的前提下，先建立精确观测，再把最昂贵的 flow scope 统计迁移到 NPU，随后减少评分全图扫描并重构两阶段 pending 调度。A3 参数选择以完整 chunk 的端到端速度、内存峰值和无持续换页为准，不单独追求更大的模型 batch。

P0：补齐精确性能观测

现有 `scoring` 和 `motion_gates` 计时继续拆分为：

- `validity_difference`
- `validity_histogram`
- `validity_quantile`
- `flow_oob`
- `flow_background_median`
- `flow_gradient`
- `flow_quantile`
- `error_map_generation`
- `candidate_quantile`
- `native_component_label`
- `integral_windows`
- `summary_metrics`
- `phase2_diagnosis`

每个任务开始时必须打印实际解析后的：

- `resolved_postproc_workers`
- `resolved_microbatch_size`
- `postproc_buffer_mb`
- `candidate_ratio`
- `actual_tier2_d2h_bytes`
- `phase1_queue_depth`
- `phase2_queue_depth`

周期日志和最终日志继续报告这些数据，使自动 `postproc_workers` 因 `cpu_threads_per_worker` 或 CPU fair-share 退化到 1 时可以直接观察。

P1：将 flow scope metrics 迁移到 NPU

重建阶段在设备侧直接生成：

- `out_of_bounds_ratio`
- `discontinuity_ratio`
- `foreground_ratio`
- `occlusion_ratio`
- `background_motion`
- 低分辨率或压缩后的 `flow_discontinuity_map`

CPU Tier-1 只保留 prediction 3 通道、scope scalar 和可选 1 通道压缩 discontinuity map，不再传回 4 通道 full-resolution flow，也不再执行两次 CPU `_flow_metrics`。目标是消除主要的 `motion_gates` CPU 扫描、降低 retained memory，并释放更多微批次并发空间。

P2：保持评分语义不变，减少全图扫描

按低风险顺序实施：

1. 四种窗口统计共用一张 integral image。
2. Phase 1 只计算 structure 的完整统计。
3. RGB、luminance 和 directional edge 的详细 metrics 延迟到 Phase 2。
4. quantile 和 top-area 统计合并为一次 partition。
5. 缓存 GT basis 和 endpoint basis，避免 stride=1 相邻 triplet 重复计算 GT Sobel。

第 1 项要求数值完全一致；其余项目增加 parity test，确保 candidate region、排序、状态和困难样本判断不变。

P3：重构 pending 调度

调度状态明确拆分为：

- `phase1_pending`
- `phase2_pending`
- `completed_reorder_buffer`

主线程优先回收任意已完成 Future，而不是只等待队首。Phase 1 完成后按 candidate 数量重算 reservation，由 NPU worker 主线程统一执行 Tier-2 D2H；非候选样本立即释放 Tier-1/Tier-2 引用。最终通过 batch sequence id 和 sample sequence id 恢复输入顺序，消除 FIFO head-of-line blocking，同时保持 JSONL 稳定顺序和异常传播语义。

P4：A3 配置 A/B 测试

第一组对照参数：

```json
{
  "cpu_threads_per_worker": 1,
  "postproc_workers": 2,
  "postproc_buffer_mb": 4096,
  "postproc_microbatch_size": 4
}
```

同时设置：

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

对照 `postproc_microbatch_size=8` 和自动微批次，记录完整 chunk 的 samples/s、各细分阶段 ms/sample、NPU/主机内存峰值、swap、Tier-2 实际传输量和两个 phase 的队列深度。配置优化只作为调度层收益验证，数量级收益应主要来自 P1 和 P2。

验收条件

- 候选 region、排序、状态、影响判定的指标和 JSONL 记录顺序与优化前保持 parity；快速拒绝样本不再生成未参与判定的 RGB/luminance/directional-edge 详细汇总键。
- Tier-1 不再包含 full-resolution flow0/flow1。
- flow scope scalar 在 CPU 与设备侧实现之间通过数值 parity。
- Phase 1/Phase 2 任意完成顺序下，最终记录顺序稳定。
- `postproc_buffer_mb` 同时约束 CPU retained、scratch、transient 和仍驻留设备的 Tier-2。
- A3 多 chunk 运行无超过 30 秒的无解释静默、无持续换页，且日志能解释 worker、微批次、队列深度及 Tier-2 D2H。

实施状态（2026-07-27）

- P0 已实施：周期和最终 timing 包含上述 13 个细分阶段；任务启动与进度日志包含解析后的 worker/微批次、buffer、候选比例、Tier-2 实际字节数和两阶段队列深度。
- P1 已实施：main 的 native flow scope 在 reconstruction device 上计算；Tier-1 改为 prediction、6 个 scalar 和 uint8 discontinuity support，full-resolution flow 不再回传 CPU。CPU fallback 路径沿用同一 torch 公式。
- P2 已实施：四类窗口共享 integral image；Phase 1 只汇总 structure；候选在 Phase 2 补齐详细指标；quantile/top-area 共享一次 partition；同一微批的重叠帧使用三项 rolling basis cache。
- P3 已实施：`legacy_pending`、`phase1_pending`、`phase2_pending` 和 `completed_reorder_buffer` 分离；按 Future 完成状态回收；Phase 1 后按候选数切换为 Phase-2 reservation；最终按全局 sequence id 输出。
- P4 已写入 `configs/example.json` 和 `configs/example.jsonc`：`cpu_threads_per_worker=1`、`postproc_workers=2`、`postproc_buffer_mb=4096`、`postproc_microbatch_size=4`。A3 仍需与 8/auto 做同一 chunk 的实机墙钟对照。
