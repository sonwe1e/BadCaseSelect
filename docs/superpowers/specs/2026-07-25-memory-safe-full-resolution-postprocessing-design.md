# 内存安全的全分辨率后处理设计

日期：2026-07-25

## 目标

当前 A3 流程已经避免了 CPU Future 无界堆积，但仍会在一个完整模型 batch 的生命周期内同时保留：

- `img0/gt/img1` 的原分辨率 float32 NumPy 数组；
- `img0_tensor/img1_tensor` 的原分辨率 float32 `torch.stack` 副本；
- 低分辨率模型输出；
- 当前 reconstruction 和 CPU scoring Future。

这些整批原分辨率数据没有进入 `reserved` 日志。大 batch 和多 worker 会因此产生数十 GiB 隐藏驻留，引发内存带宽竞争、NUMA 远端访问甚至换页。

本次改造的目标是：

1. 使 `model.batch_size` 只影响固定网络分辨率输入与模型输出；
2. 让原分辨率 float32 数据只存在于当前 reconstruction 微批次及其 CPU Future；
3. 对明确无效、越界或简单的样本跳过完整 branch diagnosis；
4. 将 branch evidence 从整图计算缩小为候选区域计算；
5. 补齐端到端阶段耗时和可归因内存日志，以 A3 整体吞吐选择最终 batch 和并发。

不修改模型、阈值、困难样本等级、训练数据复制规则或正式发布语义。

## 方案选择

采用“可配置 batch + 低分辨率推理批次”方案。

不把 batch 固定降为 16 或 32，因为这只会缩小而不会消除隐藏内存问题，并可能损失 NPU 吞吐。也不在首版加入运行时自动 batch 调优，因为不同视频复杂度会污染短期采样结果，并增加恢复与重复推理成本。

示例配置暂时保持 `model.batch_size=64`。改造完成后，A3 使用同一数据分别测试 16、32、64，以完整 chunk 墙钟时间和 `scored samples/s` 选择最终值。

## 数据表示与生命周期

### 解码批次

主阶段和 diagnostic 阶段的 producer 只维护 uint8 LRU cache。批次元素保存：

- 原始记录；
- `img0/gt/img1` 的 uint8 HWC 数组；
- 已验证的共同形状。

取消 float32 frame cache。stride=1 的重叠帧继续通过 uint8 cache 复用解码结果。队列和 pending batch 只引用 uint8 数据。

### 网络输入

新增网络输入构造器，按样本执行：

1. 从 uint8 端点帧创建临时 Tensor；
2. 按现有 `float32 / 255` 语义归一化；
3. 使用现有 bilinear、`align_corners=False` 规则缩放到模型固定输入尺寸；
4. 将结果写入预分配的低分辨率 batch Tensor；
5. 立即释放该样本的原分辨率临时 float32 Tensor。

尾批继续复制最后一个低分辨率样本进行 padding。`ModelAdapter` 接收已经达到 `network_size` 的输入，因此不会再次缩放。模型输出、顺序、验证和 tail padding 语义保持不变。

### Reconstruction 微批次

完成一次模型推理后，低分辨率输出按后处理预算切片。每个微批次开始前才把对应 uint8 `img0/gt/img1` 转为原分辨率 float32：

- `img0/img1` stack 只覆盖当前 reconstruction 微批次；
- `gt` 和三个 float32 帧随 CPU Future 保留；
- reconstruction、D2H 和 CPU 评分完成后立即释放；
- 完整模型 batch 不再持有任何原分辨率 float32 stack。

主阶段、diagnostic 和 CGVQM 复用同一网络输入及微批次转换接口，避免旁路阶段继续保留整批 float32 数据。

## 分级评分

每个重建样本依次执行：

1. validity 基础检查；
2. flow motion evidence 与 scope gate；
3. local error score；
4. fast-reject 判断；
5. 对剩余样本执行完整 diagnosis 和 solvability；
6. 最终 hard-case decision。

以下任一条件成立时直接生成简化记录：

- `validity.label == "reject"`；
- `scope.label == "reject"`；
- `scoring.mining_p_wrong < thresholds.wrong_reject_below`。

简化记录保留原始样本元数据、status、validity/scope 标签、基础 scoring/validity/scope 指标和明确原因。其 `regions=[]`、`primary_region_index=null`、`p_solvable=0`。这里的零表示没有执行 solvability diagnosis，而不是证明模型不可修复。

任何可能进入 A/B/C 训练目录的样本都必须完成完整 diagnosis、solvability 和最终 decision。fast-reject 不得降低训练集质量门槛。

## 候选区域 Branch Evidence

当前实现已把“每个 region 重算整图”降为“每个 branch 重算一次整图”。本次继续把 branch evidence 限制到候选区域：

1. 每个候选 box 向外扩展 1 像素 Sobel halo；
2. 裁取 branch 与 reference 的相同区域；
3. 在 crop 上计算 RGB、luminance、Sobel 和 structure evidence；
4. 去掉 halo 后，仅对原候选 box 计算 region score。

图像边界沿用原有 edge padding。随机区域、贴边区域和角落区域必须与“整图计算后裁剪”的结果在 `1e-7` 内一致。

首版按 region 独立计算，避免引入复杂的区域合并语义。若候选区域高度重叠，后续可以在不改变接口的前提下增加 crop 合并。

flow scope 仍进行一次全分辨率计算，因为它参与整帧 out-of-scope 判断。mask diagnosis 只为候选区域生成必要证据；不能改变 tearing 判定的全局归一化基准。

## 内存预算

内存日志和准入判断区分四类数据：

1. `decode_uint8`：当前已交付解码 batch 的实际 uint8 字节数；
2. `network`：低分辨率模型输入与模型输出字节数；
3. `reconstruction_transient`：下一微批次在 Future retained 之外额外需要的 endpoint stack、重建工作空间和 D2H 临时数据；
4. `pending`：已提交 CPU Future 的 retained、scoring scratch 和固定开销。

日志增加 `resident_estimate`：

```text
decode_uint8 + network + reconstruction_transient + pending_reserved
```

该值是程序可归因估算，不宣称等于进程 RSS。

`postproc_buffer_mb` 继续只约束 reconstruction 与 CPU 后处理，不把 uint8
解码 batch 或低分辨率网络 batch 重复计入该预算；后两者只进入
`resident_estimate`，用于解释进程 RSS。`next_future_reserved` 已包含提交后会被
Future 保留的原分辨率输入和重建结果，`reconstruction_transient` 只计算二者之外
的临时峰值，因此下面的准入公式不会重复计数同一分配。

启动 reconstruction 前必须满足：

```text
existing_pending_reserved
+ next_reconstruction_transient
+ next_future_reserved
<= postproc_buffer
```

如果单样本自身超过预算，则先回收所有 pending Future，再独占运行该样本，并只输出一次 `oversize=1` 警告。

自动并发继续限制为每 worker 1–2 个 postprocess Future。首版不自动增大 `postproc_buffer_mb`。A3 在内存改造后分别测试 `8 workers × 1 postproc` 与 `8 workers × 2 postproc`，以整机吞吐而非单 worker 速度选择。

## 可观测性

统一累计以下阶段耗时：

- `decode_ms/sample`；
- `inference_ms/sample`；
- `reconstruction_ms/sample`；
- `future_wait_ms/sample`；
- `scoring_ms/sample`；
- `branch_evidence_ms/sample`；
- `motion_gates_ms/sample`；
- `serialization_ms/sample`。

每完成 16 个 scored 样本打印一次累计均值，chunk 结束打印最终汇总。进度吞吐显示至少两位小数，避免把明显不同的低速结果都舍入为 `0.1/s`。

`future_wait` 只统计 worker 主线程在 `Future.result()` 上的实际阻塞时间；CPU Future 内部耗时由四个 CPU 子阶段统计。`decode` 统计 producer 的实际读取与批次构造时间，不把队列背压等待误记为解码耗时。

## 错误与恢复语义

- 解码错误继续生成 invalid record，并保持输入顺序；
- 模型、reconstruction、CPU scoring 和 diagnosis 异常继续向上传播；
- Future 失败不得被 fast-reject 吞掉；
- lease heartbeat 在 Future 等待和长阶段日志期间继续执行；
- JSON part、SQLite 提交和 per-video 物化语义保持不变；
- 中断恢复不得因为 uint8 batch 或简化记录产生重复推理或错误胜出 part。

## 测试与验收

自动测试覆盖：

- batch 16、32、64，以及 250、256、257 条尾批；
- 低分辨率推理输入与旧 Adapter resize 路径的数值等价；
- producer 和队列不持有原分辨率 float32 batch；
- reconstruction 只转换当前微批次；
- fast-reject 跳过 diagnosis，且不能进入训练目录；
- hard case 的字段、原因、等级和顺序保持不变；
- crop branch evidence 在普通、重叠、贴边和角落区域达到 `1e-7` 等价；
- pending 加 reconstruction transient 的预算准入；
- 超预算单样本独占执行；
- 慢 Future 持续输出进度、内存、耗时并续租；
- diagnostic、CGVQM、teacher 和默认 finalize 路径兼容。

A3 验收使用相同视频、相同 worker 数和相同阈值，分别测试 batch 16、32、64。必须满足：

- worker RSS 不再随原分辨率模型 batch 线性增长；
- `inferred=total` 后不存在无法解释的长时间静默；
- 无持续 swap；
- 日志能解释 decode、inference、reconstruction、Future wait 和 CPU diagnosis 各自占比；
- 最终选择的 batch 以完整 chunk 墙钟时间最短为准；
- 训练目录中的 hard case 与旧逻辑保持同等或更高质量。
