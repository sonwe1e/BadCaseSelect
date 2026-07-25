# 内存安全的全分辨率后处理实施计划

日期：2026-07-25
设计规格：`docs/superpowers/specs/2026-07-25-memory-safe-full-resolution-postprocessing-design.md`

## 实施原则

- 先消除原分辨率完整 batch 的 float32 NumPy 与 Tensor 驻留，再优化诊断计算。
- `model.batch_size` 只控制固定网络分辨率输入、模型输出和 tail padding。
- 原分辨率 float32 数据只允许存在于当前 reconstruction 微批次及其 CPU Future。
- fast-reject 只能减少被拒绝样本的诊断成本，不能降低进入 A/B/C 训练目录的质量门槛。
- 模型输出、阈值、困难等级、记录顺序、异常传播、恢复和物化语义保持不变。
- 所有数值重构均以旧路径为参考，允许的浮点误差不超过 `1e-7`。

## Task 1：uint8 解码批次与低分辨率模型输入

文件：

- `src/vfi_hard_miner/worker.py`
- `src/vfi_hard_miner/diagnostics.py`
- `src/vfi_hard_miner/cgvqm_stage.py`
- `src/vfi_hard_miner/model_adapter.py`（仅在需要公开已缩放输入契约时修改）
- `tests/test_worker.py`
- `tests/test_diagnostics.py`
- `tests/test_cgvqm_stage.py`
- `tests/test_model_adapter.py`

内容：

1. 将生产路径 `DecodedItem` 明确为 uint8 HWC 三元组，校验 dtype、范围和共同形状。
2. 删除主 producer 的 float32 LRU；diagnostic producer 同样只返回 uint8。
3. 新增网络输入构造器：
   - 预分配 `[production_batch, 3, input_h, input_w]` float32 Tensor；
   - 每次只转换一个原分辨率 uint8 端点；
   - 使用现有 bilinear、`align_corners=False` 缩放；
   - 直接写入低分辨率 batch；
   - 尾批复制最后一个低分辨率样本。
4. `_infer_model_batch` 只返回低分辨率模型输出和可观测性字节数，不返回原分辨率 `img0_tensor/img1_tensor`。
5. 主阶段、diagnostic 和 CGVQM 改用同一 helper。
6. 对比旧 Adapter 自动 resize 路径，验证输入和模型输出等价。

## Task 2：按 reconstruction 微批次转换 float32

文件：

- `src/vfi_hard_miner/worker.py`
- `src/vfi_hard_miner/diagnostics.py`
- `src/vfi_hard_miner/cgvqm_stage.py`
- `src/vfi_hard_miner/image_io.py`（只在需要批量转换 helper 时修改）
- `tests/test_worker.py`
- `tests/test_diagnostics.py`
- `tests/test_cgvqm_stage.py`

内容：

1. 新增微批次准备对象，包含：
   - 转换后的 float32 `DecodedItem`；
   - 仅覆盖当前微批次的 `img0/img1` NCHW stack；
   - uint8、float32 和 stack 的实际字节统计。
2. 在预算准入通过后、调用 `_reconstruct_outputs` 前才创建该对象。
3. CPU Future 持有当前微批次 float32 三元组和 reconstruction 结果；完成后全部释放。
4. 完整推理 batch 在 microbatch 循环中只保留 uint8 items 与低分辨率 outputs。
5. 保留 `_infer_and_reconstruct` 等测试/兼容入口，但内部复用新 helper，避免生产旁路重新引入整批 stack。
6. 通过弱引用或对象追踪测试确认第二个微批次开始前不存在完整 batch float32 Tensor。

## Task 3：完整生命周期内存预约

文件：

- `src/vfi_hard_miner/worker.py`
- `src/vfi_hard_miner/diagnostics.py`
- `configs/example.jsonc`
- `README.md`
- `docs/offline_deployment.md`
- `method.html`
- `tests/test_worker.py`
- `tests/test_diagnostics.py`

内容：

1. 将 reservation 拆为：
   - `future_retained`：18 reconstruction channels + 9 float32 input channels；
   - `future_scratch`：CPU scoring 的 24 个 float32 plane；
   - `reconstruction_transient`：Future retained 之外的 endpoint stack、D2H/CPU reconstruction 临时峰值；
   - 每 Future 1 MiB 固定开销。
2. 提交前准入使用：

   ```text
   pending_reserved + next_future_reserved + next_reconstruction_transient
   ```

3. 单样本超预算时排空 pending、独占 reconstruction 和评分，并记录 `oversize=1`。
4. `decode_uint8`、`network`、`reconstruction_transient`、pending retained/reserved 和 `resident_estimate` 分开打印。
5. `resident_estimate` 明确标注为可归因估算，不等同 RSS。
6. diagnostic 调度应用相同准入顺序；同步 CGVQM 循环使用完整预算但仍统计 transient。

## Task 4：Fast-reject 分级评分

文件：

- `src/vfi_hard_miner/worker.py`
- `src/vfi_hard_miner/gates.py`（仅在需要公共简化记录 helper 时修改）
- `src/vfi_hard_miner/grading.py`
- `src/vfi_hard_miner/outputs.py`
- `tests/test_worker.py`
- `tests/test_gates.py`
- `tests/test_grading.py`
- `tests/test_outputs.py`

内容：

1. `_sample_record` 调整顺序为 validity → motion/scope → local score → fast-reject → diagnosis → final decision。
2. validity reject、scope reject 或 `mining_p_wrong < wrong_reject_below` 时不调用 `diagnose_sample`。
3. 简化记录保持稳定顶层 schema：
   - `regions=[]`；
   - `primary_region_index=None`；
   - `p_solvable=0.0`；
   - 保留 scoring、validity、scope；
   - diagnosis metrics 明确记录 `skipped=1` 与 skip reason。
4. 确保 fast-reject 始终为非训练级，A/B/C 物化拒绝任何未完成完整 diagnosis 的记录。
5. 完整 diagnosis 样本与旧实现比较字段、原因、状态、等级和排序。

## Task 5：候选区域 Branch Evidence

文件：

- `src/vfi_hard_miner/diagnosis.py`
- `src/vfi_hard_miner/scoring.py`
- `tests/test_diagnosis.py`
- `tests/test_scoring.py`

内容：

1. 新增带 1 像素 halo 的 branch-region structure scorer。
2. 分别处理普通区域、图像四边和四角；halo 外部使用现有 edge padding 语义。
3. `teacher`、`warp0`、`warp1`、`warp_blend`、`img1-vs-GT` 和 `prediction-vs-img1` 只计算候选 crop。
4. GT 与 img1 crop basis 在同一 region 内复用。
5. flow scope 继续复用整帧 motion evidence；mask tearing evidence 保留全局归一化基准，只物化候选区域支持。
6. 随机图、重叠 region、贴边 region 与旧整图结构图结果做 `1e-7` 等价测试。

## Task 6：阶段耗时与周期日志

文件：

- `src/vfi_hard_miner/worker.py`
- `src/vfi_hard_miner/diagnostics.py`
- `tests/test_worker.py`
- `tests/test_diagnostics.py`

内容：

1. 将现有 CPU totals 扩展为线程安全的 pipeline timing totals：
   - decode；
   - inference；
   - reconstruction；
   - future wait；
   - scoring；
   - branch evidence；
   - motion gates；
   - serialization。
2. producer 在读取和批次构造时计时，不把 queue put 阻塞算入 decode。
3. `_infer_model_batch` 和 `_reconstruct_outputs` 调用点按有效样本数累计。
4. `drain_one` 只累计 `Future.result()` 实际阻塞时间。
5. 每完成 16 个 scored 样本打印一次累计 `ms/sample`；chunk 完成打印 final。
6. 进度吞吐改为两位小数，并在 reconstruction 期间显示当前 transient 状态。
7. 慢 Future 等待仍每 30 秒给出解释性日志并续租。

## Task 7：兼容性与端到端测试

文件：

- `tests/test_worker.py`
- `tests/test_diagnostics.py`
- `tests/test_cgvqm_stage.py`
- `tests/test_pipeline_main.py`
- `tests/test_pipeline_teacher.py`
- `tests/test_finalize.py`
- `tests/test_cli.py`

内容：

1. 覆盖 batch 16、32、64 和 250、256、257 条任务。
2. 验证 tail padding、输入顺序、JSONL 顺序和异常传播不变。
3. 验证 teacher、diagnostic、CGVQM 和默认 finalize 路径不回退到整批 float32 stack。
4. 验证 fast-reject 不进入训练目录，完整 hard case 仍正常分级和物化。
5. 验证 postproc budget、超预算独占、两个 Future 并发和中断恢复。
6. 验证周期 timing 日志、内存字段、两位吞吐和 heartbeat。

## Task 8：验证与 A3 验收说明

1. 运行新增模块 targeted tests。
2. 运行 `python -m pytest` 全量测试。
3. 运行 `python -m compileall -q src tests`。
4. 解析 `configs/example.json` 和 `method.html`。
5. 运行 `git diff --check`。
6. 确认 `.serena/`、`.superpowers/` 和 `plan.md` 未被修改或纳入。
7. 输出 A3 现场矩阵：
   - batch 16/32/64；
   - postproc workers 1/2；
   - 记录完整 chunk 时间、scored/s、worker RSS、swap 和八阶段 timing。
8. 以完整 chunk 最短、无持续 swap、训练 hard case 质量不下降作为最终配置选择标准。
