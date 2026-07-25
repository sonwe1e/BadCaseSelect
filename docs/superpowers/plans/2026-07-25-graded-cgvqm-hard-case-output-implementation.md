# A/B 分级困难样本与 CGVQM 精判实施计划

日期：2026-07-25
设计规格：`docs/superpowers/specs/2026-07-25-graded-cgvqm-hard-case-output-design.md`

## 实施原则

- 保留现有 main 推理数值、batch padding、记录顺序和异常传播。
- 传统评分继续负责全量高召回，CGVQM 只处理候选。
- 新阶段必须具有独立状态、幂等结果和明确日志。
- 训练帧按源文件字节复制；诊断图单独 JPEG 编码。
- 旧分阶段命令保留，但 `run` 是唯一推荐入口。
- 测试先覆盖纯函数和输出契约，再接入多进程及端到端流程。

## Task 1：配置与分级纯函数

文件：

- `src/vfi_hard_miner/config.py`
- 新增 `src/vfi_hard_miner/grading.py`
- `configs/example.json`
- `configs/example.jsonc`
- `tests/test_config.py`
- 新增 `tests/test_grading.py`

内容：

1. 新增 `CGVQMConfig`，包含 enabled、backbone/calibration 路径、backend、CPU 降级、clip frames、crop size、batch size、候选上限和 A/B error 阈值。
2. `ThresholdConfig` 新增 `severe_wrong_accept_at`，校验其不低于 `wrong_accept_at`。
3. `OutputConfig` 新增 JPEG quality；默认训练物化改为 copy，分级输出固定为 A/B flat。
4. 建立纯函数 `grade_candidate(record, cgvqm_result, config)`：
   - invalid/out-of-scope/solvability 灰区直接非训练级。
   - 传统/CQVQM 冲突为 Review。
   - A 需要高 CGVQM、高传统严重度和强结构/时序证据。
   - B 需要双方明确确认。
5. 记录 `grade`、`quality_gate`、`severity_score`、`reason_confidence`。

## Task 2：CGVQM-2 核心 scorer

文件：

- 新增 `src/vfi_hard_miner/cgvqm.py`
- `third_party/manifest.json`
- `third_party/README.md`
- 新增 `tests/test_cgvqm.py`

内容：

1. 实现不依赖 torchvision 运行时下载的 R3D-18 网络结构。
2. 从本地文件加载 Kinetics 主干和 CGVQM-2 校准权重。
3. 实现与官方一致的 RGB 标准化、多层通道归一化特征距离、时空 error map 和 raw error。
4. 提供 CPU/NPU probe；自动模式只有在配置允许时才可显式降级 CPU。
5. 提供可注入的轻量 fake scorer，确保本地单元测试不依赖大模型文件。
6. 在 manifest 中登记 R3D 权重，不允许运行时联网。

## Task 3：候选时序窗口与 CGVQM durable stage

文件：

- 新增 `src/vfi_hard_miner/cgvqm_stage.py`
- `src/vfi_hard_miner/pipeline.py`
- `src/vfi_hard_miner/state.py`（仅在需要公共状态辅助时）
- `src/vfi_hard_miner/worker.py`（复用批量推理/重建 helper）
- 新增 `tests/test_cgvqm_stage.py`

内容：

1. 从 main/teacher 合并结果中选取 accept 和高召回 review 候选。
2. 按视频和中心帧构建固定 16 帧窗口；首尾确定性 padding，不跨视频。
3. 合并重叠窗口，建立“上下文 sample → 候选 crop”依赖，避免重复重建。
4. worker 批量重新生成必要 prediction，并只保留缩放后的候选 crop。
5. CGVQM 微批次输出中心 error、时间持续性、突变和传统区域重叠率。
6. 用独立 SQLite stage、attempt part 和 winning result 支持断点恢复。
7. overlay 后生成 `graded_results.jsonl`，每个原始 record 恰好一条。

## Task 4：A/B 扁平复制与增量物化

文件：

- 重构 `src/vfi_hard_miner/materialization.py`
- `src/vfi_hard_miner/outputs.py`
- `src/vfi_hard_miner/indexing.py`
- `tests/test_outputs.py`
- `tests/test_finalize.py`
- `tests/test_pipeline_index.py`

内容：

1. 从每个 A/B 中心直接生成三帧映射，不再先构造 segment 子目录。
2. 同级映射按目标 basename 去重；跨级保留重复。
3. 写入前全量检查同名冲突，异源/异内容失败。
4. 训练帧固定 byte-copy，使用临时文件、fsync 和原子改名。
5. 暂存目录为 `<execution_id>/hard_case/A|B`，每视频状态记录中心数、唯一帧数、字节数、复制数和耗时。
6. 索引排除暂存、正式 A/B 和诊断目录。
7. 恢复时验证现有文件大小/哈希，不重复复制。

## Task 5：finalize 与一键 run

文件：

- `src/vfi_hard_miner/cli.py`
- `src/vfi_hard_miner/finalize.py`
- `src/vfi_hard_miner/pipeline.py`
- `tests/test_cli.py`
- `tests/test_pipeline_main.py`
- `tests/test_finalize.py`

内容：

1. `run` 顺序调整为 index → main → optional teacher → CGVQM → finalize。
2. `finalize` 读取 graded results，验证 CGVQM stage 完整后才允许发布。
3. 发布 A/B、visualization、manifest 和 segments/grade summary 时保持现有原子回滚。
4. `segments.json` 转为按 grade 记录训练中心和三帧映射摘要，不再表示训练叶目录。
5. 最终摘要增加 A/B 中心数、唯一帧数、copy 计数、Review/Reject 和 CGVQM 统计。

## Task 6：两行五列 JPEG 诊断图

文件：

- `src/vfi_hard_miner/visualization.py`
- `src/vfi_hard_miner/diagnostics.py`
- `tests/test_visualization.py`
- `tests/test_diagnostics.py`

内容：

1. `make_diagnostic_grid` 固定生成两行五列：
   - `img0 | GT | prediction | img1 | structure error`
   - `flow_t0 | flow_t1 | CGVQM center | temporal sensitivity | fused confidence`
2. 删除 mask 和第三行局部 crop。
3. 默认 panel width 480。
4. 诊断 artifact 扩展名改为 `.jpg`，quality 默认 92，固定高质量色度采样。
5. CGVQM heatmap 缺失时属于流程错误，不生成占位图。
6. manifest 记录尺寸、格式和 quality。

## Task 7：方法文档与用户文档

文件：

- 新增根目录 `method.html`
- `README.md`
- `docs/offline_deployment.md`
- `third_party/README.md`

内容：

1. `method.html` 使用纯 HTML/CSS 详细解释传统误差、候选区域、质量门、CGVQM、原因标签和 A/B 映射。
2. README 只把 `run` 放在主流程，分阶段命令放入恢复章节。
3. 文档明确训练帧 copy 与诊断 JPEG 的区别。
4. 离线部署说明新增 R3D 权重验证与禁止自动下载。

## Task 8：验证

1. 运行新增模块的 targeted tests。
2. 运行 `python -m pytest -q`。
3. 运行关键模块 `py_compile`。
4. 运行 `git diff --check`。
5. 检查未跟踪 `.serena/`、`.superpowers/` 和 `plan.md` 未被改动或纳入。
6. 总结未能在本机验证的 A3 NPU probe，并提供明确现场验收命令和日志指标。
