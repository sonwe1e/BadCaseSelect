# 游戏插帧离线困难样本挖掘设计

日期：2026-07-18

## 1. 目标与边界

系统遍历一个游戏下所有场景目录和视频帧，调用当前插帧模型生成中间帧，并从全部视频中找出“数据有效、难度仍在合理范围内、模型理论上应能做好但实际出现明显局部或时序错误”的样本。

系统不按 Top-K 或固定百分比截断。所有通过判定的样本均进入该游戏的 `extremely_hard_case`。重叠或相邻的失败 triplet 合并为连续原始帧片段，供现有训练器直接读取。

首版重点检测：建筑和细长物体断裂、人物头部或武器缺失/断裂、鬼影、边界撕裂、视角旋转时的路灯等背景物体扭曲或闪烁。文字对话框和技能特效抖动只记录为低优先级，不单独触发。跳帧、菜单切换、镜头切换和人物/NPC 大面积遮挡镜头属于无效或超范围数据，不进入困难样本池。

数据清洗流水线、训练器命名解析逻辑和端到端语义命名模型不在首版实现范围内。

## 2. 输入、模型与重建契约

- 样本固定为中点插帧：`img0=F[k]`、`GT=F[k+s]`、`img1=F[k+2s]`，模型目标时刻为 `t=0.5`。
- `img0/img1` 以 RGB `[0,1]` 输入，并在推理前 resize 到模型固定 H×W；GT 保持原始分辨率。
- 当前模型位于 `models/`，权重位于 `ckpts/`，输出 `flow_t0、flow_t1、mask0、mask1`。flow 为 backward flow；两个 mask 已经过 sigmoid。
- flow 默认解释为网络输入 H×W 坐标系中的像素位移。上采样到原图后，x/y 分量分别乘 `W_original/W_network` 和 `H_original/H_network`；mask 只做双线性插值。
- `warp0=backward_warp(img0, flow_t0)`，`warp1=backward_warp(img1, flow_t1)`；`mask0` 的具体两路对应关系必须从模型代码或一次性全局校准中锁定并写入配置，生产过程禁止逐样本选择误差更低的公式。
- 最终预测遵循已知残差分支：`pred=mask1*img1+(1-mask1)*warp_blend`。
- 上线前以恒等、已知平移和边界采样合成样本验证 flow 单位、方向、缩放、`align_corners`、padding、mask0 对应关系与 CPU/NPU 数值一致性；任一项未通过时拒绝全量运行。

模型适配器必须同时返回 `warp0、warp1、warp_blend、prediction` 以及 flow/mask，供错误归因使用。teacher 使用同样的标准 prediction 接口，但不参与“正确/错误”的真值定义。

## 3. 三级判定与局部正确性

三个判定彼此独立：

1. `validity` 排除坏图、编号/时间异常、重复或跳帧、镜头/菜单过渡及明显非连续内容。
2. `in_scope` 通过全局背景运动、前景残差运动、出界比例和双向遮挡估计判断可处理范围。背景大运动允许；大面积前景遮挡镜头和极端不可解释运动进入 `out_of_scope`。
3. `correctness` 只比较 prediction 与 GT，寻找局部可见错误。

目标条件为：

```text
valid && in_scope &&
exists region: p_wrong(region) >= T_wrong && p_solvable(region) >= T_solve
```

`p_wrong` 只由 prediction 与 GT 的局部证据产生。teacher、best-of-warp 和输入运动条件只形成 `p_solvable`；teacher 失败不能证明数据无效，只会降低可解性置信度或进入复核状态。

### 3.1 全量高召回候选

对原分辨率 prediction/GT 计算 RGB/亮度残差、梯度和缺失/额外边缘图，并使用多尺度滑窗和连通区域产生候选框。不得使用全图平均作为唯一判据；至少保留局部窗口最大值、高分位、top-area mean、连通块峰值/能量/边界长度。区域面积只用于去除像素噪声，不用于压低细小结构的重要性。

### 3.2 候选精判

- 候选区域使用 FLIP 感知误差、缺失/额外边缘、骨架连通性和结构特征。
- 候选裁剪可使用 DINO dense patch feature：同时计算对齐位置距离和邻近 patch 双向匹配距离，以区分轻微位移与真实结构消失。
- 连续候选窗口使用运动补偿后的误差变化及 CGVQM 时空误差图复核闪烁、扭曲和鬼影。
- teacher 只在候选区域计算局部 GT 误差，用于增强“当前模型本应能做好”的置信度。

错误原因输出稳定的非语义标签：`missing_part`、`broken_structure`、`ghosting`、`edge_tearing`、`flicker`、`blur`、`endpoint_copy`、`blend_mask_error`。其中：

- GT 强边在 prediction 中消失且结构特征不匹配，标记结构缺失。
- 细长 GT 骨架在 prediction 中断开、端点增加或最长路径明显缩短，标记结构断裂。
- prediction 出现与端点/warp 对齐的平行双边缘，标记鬼影。
- 错误集中在 flow 或 mask 突变边界，标记边界撕裂。
- `warp0/warp1` 中存在较好结果但 blend 恶化，标记可能的 mask0 融合问题；blend 正确但最终 residual 分支恶化且结果更接近 img1，标记 endpoint copy。归因属于诊断相关性，不宣称严格因果。

首版不承诺零配置输出 `head_missing` 或 `weapon_broken`。未来可按游戏增加 Grounding DINO/SAM2 prompt pack 或少量关键部件标注；这些语义组件不作为首版硬门禁。

## 4. 连续片段与输出

每个失败 triplet 映射为原始闭区间 `[start,end]`。同一原视频内，若 `next.start <= current.end+1`，则合并区间，因此 `[1,3]+[2,4]` 得到 `[1,4]`，`[1,3]+[4,6]` 得到 `[1,6]`。无效或超范围区间强制切断，不跨原视频合并。

每个游戏输出：

```text
game/
├─ extremely_hard_case/
│  └─ 合并后的连续原始帧，遵循现有训练器命名契约
├─ extremely_hard_case_visualization/
│  └─ 片段总览、逐失败中心诊断图和局部放大图
└─ hard_case_manifest.jsonl
```

诊断图第一行显示 `img0 | GT | prediction | img1`；第二行显示感知误差、GT-only/pred-only 边缘、flow/mask 诊断；下方显示最高分局部区域的 GT/prediction 放大图。manifest 记录来源游戏/场景/视频/帧号、模型与权重哈希、`p_wrong/p_solvable`、原因、各分支误差和产物路径。

原始帧不删除、不重新编码。最终片段优先 hardlink，跨文件系统或用户配置禁止 hardlink 时复制。无效、超范围和不确定样本只写 manifest，不进入 `extremely_hard_case`。

## 5. 8 卡昇腾执行架构

目标环境为 aarch64、8 卡昇腾 A3/910B、CANN 9.0、服务器现有配套 PyTorch/`torch_npu`。公开版本矩阵暂未列出 CANN 9.0，因此项目不得自动替换服务器现有 torch/`torch_npu`；启动探针记录并验证 driver、firmware、CANN、torch、torch_npu、Python、OS 和 8 张设备。

采用三个离线阶段：

1. **全量主模型阶段**：8 个 `spawn` 进程，每进程先绑定一张 NPU 再加载模型；不使用 DDP/HCCL。按完整视频或带 halo 的长视频块调度，使用三帧滑窗复用 PNG。固定网络 H×W，实测 batch 档位后只运行一个生产 batch，尾批复制 padding。NPU 运行主模型并返回低分辨率 flow/mask；首版在 CPU FP32 完成上采样、warp、blend 和便宜候选检测。
2. **候选复核阶段**：整批切换为 teacher；FLIP 和结构分析走 aarch64 CPU。DINO、CGVQM 仅处理候选 crop/window，先做 NPU 算子探针；不通过则 CPU 降级或关闭相应辅助项，不阻塞主链。
3. **最终产物阶段**：对最终入选样本重跑主模型生成完整诊断，CPU 异步完成区间合并、图片和 manifest 落盘。

每卡使用有界预取队列和 2–3 组可复用 host buffer。默认单 stream；只有 profiler 证明 H2D/计算可重叠时才加入第二 stream。内循环禁止频繁 `.item()`、`.cpu()`、全卡 `synchronize()` 和 `empty_cache()`。OpenCV/libpng 每进程限制单线程，CPU worker 按 NUMA 拓扑绑核并保留系统/I/O 余量。

Eager FP32 先建立黄金基线，再验证 FP16/BF16，最后才尝试 TorchAir 静态图/aclgraph。flow 坐标、warp、最终融合和阈值判定首版保持 FP32。`grid_sample`、Conv3d、attention 与图模式均以目标 CANN 9.0 环境实测为准，禁止隐式 CPU fallback；若 CPU warp 成为瓶颈且 NPU 探针通过，再为常见原分辨率增加固定 shape 后处理图。

## 6. 断点、失败处理与可复现性

- 首次扫描生成稳定 `sample_id` 和视频索引，运行时不重复 glob/stat。
- 本机 SSD 上的 SQLite WAL 保存 `pending/running/done/failed` 和租约；数据库不放 NFS。
- 视频块先原子写结果，再提交 done。worker 异常后租约超时重排；幂等 sample_id 防止重复产物。
- PNG 损坏记永久样本错误并继续；NPU 进程异常可重启并重跑视频块；OOM 只允许降到预先验证的 batch 档位，不自动改变分辨率、精度或阈值。
- 不同模型、权重、阈值、数据快照或软件栈生成不同 run_id，禁止混写。
- 阈值附近保留灰区；设备间微小数值差异不得直接改变边界样本的最终分类。

## 7. 第三方依赖与离线发布

当前联网开发服务器负责准备全部第三方资源，目标服务器通过压缩包部署。项目内统一使用：

```text
third_party/
├─ wheelhouse/linux-aarch64/
├─ src/{flip,cgvqm,dinov2}/
├─ weights/{cgvqm,dinov2}/
├─ licenses/
└─ manifest.json
```

当前模型和 teacher 权重分别位于 `ckpts/current/` 与 `ckpts/teacher/`。所有源码、wheel 和权重固定版本、来源、许可证及 SHA-256。运行代码禁止 torch.hub、自动下载和运行时 pip；缺失或校验失败立即报错。

离线安装使用 `pip install --no-index --find-links third_party/wheelhouse/linux-aarch64 -r requirements.lock`。CANN、driver、firmware、torch 和 torch_npu 视为目标服务器预装组件；其余依赖进入 wheelhouse。由于当前开发机为 Windows/x86_64，aarch64 二进制依赖需在匹配目标 OS/Python 的联网 aarch64 环境下载或构建，并最终在目标服务器验收。

Grounding DINO、SAM2 和 VLM 不进入首版必需包。发布包包含环境检查、哈希验证、单样本 smoke test、8 卡启动脚本和完全断网安装说明。

## 8. 验证与验收

### 正确性

- 合成恒等/平移/边界样本验证 backward warp、flow 缩放、mask 公式及边界行为。
- RTX/CPU FP32 与 NPU Eager FP32 比较 flow、mask、各重建分支和 prediction；再比较混合精度与图模式。
- 同一样本在 8 张卡、不同 batch、视频 chunk 边界和断点恢复后应在设定容差内一致。
- 人工检查覆盖建筑断裂、头/武器缺失、路灯闪烁、鬼影、菜单/跳帧、大面积遮挡及可接受 UI/技能抖动。
- 连续合并覆盖重叠、首尾相邻、无效区间切断、跨视频禁止合并和整视频失败。

### 性能

- 分别记录 PNG decode、H2D、forward、D2H、CPU warp、候选检测、teacher、可视化和落盘吞吐。
- 单卡稳定预热后再测试 1/2/4/8 卡；8 卡端到端吞吐目标不低于单卡 6.5 倍。
- 输入队列有数据时，NPU 等待数据造成的空闲占比目标低于 10%；主模型路径不得出现隐式 CPU fallback。
- profiler 只在单 worker 的短稳定窗口开启，不在 8 卡生产全程采集。

### 离线交付

- 在禁止网络访问的干净环境完成安装、哈希校验、单样本推理、候选复核和断点恢复。
- 压缩包解压后不依赖项目目录外的模型文件；系统级 CANN/torch_npu 除外。
- 缺少可选 DINO/CGVQM 时，系统按配置明确降级到 FLIP、边缘/结构和 teacher，不静默改变判定逻辑。

## 9. 已锁定默认值

- 所有视频全量扫描；不限制每游戏入选数量。
- prediction 的明显局部错误只有下限；无效/超范围通过独立门控排除，不按“最严重误差”截断。
- teacher 不定义对错，只辅助可解性。
- 首版输出非语义原因标签；UI/文字/技能抖动低优先级。
- 原始连续帧合并落盘，诊断图与 manifest 分开保存。
- CPU 是 NPU 不支持算子和重指标的显式后端，不允许隐式 fallback。
