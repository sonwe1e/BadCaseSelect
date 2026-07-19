# 模型适配协议

## 工厂

配置项 `model.factory` 使用 `module:function`。每个设备 worker 在绑定设备后调用工厂一次：

```python
def create_current_model(*, checkpoint, device, **factory_kwargs):
    model = MyNetwork(...)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state["model"])
    return model.eval().to(device)
```

`checkpoint` 可以为空；`device` 是 `cpu`、`cuda:N` 或 `npu:N`。所有额外参数来自 `factory_kwargs`。工厂不得联网、调用 `torch.hub` 或修改全局设备。

## 调用与输出

模型接收两个 `[B,3,H,W]`、RGB `[0,1]` tensor：

```python
raw = model(img0, img1)
```

推荐返回命名字典：

```python
{
    "flow_t0": flow_t0,  # [B,2,h,w]
    "flow_t1": flow_t1,  # [B,2,h,w]
    "mask0": mask0,      # [B,1,h,w]，已经 sigmoid
    "mask1": mask1,      # [B,1,h,w]，已经 sigmoid
}
```

也可返回四元素序列，此时顺序由 `model.output_order` 明确指定。所有输出必须是有限浮点 tensor，batch 和空间尺寸必须一致，mask 值必须位于 `[0,1]` 容差内。

## 重建公式

flow 是网络输入 H×W 坐标系中的 backward pixel displacement。上采样至原图后：

```text
flow_x *= W_original / W_network
flow_y *= H_original / H_network
```

然后：

```text
warp0 = backward_warp(img0, flow_t0)
warp1 = backward_warp(img1, flow_t1)
```

当 `mask0_role=warp0_weight`：

```text
warp_blend = mask0 * warp0 + (1-mask0) * warp1
```

当 `mask0_role=warp1_weight`：

```text
warp_blend = (1-mask0) * warp0 + mask0 * warp1
```

最终残差分支固定为：

```text
prediction = mask1 * img1 + (1-mask1) * warp_blend
```

`mask0_role` 必须从真实网络代码或一次性全局校准确定，禁止逐样本根据 GT 选择。

## 上线前校准

目标机全量运行前必须通过：

1. 零 flow 恒等样本。
2. 已知水平/垂直平移样本。
3. `align_corners` 和 padding 边界样本。
4. mask0 两路对应关系。
5. RTX/CPU FP32 与 NPU Eager FP32 的逐张量比较。
