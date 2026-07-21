"""Template for adding a new VFI model to this package.

Quick-start
-----------
1. Copy this file and rename it to ``<your_model>.py``
   (lowercase, underscores OK — the filename becomes the config key).
2. Rename ``TemplateModel`` → ``<YourModel>Model``.
3. Fill in every ``# TODO`` block.
4. In your JSON config, set ``"factory": "<your_model>"``.

That's it.  No registration step, no changes to any other file.

Example
-------
File saved as ``models/rife.py``, config entry::

    "model": {
        "factory": "rife",
        "checkpoint": "ckpts/rife.pth",
        "input_height": 540,
        "input_width":  960,
        "batch_size":   4,
        "factory_kwargs": {}
    }

Calling the model
-----------------
Through ``ModelAdapter`` (normal pipeline use)::

    adapter = ModelAdapter.from_config(config.model, device=device)
    outputs = adapter(img0, img1)   # returns ModelOutputs dataclass

Standalone (custom scripts, evaluation)::

    model = create_model(checkpoint=..., device=device)
    outputs = model.infer(img0, img1)   # dict with flow_t0/t1, mask0/1
"""

from __future__ import annotations

import torch
from torch import nn


class TemplateModel(nn.Module):
    """Replace this docstring with a one-line description of your model."""

    def __init__(
        self,
        *,
        # TODO: add your model-specific constructor parameters here.
        #       Keep them keyword-only (after the bare *) and give each
        #       a type annotation and a sensible default where possible.
        #
        # Example:
        #   base_channels: int = 32,
        #   use_attention: bool = True,
    ) -> None:
        super().__init__()
        # TODO: build the network layers.
        raise NotImplementedError("fill in __init__ before using this model")

    def forward(
        self,
        img0: torch.Tensor,
        img1: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Estimate midpoint optical flow and blending masks.

        Called by ``ModelAdapter`` after it has already resized inputs to
        ``(input_height, input_width)`` and moved them to the target device.
        Do not add ``torch.no_grad()`` or device transfers here.

        Parameters
        ----------
        img0, img1:
            ``[B, 3, H, W]`` float32, RGB normalized to ``[0, 1]``.

        Returns
        -------
        dict with keys ``flow_t0``, ``flow_t1``, ``mask0``, ``mask1``:

        ``flow_t0``  ``[B, 2, h, w]``  backward flow from frame 0 to midpoint
        ``flow_t1``  ``[B, 2, h, w]``  backward flow from frame 1 to midpoint
        ``mask0``    ``[B, 1, h, w]``  blending weight, **already** sigmoid-normalized to [0, 1]
        ``mask1``    ``[B, 1, h, w]``  blending weight, **already** sigmoid-normalized to [0, 1]

        ``(h, w)`` may differ from ``(H, W)`` (e.g. quarter resolution).
        All four tensors must share the same spatial shape and reside on
        the same device as the inputs.
        """
        # TODO: implement the forward pass.
        raise NotImplementedError("fill in forward before using this model")

    @torch.inference_mode()
    def infer(
        self,
        img0: torch.Tensor,
        img1: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run a single forward pass with gradient tracking disabled.

        Use this when calling the model directly (e.g. in a custom
        evaluation script).  ``ModelAdapter`` manages ``inference_mode``
        internally, so you do not need this method when going through
        the adapter.

        Parameters
        ----------
        img0, img1:
            ``[B, 3, H, W]`` float32 tensors already on the correct
            device, RGB normalized to ``[0, 1]``.  No resizing is
            applied — pass the network's fixed ``input_height ×
            input_width`` directly.
        """
        return self(img0, img1)


def create_model(
    *,
    checkpoint: str | None,
    device: torch.device,
    # TODO: mirror any extra constructor parameters here so they can be
    #       passed via "factory_kwargs" in the config.
    #
    # Example (matching the constructor above):
    #   base_channels: int = 32,
    #   use_attention: bool = True,
) -> TemplateModel:
    """Factory called by ``ModelAdapter`` — do not rename or change the signature.

    Parameters
    ----------
    checkpoint:
        Path to the weight file (``state_dict``), or ``None`` for a
        randomly initialized model (useful during development / smoke tests).
    device:
        Target device, already resolved from the runtime config.
    **extra kwargs:
        Forwarded verbatim from ``model.factory_kwargs`` in the config.
    """
    model = TemplateModel(
        # TODO: forward constructor arguments.
        #   base_channels=base_channels,
        #   use_attention=use_attention,
    )
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    return model.eval().to(device)
