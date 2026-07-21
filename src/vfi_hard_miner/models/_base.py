"""Typing contract for VFI models in the models/ package.

Import this only for type annotations or ``isinstance`` checks — model
classes do **not** need to inherit from anything here.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class VFIModelContract(Protocol):
    """Runtime-checkable interface that every VFI model must satisfy.

    ``ModelAdapter`` only requires the model to be callable with
    ``(img0, img1) -> dict | sequence``, so no explicit base class is
    needed.  This Protocol exists for static analysis and sanity checks.
    """

    def __call__(
        self,
        img0: torch.Tensor,
        img1: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run one forward pass.

        Parameters
        ----------
        img0, img1:
            ``[B, 3, H, W]`` float32, RGB normalized to ``[0, 1]``.
            ``ModelAdapter`` handles resizing to the network's fixed
            ``(input_height, input_width)`` before calling you, so
            ``H`` and ``W`` here are always the network dimensions.

        Returns
        -------
        dict with exactly these keys (all tensors on the same device as
        the inputs):

        ``flow_t0``  ``[B, 2, h, w]``
            Optical flow from frame *t = 0* to the midpoint.
        ``flow_t1``  ``[B, 2, h, w]``
            Optical flow from frame *t = 1* to the midpoint.
        ``mask0``    ``[B, 1, h, w]``
            Blending weight from *t = 0*; already sigmoid-normalized to
            ``[0, 1]``.
        ``mask1``    ``[B, 1, h, w]``
            Blending weight from *t = 1*; already sigmoid-normalized to
            ``[0, 1]``.

        The spatial size ``(h, w)`` may differ from ``(H, W)`` (e.g.
        quarter-resolution outputs are fine).  All four tensors must
        share the same ``(h, w)``.
        """
        ...


class ModelFactory(Protocol):
    """Signature every ``create_model`` factory function must follow."""

    def __call__(
        self,
        *,
        checkpoint: str | None,
        device: torch.device,
        **kwargs: object,
    ) -> VFIModelContract:
        """Construct the model, load weights, and return it eval-mode.

        Parameters
        ----------
        checkpoint:
            Path to a weight file, or ``None`` if the model is
            randomly initialized / weights are baked in.
        device:
            Target device (already resolved by ``ModelAdapter``).
        **kwargs:
            Forwarded verbatim from ``model.factory_kwargs`` in the
            config.  Declare them as explicit keyword arguments so IDEs
            and type-checkers can catch mismatches.

        Returns
        -------
        The model, already in ``eval()`` mode and on *device*.
        (Call ``model.eval().to(device)`` before returning.)
        """
        ...
