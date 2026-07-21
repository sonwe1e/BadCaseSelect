"""Built-in VFI interpolation models.

Drop a new ``.py`` file here and reference it by name in your config —
no registration or code changes needed anywhere else.

Config usage
------------
Short name (recommended)::

    "model": { "factory": "rife", "checkpoint": "ckpts/rife.pth", ... }

  Resolves to ``vfi_hard_miner.models.rife:create_model`` automatically.

Explicit module path (unchanged behaviour)::

    "model": { "factory": "my_project.adapter:create_my_model", ... }

Listing available models
------------------------
>>> from vfi_hard_miner.models import list_models
>>> list_models()
['rife', 'raft_stereo', ...]
"""

from __future__ import annotations

import importlib
import pkgutil


def list_models() -> list[str]:
    """Return the names of all non-private models in this package.

    Each name corresponds to a ``.py`` file (excluding those starting with
    ``_``) and can be passed directly as ``model.factory`` in the config.
    """
    pkg = importlib.import_module(__name__)
    return sorted(
        module.name
        for module in pkgutil.iter_modules(pkg.__path__)
        if not module.name.startswith("_")
    )
