"""ExMed-BERT package.

Subpackages are intentionally not imported eagerly.  The legacy data package
pulls in optional classical-ML dependencies that are unnecessary for model-only
AI Runtime workloads.
"""

from importlib import import_module

__all__ = ["data", "models"]


def __getattr__(name):
    if name in __all__:
        return import_module(f"exmed_bert.{name}")
    raise AttributeError(name)
