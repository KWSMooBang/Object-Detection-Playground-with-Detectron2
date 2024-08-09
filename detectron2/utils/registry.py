import pydoc
from typing import Any
from fvcore.common.registry import Registry

"""
'Registry' and 'locate' provide ways to map 
a string (typically found in config files) to callable objects
"""

__all__ = [
    'Registry',
    'locate'
]


def _convert_target_to_string(t: Any) -> str:
    """
    Inverse of 'locate()

    Args:
        t: any object with '__module__' and '__qualname__'
    """
    module, qualname = t.__module__, t.__qualname__

    module_parts = module.split('.')
    for k in range(1, len(module_parts)):
        prefix = '.'.join(module_parts[:k])
        candidate = f"{prefix}.{qualname}"
        try:
            if locate(candidate) is t:
                return candidate
        except ImportError:
            pass
    return f"{module}.{qualname}"


def locate(name: str) -> Any:
    """
    Locate and return an object ``x`` using an input string ``{x.__module__}.{x.__qualname__}``,
    such as "module.submodule.class_name".

    Raise Exception if it cannot be found.
    """
    obj = pydoc.locate(name)

    # Some cases (e.g. torch.optim.sgd.SGD) not handled correctly
    # by pydoc.locate. Try a private function from hydra.
    if obj is None:
        try:
            # from hydra.utils import get_method - will print many errors
            from hydra.utils import _locate
        except ImportError as e:
            raise ImportError(f"Cannot dynamically locate object {name}!") from e
        else:
            obj = _locate(name)  # it raises if fails

    return obj