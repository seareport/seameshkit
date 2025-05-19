from __future__ import annotations

import abc
import logging
import pathlib
import typing as T
import xarray as xr

from seameshkit._mappings import normalize_format
from seameshkit._mappings import MESH_FORMATS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class MeshAdapterRegistry:
    _adapters: T.Dict[MESH_FORMATS, T.Type[MeshAdapter]] = {}

    @classmethod
    def register(cls, format_type: MESH_FORMATS):
        def decorator(adapter_cls):
            cls._adapters[format_type] = adapter_cls
            return adapter_cls

        return decorator

    @classmethod
    def get_adapter(
        cls, format_type: T.Union[MESH_FORMATS, str]
    ) -> T.Optional[T.Type[MeshAdapter]]:
        fmt = normalize_format(format_type)
        return cls._adapters.get(fmt)

    @classmethod
    def list_formats(cls) -> T.List[MESH_FORMATS]:
        return list(cls._adapters.keys())


class MeshAdapter(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def can_read(cls, path: T.Union[str, pathlib.Path]) -> bool:
        pass

    @classmethod
    @abc.abstractmethod
    def read(cls, path: T.Union[str, pathlib.Path]) -> xr.Dataset:
        pass

    @classmethod
    @abc.abstractmethod
    def write(cls, ds: xr.Dataset, path: T.Union[str, pathlib.Path]) -> None:
        pass
