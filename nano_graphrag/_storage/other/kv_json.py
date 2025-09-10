import os
from dataclasses import dataclass

from ..._utils import load_json, logger, write_json
from ...base import (
    BaseKVStorage,
)


@dataclass
class JsonKVStorage(BaseKVStorage):
    def __post_init__(self):
        working_dir = self.global_config["working_dir"]
        self._file_name = os.path.join(working_dir, f"kv_store_{self.namespace}.json")
        self._data = load_json(self._file_name) or {}
        logger.info(f"Load KV {self.namespace} with {len(self._data)} data")

    async def all_keys(self) -> list[str]:
        return list(self._data.keys())

    async def index_done_callback(self):
        write_json(self._data, self._file_name)

    async def get_by_id(self, id):
        return self._data.get(id, None)

    async def get_by_ids(self, ids, fields=None):
        if fields is None:
            return [self._data.get(id, None) for id in ids]
        return [
            (
                {k: v for k, v in self._data[id].items() if k in fields}
                if id in self._data and self._data[id] is not None
                else None
            )
            for id in ids
        ]

    async def filter_keys(self, data: list[str]) -> set[str]:
        return set([s for s in data if s not in self._data])

    async def upsert(self, data: dict[str, dict]):
        self._data.update(data)

    async def drop(self):
        self._data = {}
    
    # 添加标准接口方法
    async def get(self, key: str, default=None):
        """标准get接口"""
        return await self.get_by_id(key) or default
    
    async def put(self, key: str, value: dict):
        """标准put接口"""
        await self.upsert({key: value})
    
    async def delete(self, key: str):
        """标准delete接口"""
        if key in self._data:
            del self._data[key]
