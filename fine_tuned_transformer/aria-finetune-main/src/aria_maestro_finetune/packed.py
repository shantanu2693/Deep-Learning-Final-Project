from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class PackedConfig:
    block_size: int


class PackedBlocksDataset(Dataset):
    """
    Dataset backed by a memory-mapped `.npy` of shape [N, block_size] (int32).
    Each row is a contiguous block of token ids.
    """

    def __init__(self, npy_path: str | Path):
        self.npy_path = Path(npy_path)
        self._arr = np.load(self.npy_path, mmap_mode="r")
        if self._arr.ndim != 2:
            raise ValueError(f"Expected 2D array at {self.npy_path}, got shape {self._arr.shape}")

    @property
    def block_size(self) -> int:
        return int(self._arr.shape[1])

    def __len__(self) -> int:
        return int(self._arr.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self._arr[idx].astype(np.int64, copy=False)
        input_ids = torch.from_numpy(row)
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
            "attention_mask": torch.ones_like(input_ids),
        }


def load_stats(stats_path: str | Path) -> dict[str, Any]:
    return json.loads(Path(stats_path).read_text())


def simple_collate(features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    # All blocks are fixed length, so plain stacking is fine.
    batch: dict[str, torch.Tensor] = {}
    for k in features[0].keys():
        batch[k] = torch.stack([f[k] for f in features], dim=0)
    return batch

