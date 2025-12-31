import torch
from torch.utils.data import Dataset
import polars as pl
from typing import Dict, Any, List


class YambdaDataset(Dataset):
    """
    PyTorch Dataset for sequential user interaction histories.
    """

    def __init__(
        self,
        dataframe: pl.DataFrame,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        self.df = dataframe
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return self.df.shape[0]

    def __getitem__(self, index: int) -> Dict[str, Dict[str, Any]]:
        row = self.df.row(index=index, named=True)
        items = row["item_id"]

        if len(items) > self.max_seq_len + 1:
            items = items[-(self.max_seq_len + 1) :]

        history_items = items[:-1]
        label_items = items[1:]
        length = len(history_items)
        positions = list(range(length))

        return {
            "history": {
                "item_id": list(map(int, history_items)),
                "lengths": int(length),
                "positions": positions,
            },
            "labels": {
                "item_id": list(map(int, label_items)),
            },
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collates a batch of samples into a single batched tensor representation.
    """
    lengths = torch.tensor(
        [sample["history"]["lengths"] for sample in batch], dtype=torch.long
    )
    history_item_ids = torch.tensor(
        [x for sample in batch for x in sample["history"]["item_id"]], dtype=torch.long
    )
    history_positions = torch.tensor(
        [x for sample in batch for x in sample["history"]["positions"]],
        dtype=torch.long,
    )
    label_item_ids = torch.tensor(
        [x for sample in batch for x in sample["labels"]["item_id"]], dtype=torch.long
    )

    return {
        "history": {
            "item_id": history_item_ids,
            "lengths": lengths,
            "positions": history_positions,
        },
        "labels": {
            "item_id": label_item_ids,
        },
    }
