import pandas as pd
import torch
from torch.utils.data import Dataset


class WSIBagDataset(Dataset):
    """Dataset for pre-extracted WSI bags.

    Each row in the CSV should contain: slide_id, path, label.
    Each .pt file should contain features, fm_features, coords, optional patches, and label.
    """

    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        item = torch.load(row["path"], map_location="cpu")
        label = int(row["label"]) if "label" in row else int(item["label"])
        return {
            "slide_id": row.get("slide_id", str(idx)),
            "features": item["features"].float(),
            "fm_features": item["fm_features"].float(),
            "coords": item["coords"].float(),
            "patches": item.get("patches", None),
            "label": torch.tensor(label, dtype=torch.long),
        }
