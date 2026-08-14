from __future__ import annotations

import os
from pathlib import Path

import requests

BLENDERKIT_DATA = Path(os.path.expanduser("~/blenderkit_data"))


class BlenderKitAssets:
    def __init__(self, root: Path = BLENDERKIT_DATA):
        self.root = root
        self._id_cache: dict[str, str | None] = {}

    def resolve(self, asset_id: str, asset_type: str) -> Path | None:
        """Find the .blend file for a BlenderKit asset in the local cache."""
        type_dir = self.root / f"{asset_type}s"
        if not type_dir.exists():
            return None
        hit = self._scan(type_dir, asset_id)
        if hit is not None:
            return hit
        folder_id = self._lookup_folder_id(asset_id)
        if folder_id and folder_id != asset_id:
            return self._scan(type_dir, folder_id)
        return None

    def _scan(self, type_dir: Path, needle: str) -> Path | None:
        for folder in type_dir.iterdir():
            if folder.is_dir() and needle in folder.name:
                for f in folder.iterdir():
                    if f.suffix == ".blend":
                        return f
        return None

    def _lookup_folder_id(self, asset_base_id: str) -> str | None:
        if asset_base_id in self._id_cache:
            return self._id_cache[asset_base_id]
        folder_id: str | None = None
        try:
            url = f"https://www.blenderkit.com/api/v1/search/?query=asset_base_id:{asset_base_id}"
            resp = requests.get(url, timeout=10)
            if resp.ok and resp.json().get("results"):
                folder_id = resp.json()["results"][0].get("id")
        except Exception:
            folder_id = None
        self._id_cache[asset_base_id] = folder_id
        return folder_id
