from __future__ import annotations

import argparse
import os
import sys
import tempfile
import uuid
import zipfile
from pathlib import Path

import requests

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

BLENDERKIT_DATA = Path.home() / "blenderkit_data"
SCENE_UUID = str(uuid.uuid4())


def get_asset_info(asset_id: str) -> dict:
    url = f"https://www.blenderkit.com/api/v1/search/?query=asset_base_id:{asset_id}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    results = resp.json().get("results", [])
    if not results:
        raise ValueError(f"Asset not found or no longer listed by BlenderKit: {asset_id}")
    return results[0]


def get_download_url(asset_data: dict, api_key: str = None) -> str | None:
    files = asset_data.get("files", [])
    blend_file = next((f for f in files if f.get("fileType") == "blend"), None)

    if not blend_file or not blend_file.get("downloadUrl"):
        return None

    download_endpoint = blend_file["downloadUrl"]
    url_with_uuid = f"{download_endpoint}?scene_uuid={SCENE_UUID}"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    resp = requests.get(url_with_uuid, headers=headers, timeout=30)
    if resp.status_code in {401, 403}:
        raise PermissionError(
            "BlenderKit denied this download. Use your own API key and ensure "
            "your account has the required access plan."
        )
    resp.raise_for_status()

    return resp.json().get("filePath")


def download_asset(asset_id: str, asset_type: str, api_key: str = None, dest_root: Path | None = None) -> Path:
    print(f"Fetching asset info for {asset_id}...")
    asset_data = get_asset_info(asset_id)

    name = asset_data.get("name", "Unknown")
    folder_id = asset_data.get("id", asset_id)
    print(f"Asset: {name}")

    if asset_data.get("isFree") is False and not api_key:
        raise PermissionError(
            f"{name} requires paid BlenderKit access. Set BLENDERKIT_API_KEY to a key from your own eligible account."
        )

    root = dest_root if dest_root is not None else BLENDERKIT_DATA
    type_dir = root / f"{asset_type}s"
    if type_dir.exists():
        for folder in type_dir.iterdir():
            if folder.is_dir() and (asset_id in folder.name or folder_id in folder.name):
                for f in folder.iterdir():
                    if f.suffix == ".blend":
                        print(f"Already downloaded: {f}")
                        return f

    download_url = get_download_url(asset_data, api_key)
    if not download_url:
        raise ValueError("Cannot get download URL - provide --api-key for authenticated download")

    print("Downloading...")
    resp = requests.get(download_url, stream=True, timeout=300)
    resp.raise_for_status()

    safe_name = "".join(c if c.isalnum() or c in "._-" else "-" for c in name).strip("-")
    out_dir = type_dir / f"{safe_name}_{asset_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    content_type = resp.headers.get("content-type", "")

    if "zip" in content_type or download_url.endswith(".zip"):
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            for chunk in resp.iter_content(chunk_size=8192):
                tmp.write(chunk)
            tmp_path = tmp.name

        with zipfile.ZipFile(tmp_path, "r") as zf:
            zf.extractall(out_dir)
        os.unlink(tmp_path)
    else:
        out_path = out_dir / f"{safe_name}.blend"
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

    for f in out_dir.rglob("*.blend"):
        print(f"Downloaded: {f}")
        return f

    raise ValueError("Download completed but no .blend file found")


def download_configured_assets(api_key: str | None = None, dest_root: Path | None = None) -> list[Path]:
    """Download every asset referenced by the bundled simulator configuration."""
    from rigidbench.scenes.schema.registry import get_registry

    refs = get_registry().asset_refs()
    downloaded: list[Path] = []
    failures: list[str] = []
    for index, ref in enumerate(refs, start=1):
        print(f"\n[{index}/{len(refs)}] {ref.type} {ref.id}")
        try:
            downloaded.append(download_asset(ref.id, ref.type, api_key, dest_root))
        except Exception as error:
            failures.append(f"{ref.type} {ref.id}: {error}")
    if failures:
        preview = "\n".join(f"  - {failure}" for failure in failures)
        raise RuntimeError(f"Could not download {len(failures)} configured asset(s):\n{preview}")
    return downloaded


def main():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]

    parser = argparse.ArgumentParser(description="Download BlenderKit assets used by the simulator.")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--all",
        action="store_true",
        help="Download every configured asset; some require paid BlenderKit access",
    )
    selection.add_argument("--asset-id", help="Download one BlenderKit asset ID")
    parser.add_argument("--type", choices=["scene", "model", "material"], default="scene")
    parser.add_argument(
        "--api-key",
        help="API key from your BlenderKit account (or set BLENDERKIT_API_KEY)",
    )
    parser.add_argument("--dest", help="Override download root (default: ~/blenderkit_data)")
    args = parser.parse_args(argv)

    api_key = args.api_key or os.environ.get("BLENDERKIT_API_KEY")
    dest_root = Path(args.dest) if args.dest else None

    try:
        if args.all:
            paths = download_configured_assets(api_key, dest_root)
            print(f"\nReady: {len(paths)} configured assets")
        else:
            path = download_asset(args.asset_id, args.type, api_key, dest_root=dest_root)
            print(f"\nSuccess: {path}")
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
