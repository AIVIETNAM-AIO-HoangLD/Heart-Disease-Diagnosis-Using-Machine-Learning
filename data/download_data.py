"""
Script to download Cleveland Heart Disease dataset from UCI Machine Learning Repository
or Google Drive backups.
"""

import os
import sys
from pathlib import Path
import urllib.request
import gdown

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import DATA_DIR, RAW_DATA_PATH, UCI_CLEVELAND_URL, GDOWN_DRIVE_ID_PART1

def download_from_uci(dest_path: Path) -> bool:
    """Download directly from UCI ML Repository."""
    try:
        print(f"[+] Downloading raw Cleveland data from UCI: {UCI_CLEVELAND_URL} ...")
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(UCI_CLEVELAND_URL, dest_path)
        print(f"[✓] Successfully downloaded to {dest_path}")
        return True
    except Exception as e:
        print(f"[!] Failed to download from UCI: {e}")
        return False

def download_from_gdrive(file_id: str, dest_path: Path) -> bool:
    """Download from Google Drive using gdown."""
    try:
        print(f"[+] Downloading via gdown (File ID: {file_id}) ...")
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, str(dest_path), quiet=False)
        print(f"[✓] Successfully downloaded to {dest_path}")
        return True
    except Exception as e:
        print(f"[!] Failed to download via gdown: {e}")
        return False

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_DATA_PATH.exists() and RAW_DATA_PATH.stat().st_size > 0:
        print(f"[✓] Raw dataset already exists at {RAW_DATA_PATH}")
        return

    success = download_from_uci(RAW_DATA_PATH)
    if not success:
        print("[*] Trying fallback download via Google Drive...")
        download_from_gdrive(GDOWN_DRIVE_ID_PART1, RAW_DATA_PATH)

if __name__ == "__main__":
    main()
