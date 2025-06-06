"""
drive_utils.py – updated for separate production area folders
-------------------------------------------------------------
• Dev mode  : pulls newest PNG/JPG from sample_frames/ and logs each match
• Prod mode : pulls newest file per region from separate Google Drive folders
              and logs query, candidate list, chosen file, local path
• Both modes: returns {region: {"path": str, "id": str|None}}
• delete_files()  – bulk-deletes Drive IDs with full logging
"""

import os, glob, logging
from pathlib import Path
from typing import Dict, List, Optional

try:
    from pydrive.auth import GoogleAuth
    from pydrive.drive import GoogleDrive
except ModuleNotFoundError:
    GoogleAuth = GoogleDrive = None

logger = logging.getLogger("drive_utils")

# --------------------------------------------------------------------------------
# CONFIG – regions and their corresponding Drive folder IDs
# --------------------------------------------------------------------------------
REGIONS = ["powder_booth", "general_labor", "sandblast"]

# Map each region to its specific Google Drive folder ID
FOLDER_IDS = {
    "powder_booth":  "1Dlj9h4Dt6_E99yYqISd4mbjKTsQLARL1",
    "general_labor": "1Lea9p_Cbw9owS31eKCafwhFY8Y_S_tUz", 
    "sandblast":     "1bWh6L7wHP4NxCM0x2qGRx12GFpco0Qhv",
}

# --------------------------------------------------------------------------------
# AUTH
# --------------------------------------------------------------------------------
def authenticate_drive() -> Optional["GoogleDrive"]:
    if GoogleAuth is None:
        logger.warning("PyDrive not installed – running in LOCAL mode only.")
        return None

    try:
        gauth = GoogleAuth()
        gauth.LocalWebserverAuth()
        logger.info("✅  Google Drive OAuth complete.")
        return GoogleDrive(gauth)
    except Exception as exc:
        logger.exception("❌  Drive authentication failed (%s) – LOCAL mode only.", exc)
        return None

# --------------------------------------------------------------------------------
# LOCAL fallback
# --------------------------------------------------------------------------------
LOCAL_DIR = Path("sample_frames")

def _latest_local_images() -> Dict[str, Dict]:
    """
    Return newest *.png / *.jpg per region from LOCAL_DIR.
    """
    latest: Dict[str, Dict] = {}

    for region in REGIONS:
        candidates = (
            sorted(LOCAL_DIR.glob(f"{region}_*.png"))
            + sorted(LOCAL_DIR.glob(f"{region}_*.jpg"))
        )
        if not candidates:
            logger.debug("⏭️  LOCAL: no files found for %s", region)
            continue

        newest = candidates[-1]  # lexicographic order == timestamp order
        latest[region] = {"path": str(newest), "id": None}
        logger.debug("📂  LOCAL: %s → %s", region, newest.name)

    return latest

# --------------------------------------------------------------------------------
# Main fetch helper - UPDATED for separate folders
# --------------------------------------------------------------------------------
def download_latest_images(
    drive: Optional["GoogleDrive"],
    folder_id: str = None,  # This parameter is now ignored but kept for compatibility
    out_dir: str = "latest_images",
) -> Dict[str, Dict]:
    """
    Returns {region: {"path": local_path, "id": drive_file_id|None}}
    Now searches each region's specific folder instead of a single shared folder.
    """
    if drive is None:
        logger.debug("🔄  Using LOCAL file mode.")
        return _latest_local_images()

    os.makedirs(out_dir, exist_ok=True)
    result: Dict[str, Dict] = {}

    for region in REGIONS:
        region_folder_id = FOLDER_IDS.get(region)
        if not region_folder_id:
            logger.warning("❌  No folder ID configured for region: %s", region)
            continue

        # Updated query to search in the region-specific folder
        query = (
            f"'{region_folder_id}' in parents and trashed=false "
            f"and mimeType contains 'image/' "
            f"and title contains '{region}_'"
        )
        logger.debug("🔎  Drive query for %s: %s", region, query)

        try:
            files = drive.ListFile({"q": query, "maxResults": 100}).GetList()
        except Exception as exc:
            logger.exception("❌  Drive list error for %s : %s", region, exc)
            continue

        if not files:
            logger.debug("⏭️  Drive: no files found for %s in folder %s", region, region_folder_id)
            continue

        # Get the newest file (by filename which should contain timestamp)
        newest = max(files, key=lambda f: f["title"])
        logger.debug(
            "🆕  Drive candidates for %s → picked %s (modified %s)",
            region, newest["title"], newest["modifiedDate"]
        )

        local_path = os.path.join(out_dir, newest["title"])
        try:
            newest.GetContentFile(local_path)
            result[region] = {"path": local_path, "id": newest["id"]}
            logger.debug("⬇️  Downloaded %s → %s", newest["title"], local_path)
        except Exception as exc:
            logger.exception("❌  Failed to download %s : %s", newest["title"], exc)

    logger.info("📊  Downloaded images for %d/%d regions", len(result), len(REGIONS))
    return result

# --------------------------------------------------------------------------------
# Deletion helper - UPDATED for separate folders
# --------------------------------------------------------------------------------
def delete_files(drive, file_ids):
    """
    Delete files from Google Drive by their file IDs.
    Works the same regardless of which folder they're in.
    """
    if drive is None:
        logger.debug("LOCAL mode – skip deletion of %d IDs", len(file_ids))
        return
        
    deleted_count = 0
    for fid in file_ids:
        try:
            drive.CreateFile({"id": fid}).Delete()
            logger.debug("🗑️  Deleted Drive file %s", fid)
            deleted_count += 1
        except Exception as exc:
            # swallow "File not found" 404s, log others
            if "404" in str(exc) and "File not found" in str(exc):
                logger.debug("Drive file already gone: %s", fid)
                deleted_count += 1  # Count as successful since it's gone
            else:
                logger.exception("❌  Failed to delete %s : %s", fid, exc)
    
    logger.info("🗑️  Deleted %d/%d Drive files", deleted_count, len(file_ids))

# --------------------------------------------------------------------------------
# Helper function to get folder ID for a region
# --------------------------------------------------------------------------------
def get_folder_id_for_region(region: str) -> Optional[str]:
    """
    Get the Google Drive folder ID for a specific region.
    """
    return FOLDER_IDS.get(region)

# --------------------------------------------------------------------------------
# Upload helper for the capture script
# --------------------------------------------------------------------------------
def upload_image_to_region_folder(drive: "GoogleDrive", region: str, 
                                  image_data: bytes, filename: str) -> bool:
    """
    Upload image data to the specific folder for a region.
    Returns True if successful, False otherwise.
    """
    folder_id = FOLDER_IDS.get(region)
    if not folder_id:
        logger.error("❌  No folder ID configured for region: %s", region)
        return False
    
    try:
        gfile = drive.CreateFile({
            "title": filename, 
            "parents": [{"id": folder_id}]
        })
        gfile.content = image_data
        gfile.Upload()
        logger.info("✅  Uploaded %s to %s folder (%s)", filename, region, folder_id)
        return True
    except Exception as exc:
        logger.exception("❌  Failed to upload %s to %s: %s", filename, region, exc)
        return False