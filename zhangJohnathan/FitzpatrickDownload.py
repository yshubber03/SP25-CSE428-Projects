import os
import pandas as pd
import requests
from tqdm import tqdm  # pip install tqdm

# 1) Define paths
USER_HOME       = os.path.expanduser("~")
FITZ_CSV_PATH   = os.path.join(USER_HOME, "Downloads", "fitzpatrick17k.csv")
TARGET_FITZ_DIR = os.path.join(USER_HOME, "Downloads", "fitzpatrick17k_images_cached2")

# 2) Make sure the target directory exists
os.makedirs(TARGET_FITZ_DIR, exist_ok=True)

# 3) Load the CSV; it must have columns "md5hash" and "url"
print(f"Reading Fitzpatrick17k CSV from:\n  {FITZ_CSV_PATH}")
df = pd.read_csv(FITZ_CSV_PATH)

# Verify required columns exist
if "md5hash" not in df.columns or "url" not in df.columns:
    raise ValueError("CSV must contain columns 'md5hash' and 'url'")

# 4) Download each image, saving with the md5hash + proper extension
success_count = 0
failure_count = 0

print(f"Downloading {len(df)} Fitzpatrick17k images into:\n  {TARGET_FITZ_DIR}")
for _, row in tqdm(df.iterrows(), total=len(df), desc="Downloading Fitzpatrick17k"):
    image_id = str(row["md5hash"]).strip()
    url      = str(row["url"]).strip()

    if not image_id or not url or pd.isna(url):
        failure_count += 1
        continue

    # Extract the extension from the URL (e.g. ".jpg", ".png", etc.)
    # If URL doesn't end in a known extension, default to ".jpg"
    ext = os.path.splitext(url)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif"]:
        ext = ".jpg"

    # Build the full save path, including extension
    save_filename = image_id + ext
    save_path     = os.path.join(TARGET_FITZ_DIR, save_filename)

    # Skip if already downloaded
    if os.path.isfile(save_path):
        success_count += 1
        continue

    try:
        # Send a browser‐like User-Agent header to avoid 406 errors
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(resp.content)
        success_count += 1
    except Exception as e:
        failure_count += 1
        # Print only the first few failures
        if failure_count <= 5:
            print(f"  ✗ Failed to download {url} as {save_filename}: {e}")

print(f"\nDownload complete. Successfully saved {success_count} images; {failure_count} failures.")
print("Example files in target directory:", os.listdir(TARGET_FITZ_DIR)[:5])
print("Total files in folder:", len(os.listdir(TARGET_FITZ_DIR)))
