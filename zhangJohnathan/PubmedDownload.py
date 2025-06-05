import os
import sys
import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

PICKLE_PATH = "pubmed_data.pkl"
OUTPUT_ROOT = os.path.expanduser("~/Downloads/premed_images_cached")
IMAGES_DIR  = os.path.join(OUTPUT_ROOT, "images")

os.makedirs(IMAGES_DIR, exist_ok=True)
PMC_HTML_BASE = "https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; FigureDownloader/1.0; +https://example.com/)"
}

def find_cdn_image_url(pmcid: str, href_prefix: str) -> str | None:
    url = PMC_HTML_BASE.format(pmcid=pmcid)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        return None

    html = resp.text
    soup = BeautifulSoup(html, "html.parser")
    # look at all img tags
    for img in soup.find_all("img"):
        src = img.get("src", "")
        if "cdn.ncbi.nlm.nih.gov/pmc/" not in src:
            continue
        filename = os.path.basename(src)
        if href_prefix in filename:
            if src.startswith("//"):
                src = "https:" + src
            return src
    return None

def main():
    try:
        df = pd.read_pickle(PICKLE_PATH)
    except Exception as e:
        print(f"ERROR: could not load '{PICKLE_PATH}': {e}")
        sys.exit(1)

    df = df[["PMCID", "href", "label_text"]].rename(
        columns={"href": "href_prefix", "label_text": "caption"}
    )

    records = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Downloading figures"):
        pmcid = row["PMCID"]
        href_prefix = row["href_prefix"]
        caption = row["caption"]

        cdn_url = find_cdn_image_url(pmcid, href_prefix)
        if cdn_url is None:
            records.append({
                "pmcid": pmcid,
                "href_prefix": href_prefix,
                "image_url": "",
                "caption": caption,
                "status_code": None,
                "error": "NoMatchingCDNImage"
            })
            continue

        try:
            r_img = requests.get(cdn_url, headers=HEADERS, timeout=60)
            status = r_img.status_code
            if status == 200:
                # Write the file under images/<basename>
                basename = os.path.basename(cdn_url.split("?")[0])
                outpath = os.path.join(IMAGES_DIR, basename)
                with open(outpath, "wb") as f:
                    f.write(r_img.content)

                records.append({
                    "pmcid": pmcid,
                    "href_prefix": href_prefix,
                    "image_url": cdn_url,
                    "caption": caption,
                    "status_code": 200,
                    "error": ""
                })
            else:
                records.append({
                    "pmcid": pmcid,
                    "href_prefix": href_prefix,
                    "image_url": cdn_url,
                    "caption": caption,
                    "status_code": status,
                    "error": f"HTTP_{status}"
                })
        except Exception as e:
            records.append({
                "pmcid": pmcid,
                "href_prefix": href_prefix,
                "image_url": cdn_url,
                "caption": caption,
                "status_code": None,
                "error": f"Download_failed: {e}"
            })

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    log_df = pd.DataFrame(records)

    full_log = os.path.join(OUTPUT_ROOT, "all_image_download_records.csv")
    log_df.to_csv(full_log, index=False)
    print("Wrote full log to:", full_log)

    success_df = log_df[log_df["status_code"] == 200]
    labels_csv = os.path.join(OUTPUT_ROOT, "image_labels.csv")
    success_df[["pmcid", "href_prefix", "image_url", "caption"]].to_csv(labels_csv, index=False)
    print("Wrote successful CSV to:", labels_csv)

    print(f"\nDone! {len(success_df)} images downloaded successfully. {len(log_df)-len(success_df)} failures.")
    print("Check files under:", IMAGES_DIR)


if __name__ == "__main__":
    main()
