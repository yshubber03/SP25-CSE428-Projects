import os
import numpy as np
import argparse
import pickle
import json
import csv
from tqdm import tqdm
from scipy.stats import pearsonr
from scipy.sparse import coo_matrix


def load_ground_truth(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    target = data.get("2d_target")
    if isinstance(target, coo_matrix):
        target = target.toarray()
    return np.nan_to_num(target)

def compute_metrics(pred, target):
    pred = pred.flatten()
    target = target.flatten()
    mse = np.mean((pred - target) ** 2)
    try:
        pcc = pearsonr(pred, target)[0]
    except Exception:
        pcc = float('nan')
    return float(mse), float(pcc)

def evaluate(pred_dir, data_dir, output_json):
    records = []

    for fname in tqdm(os.listdir(pred_dir)):
        if not fname.endswith("_prediction.npy"):
            continue

        base_name = fname.replace("_prediction.npy", "")
        pred_path = os.path.join(pred_dir, fname)
        pkl_path = os.path.join(data_dir, base_name)

        if not os.path.exists(pkl_path):
            print(f"Missing .pkl file for {base_name}, skipping.")
            continue

        pred = np.load(pred_path)
        target = load_ground_truth(pkl_path)

        if pred.shape != target.shape:
            print(f"Shape mismatch for {base_name}: pred {pred.shape} vs target {target.shape}")
            continue

        mse, pcc = compute_metrics(pred, target)
        records.append({"filename": base_name, "mse": mse, "pcc": pcc})

    # Save JSON
    with open(output_json, 'w') as f:
        json.dump(records, f, indent=2)
    print(f"Saved evaluation summary to {output_json}")

    # Save CSV
    output_csv = output_json.rsplit('.', 1)[0] + ".csv"
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "mse", "pcc"])
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved evaluation summary to {output_csv}")

    # Compute and save overall summary
    mse_values = [r["mse"] for r in records]
    pcc_values = [r["pcc"] for r in records if not np.isnan(r["pcc"])]

    summary_stats = {
        "mean_mse": float(np.mean(mse_values)),
        "std_mse": float(np.std(mse_values)),
        "mean_pcc": float(np.mean(pcc_values)),
        "std_pcc": float(np.std(pcc_values)),
        "n_files": len(records)
    }

    summary_json = output_json.rsplit('.', 1)[0] + "_summary.json"
    with open(summary_json, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    print(f"Saved overall summary to {summary_json}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", required=True, help="Directory with prediction .npy files")
    parser.add_argument("--data_dir", required=True, help="Directory with original .pkl files")
    parser.add_argument("--output_json", default="evaluation_summary.json", help="JSON file to save metrics")
    args = parser.parse_args()

    evaluate(args.pred_dir, args.data_dir, args.output_json)

if __name__ == "__main__":
    main()
