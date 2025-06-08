import os
import json
import random
import openslide
import multiprocessing
import numpy as np
from src.subtyping.roi_agent import ROIAgent
from src.subtyping.slide_utils import get_image_from_bbox, get_oncotree_code, calculate_f1_scores
import config
import src.subtyping.subtyping_prompt as prompt
from skimage.filters import threshold_otsu
import re
from collections import Counter
from PIL import Image, ImageDraw, ImageFont
from openai_client import get_openai_response_base64
from utils.file_utils import get_svs_files_from_folders, initialize_directories, get_svs_files_from_repo

# Initialize global variables from config
cancer_subtype_map = config.CANCER_SUBTYPE_MAP
cancer_folder_map = config.CANCER_FOLDER_MAP

def get_random_coordinates():
    x = random.uniform(0.1, 0.9)
    y = random.uniform(0.1, 0.9)
    return x, y

def process_slide(file_path, cancer_type, output_path, baseline_type, final_prompt):
    """
    Generalized method to process slides for different baselines.
    """
    file_name = os.path.basename(file_path)
    sample_id = os.path.basename(file_name).split('.')[0]
    image = openslide.OpenSlide(file_path)
    if baseline_type == "random":
        result = process_random_roi(image, sample_id, cancer_type, output_path, final_prompt)
        # result = run_majority_vote_random_baseline(image, sample_id, cancer_type, output_path, final_prompt)
    elif baseline_type == "gpt":
        result = process_gpt_selected_roi(image, sample_id, cancer_type, output_path, final_prompt)
    else:
        raise ValueError(f"Unknown baseline type: {baseline_type}")
    return result

def run_majority_vote_random_baseline(image, sample_id, cancer_type, output_path, final_prompt, n=21):
    vote_labels = []
    for i in range(n):
        single_result = process_random_roi(image, sample_id, cancer_type, output_path, final_prompt, str(i+1))
        if single_result and "predicted_label" in single_result:
            vote_labels.append(single_result["predicted_label"])
    if vote_labels:
        majority_label = Counter(vote_labels).most_common(1)[0][0]
        correct_label = get_oncotree_code(sample_id[:12])
        result = {
            "sample_id": sample_id,
            "predicted_label": majority_label,
            "correct_label": correct_label,
            "is_correct": majority_label == correct_label,
            "all_votes": vote_labels
        }
        with open(os.path.join(output_path, sample_id, "random_baseline_vote_result.json"), "w") as f:
            json.dump(result, f, indent=4)
        return result
    else:
        return None

def generate_non_blank_mask(image, downscale_size=(2048, 2048)):
    """
    Generate a binary mask of non-blank regions using a downscaled WSI.
    """
    thumbnail = image.get_thumbnail(downscale_size).convert("L")
    thumbnail_array = np.array(thumbnail)
    # Apply Otsu's method to binarize
    threshold = threshold_otsu(thumbnail_array)
    binary_mask = thumbnail_array <= threshold  # Non-blank regions are darker
    return binary_mask

def get_random_tissue_coordinates(binary_mask):
    """
    Randomly select a coordinate from the non-blank region.
    """
    tissue_coords = np.argwhere(binary_mask)
    if len(tissue_coords) == 0:
        raise ValueError("No tissue regions found in the WSI.")
    selected_coord = tissue_coords[np.random.randint(len(tissue_coords))]
    return selected_coord[1], selected_coord[0]  # x, y


def process_random_roi(image, sample_id, cancer_type, output_path, final_prompt):
    """
    Baseline 1: Randomly select a non-blank ROI for prediction.
    """
    os.makedirs(os.path.join(output_path, sample_id), exist_ok=True)
    binary_mask = generate_non_blank_mask(image)
    correct_label = get_oncotree_code(sample_id[:12])
    for _ in range(100):  # Limit maximum attempts
        try:
            x, y = get_random_tissue_coordinates(binary_mask)  # Select a random tissue region
            level = 0
            roi_path, _, _ = get_image_from_bbox(
                image, x / binary_mask.shape[1], y / binary_mask.shape[0], level,
                save_path=os.path.join(output_path, sample_id, "random_roi.png")
            )
            print(f"Selected ROI: ({x}, {y}) at level {level}")
            break
        except ValueError:
            print(f"No tissue found for {sample_id}. Skipping.")
            return None

    # Perform GPT-based prediction
    predicted_label = get_openai_response_base64(final_prompt, roi_path)

    if not predicted_label:
        print(f"WARNING: GPT failed to predict label for {sample_id}. Skipping.")
        return None

    sample_result =  {
        "sample_id": sample_id,
        "predicted_label": predicted_label,
        "correct_label": correct_label,
        "is_correct": predicted_label == correct_label,
        # "selected_roi": {"x": round(new_x / image_width, 2), "y": round(new_y / image_height, 2)},
    }

    sample_result_path = os.path.join(output_path, sample_id)
    os.makedirs(sample_result_path, exist_ok=True)
    with open(os.path.join(sample_result_path, "random_baseline_result.json"), "w") as f:
        json.dump(sample_result, f, indent=4)

    return sample_result

def parse_gpt_response(feedback, image_width, image_height):
    """
    Parse the GPT feedback to extract normalized coordinates (x, y)
    and convert them to absolute coordinates based on image dimensions.
    """
    import re
    try:
        # Use regex to extract x and y values in the format (x=..., y=...)
        match = re.search(r"x=([\d.]+), y=([\d.]+)", feedback)
        if match:
            # Extract normalized coordinates
            normalized_x = float(match.group(1))  # Ensure the assignment is complete
            normalized_y = float(match.group(2))  # Ensure no missing part

            # Convert normalized coordinates to absolute coordinates
            abs_x = int(normalized_x * image_width)
            abs_y = int(normalized_y * image_height)
            return abs_x, abs_y
        else:
            raise ValueError("No valid coordinates found in GPT response.")
    except Exception as e:
        print(f"Error parsing GPT response: {e}")
        return None

def process_gpt_selected_roi(image, sample_id, cancer_type, output_path, final_prompt):
    """
    Baseline 2: Generate 20 normalized coordinates, let GPT choose the best ROI, 
    and classify the selected ROI after parsing the response.
    """
    os.makedirs(os.path.join(output_path, sample_id), exist_ok=True)
    binary_mask = generate_non_blank_mask(image)
    image_width, image_height = binary_mask.shape[1], binary_mask.shape[0]
    correct_label = get_oncotree_code(sample_id[:12])
    # print(f"Image dimensions - Width: {image_width}, Height: {image_height}")

    candidate_coords = []
    for _ in range(20):
        try:
            x, y = get_random_tissue_coordinates(binary_mask)
            normalized_x = round(x / image_width, 2)
            normalized_y = round(y / image_height, 2)
            if isinstance(normalized_x, (int, float)) and isinstance(normalized_y, (int, float)):
                candidate_coords.append((normalized_x, normalized_y))
            else:
                print(f"Invalid coordinates generated: {(normalized_x, normalized_y)}")
        except ValueError as e:
            print(f"Failed to generate tissue coordinates: {e}")
            continue

    if not candidate_coords:
        print(f"No valid tissue coordinates found for {sample_id}. Skipping.")
        return None

    # Generate GPT prompt
    coords_text = ", ".join([f"(x={x:.2f}, y={y:.2f})" for x, y in candidate_coords])
    thumbnail_path = get_thumbnail(image, output_path=os.path.join(output_path, f"{sample_id}_thumbnail.png"))
    text_prompt = prompt.generate_prompt_for_coordinates(cancer_type, coords_text)

    try:
        best_point_response = get_openai_response_base64(text_prompt, thumbnail_path)
        selected_coord = parse_gpt_response(best_point_response, image_width, image_height)
    except Exception as e:
        print(f"Error in GPT response parsing: {e}")
        return None
    if not selected_coord:
        print(f"Failed to get valid coordinates from GPT for {sample_id}. Skipping.")
        return None
    new_x, new_y = selected_coord
    level = 0
    try:
        roi_path, _, _ = get_image_from_bbox(
            image, new_x / binary_mask.shape[1], new_y / binary_mask.shape[0], level,  # Use level 0 (highest resolution)
            save_path=os.path.join(output_path, sample_id, "gpt_selected_roi.png")
        )
    except Exception as e:
        print(f"Error extracting ROI: {e}")
        return None

    try:
        predicted_label = get_openai_response_base64(final_prompt, roi_path)
    except Exception as e:
        print(f"Error in GPT subtype prediction: {e}")
        return None

    sample_result =  {
        "sample_id": sample_id,
        "predicted_label": predicted_label,
        "correct_label": correct_label,
        "is_correct": predicted_label == correct_label,
        "selected_roi": {"x": round(new_x / image_width, 2), "y": round(new_y / image_height, 2)},
    }

    sample_result_path = os.path.join(output_path, sample_id)
    os.makedirs(sample_result_path, exist_ok=True)
    print(sample_result_path)
    with open(os.path.join(sample_result_path, "gpt_baseline_result.json"), "w") as f:
        json.dump(sample_result, f, indent=4)

    return sample_result

def get_thumbnail(image, thumbnail_size=(1024, 1024), output_path="thumbnail.png"):
    thumbnail = image.get_thumbnail(thumbnail_size)
    thumbnail.save(output_path)
    return output_path

def run_slide(args):
    file_name, cancer_type, output_path, baseline_type, final_prompt = args
    try:
        result = process_slide(file_name, cancer_type, output_path, baseline_type, final_prompt)
        return (file_name, result)
    except Exception as e:
        print(f"[ERROR] Subtype prediction failed for {file_name}: {e}")
        return (file_name, None)

def main(cancer_type, n=1, baseline_type="random", num_workers=5):
    if baseline_type == "random":
        output_path = os.path.join(config.OUTPUT_DIR, "subtyping", cancer_type, "majority_vote_baseline")
    else:
        output_path = os.path.join(config.OUTPUT_DIR, "subtyping", cancer_type, "baseline_output")
    subtypes = cancer_subtype_map[cancer_type]
    svs_files = get_svs_files_from_folders(config.CANCER_FOLDER_MAP, cancer_type)
    # if cancer_type == "HEP":
    #     svs_files = get_svs_files_from_repo("TCGA-CHOL")
    #     print(len(svs_files))
    results = []

    # Process each slide
    if n > 0:
        svs_files = random.sample(svs_files, min(n, len(svs_files)))
    total_files = len(svs_files)
    final_prompt = prompt.get_final_prompt_subtyping(cancer_type)

    args_list = [
        (file_name, cancer_type, output_path, baseline_type, final_prompt)
        for file_name in svs_files
    ]

    results = []
    with multiprocessing.Pool(processes=min(num_workers, len(svs_files))) as pool:
        for idx, (file_name, result) in enumerate(pool.map(run_slide, args_list), start=1):
            if result:
                results.append(result)
            else:
                print(f"Error in subtype prediction: {file_name}")
            print(f"Processed {idx}/{len(svs_files)}: {file_name}")

    # Calculate and save metrics
    print(type(results))
    print(results[:5])
    f1_scores, accuracy, macro_f1 = calculate_f1_scores(results, subtypes)
    final_results = {
        "Num_Samples": len(results),
        "results": results,
        "accuracy": accuracy,
        "f1_scores": f1_scores,
        "macro_f1": macro_f1,
    }
    results_file = os.path.join(output_path, f"{baseline_type}_baseline_results.json")
    with open(results_file, "w") as f:
        json.dump(final_results, f, indent=4)
    print(f"Results saved to {results_file}")

if __name__ == "__main__":
    cancer_type = "BRCA"  # Specify the cancer type
    baseline_type = "gpt"  # Choose "random" or "gpt"
    num_workers = 5
    n = 2000
    main(cancer_type, n=n, baseline_type=baseline_type, num_workers=num_workers)