import os
import json
import openslide
from PIL import Image
from src.subtyping.roi_agent import ROIAgent
from src.subtyping import slide_utils
import config
from openai_client import azure_config_list
from src.subtyping import subtyping_prompt as prompt
from utils.file_utils import initialize_directories, get_svs_files_from_folders

cancer_subtype_map = config.CANCER_SUBTYPE_MAP
cancer_folder_map = config.CANCER_FOLDER_MAP

def process_slide(file_path, cancer_type, output_path, messages):
    file_name = os.path.basename(file_path)
    sample_id = os.path.basename(file_name).split('.')[0]
    correct_label = slide_utils.get_oncotree_code(sample_id)[:12]
    if correct_label not in cancer_subtype_map.get(cancer_type, []):
        print(correct_label)
        return None
    sample_output_dir = os.path.join(output_path, sample_id)
    os.makedirs(sample_output_dir, exist_ok=True)
    if os.path.exists(os.path.join(sample_output_dir, "sample_result.json")):
        print(f"Existing sample: {sample_id}")
        return None

    image = openslide.OpenSlide(file_path)
    roi_agent = ROIAgent(
        image=image,
        cancer_type=cancer_type,
        name="ROI Agent",
        llm_config={"config_list": [azure_config_list[0]], "max_tokens": 3000},
        n_iters=config.NUM_ITER,
        task = "subtyping"
    )

    roi_agent.working_dir = sample_output_dir
    roi_agent.sample_id = sample_id
    analysis_result = roi_agent._reply_user(messages=messages)
    predicted_label = roi_agent.result

    # Identify top 3 ROIs based on AOD
    # top_rois = slide_utils.select_top_rois(sample_output_dir, num_rois=3)
    # top_roi_details = [
    #     {"filename": os.path.basename(roi), "aod": slide_utils.calculate_aod(Image.open(roi))}
    #     for roi in top_rois
    # ]

    is_correct = (predicted_label == correct_label)
    return {
        "file": file_name,
        "sample_id": sample_id,
        "predicted_label": predicted_label,
        "correct_label": correct_label,
        "is_correct": is_correct,
        # "top_rois": top_roi_details
    }
    print("done!")


def calculate_metrics(results, subtypes):
    f1_scores, accuracy, macro_f1 = slide_utils.calculate_f1_scores(results, subtypes)
    return accuracy, f1_scores, macro_f1


def save_results(results, output_path, accuracy, f1_scores, macro_f1):
    final_results = {
        "results": results,
        "accuracy": accuracy,
        "f1_scores": f1_scores,
        "macro_f1": macro_f1
    }
    with open(os.path.join(output_path, "results.json"), "w") as f:
        json.dump(final_results, f, indent=4)

    print("Results saved to results.json.")


def main(cancer_type):
    output_dir = os.path.join(config.QUICK_START_DIR, cancer_type, "roi_output")
    base_path, output_path = initialize_directories(cancer_type, output_path=output_dir)
    subtypes = config.CANCER_SUBTYPE_MAP[cancer_type]
    svs_files = get_svs_files_from_folders(config.CANCER_FOLDER_MAP, cancer_type)

    results = []
    total_files = len(svs_files)
    for idx, file_name in enumerate(svs_files, start=1):
        messages = prompt.get_iteration_messages(cancer_type)
        result = process_slide(file_name, cancer_type, output_path, messages)
        if result:
            results.append(result)
        print(f"Processed {idx}/{total_files}: {file_name}")

    f1_scores, accuracy, macro_f1 = slide_utils.calculate_f1_scores(results, subtypes)
    print(f"Total files processed: {len(results)}")
    print(f"Correct predictions: {sum(result['is_correct'] for result in results)}")
    print(f"Accuracy: {accuracy:.2%}")
    for subtype, f1 in f1_scores.items():
        print(f"F1 Score for {subtype}: {f1:.2f}")
    print(f"Macro-Averaged F1 Score: {macro_f1:.2f}")

    save_results(results, output_path, accuracy, f1_scores, macro_f1)


if __name__ == "__main__":
    cancer_type = "BRCA"  # Modify this to change cancer type
    main(cancer_type)