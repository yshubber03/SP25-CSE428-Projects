import openslide
import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import openai
import base64
import re
import cv2
import os
import math
import config

def normalize_points(history_points, slide_width, slide_height):
    return [(x / slide_width, y / slide_height) for x, y in history_points]

def is_distance_too_small(history_points, new_point, slide_width, slide_height, min_distance=0.01):
    normalized_new_point = (new_point[0] / slide_width, new_point[1] / slide_height)
    print(normalize_points)
    too_close_count = 0
    for hx, hy in history_points:
        # Normalize the history point
        normalized_hx = hx / slide_width
        normalized_hy = hy / slide_height
        if abs(normalized_hx - normalized_new_point[0]) < min_distance and abs(normalized_hy - normalized_new_point[1]) < min_distance:
            too_close_count += 1
            if too_close_count >= 2:  # If two points are too close, return True
                return True
    return False

def calculate_f1_scores(results, subtypes):
    confusion_matrix = {subtype: {"tp": 0, "fp": 0, "fn": 0} for subtype in subtypes}
    correct_predictions = sum(result["is_correct"] for result in results)
    for result in results:
        predicted = result["predicted_label"]
        correct = result["correct_label"]
        for subtype in subtypes:
            if correct == subtype:
                if predicted == subtype:
                    confusion_matrix[subtype]["tp"] += 1
                else:
                    confusion_matrix[subtype]["fn"] += 1
            elif predicted == subtype:
                confusion_matrix[subtype]["fp"] += 1
    f1_scores = {
        subtype: calculate_f1(
            confusion_matrix[subtype]["tp"],
            confusion_matrix[subtype]["fp"],
            confusion_matrix[subtype]["fn"],
        )
        for subtype in subtypes
    }
    accuracy = correct_predictions / len(results) if len(results) > 0 else 0
    macro_f1 = sum(f1_scores.values()) / len(subtypes) if subtypes else 0
    return f1_scores, accuracy, macro_f1

def select_top_rois(folder_path, num_rois=3):
    roi_files = [
        f for f in os.listdir(folder_path) 
        if re.match(r"^roi_\d+\.png$", f)  # Matches filenames like "roi_0.png", "roi_1.png"
    ]
    roi_scores = []

    for roi_file in roi_files:
        roi_path = os.path.join(folder_path, roi_file)
        with Image.open(roi_path) as img:
            quality_score = calculate_aod(img)
            roi_scores.append((roi_file, quality_score))

    # Sort the ROIs by quality in descending order and select the top ones
    top_rois = sorted(roi_scores, key=lambda x: x[1], reverse=True)[:num_rois-1]
    if roi_files:
        last_roi_file = max(roi_files, key=lambda x: int(re.search(r"roi_(\d+).png", x).group(1)))
        last_roi_path = os.path.join(folder_path, last_roi_file)
        with Image.open(last_roi_path) as img:
            last_roi_score = calculate_aod(img)
        top_rois.append((last_roi_file, last_roi_score))
    print("Top ROIs with their AOD values:")
    for roi_path, od in top_rois:
        print(f"File: {roi_path}, AOD: {od:.4f}")
    # Return only the file names of the top ROIs
    return [os.path.join(folder_path, roi[0]) for roi in top_rois]

def calculate_aod(image):
    """
    Calculate the Average Optical Density (AOD) for an image.
    """
    image = image.convert('L')
    image_array = np.array(image)
    mean_gray_value = np.mean(image_array)
    od = np.log10(255 / (mean_gray_value + 1))
    return od


def is_tissue_region(image, level, x, y, patch_size=100, aod_threshold=0.05):
    width, height = image.level_dimensions[level]
    abs_x, abs_y = int(x * width), int(y * height)
    if abs_x + patch_size > width or abs_y + patch_size > height:
        return False
    patch = image.read_region((abs_x, abs_y), level, (patch_size, patch_size)).convert("L")
    # Calculate the AOD of the patch
    patch_array = np.array(patch)
    mean_gray_value = np.mean(patch_array)
    od = np.log10(255 / (mean_gray_value + 1))
    return od > aod_threshold

def calculate_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

def get_oncotree_code(sample_id):
    sample_id = sample_id[:12]
    data_path = config.META_DATA_DIR
    df = pd.read_csv(data_path)
    result = df[df['Patient ID'] == sample_id]
    if not result.empty:
        # Oncotree Code / TCGA PanCanAtlas Cancer Type Acronym
        oncotree_code = result.iloc[0]['Oncotree Code']
        return oncotree_code
    else:
        return "None"

def get_overview_image(image, save_path):
    # Use the highest level to get the overview image
    max_level = image.level_count - 1
    x_dim, y_dim = image.level_dimensions[max_level]
    overview = image.read_region((0, 0), max_level, (x_dim, y_dim))
    # If the image is larger than 20MB, resize the image
    if x_dim * y_dim > 1024 * 1024:
        overview.thumbnail((1024, 1024))
    overview = overview.convert("RGB")
    overview.save(save_path)
    if os.path.exists(save_path):
        print(f"{save_path} exists.")
    else:
        print(f"Failed to save {save_path}")
    return overview, save_path

def get_image_from_bbox(image, x, y, level, save_path):
    max_level = image.level_count
    action_message = ""
    abs_width, abs_height = 1024, 1024
    if max_level < level:
        action_message += f"Error: The downsample level {level} is not available. The maximum level is {max_level}.\n"

    x_dim_0, y_dim_0 = image.level_dimensions[0]
    abs_x, abs_y = int(x * x_dim_0), int(y * y_dim_0)
    image.read_region((abs_x, abs_y), level, (abs_width, abs_height)).save(save_path)

    # Extract mpp (magnification per pixel) information for level 0
    mpp_x_level_0 = float(image.properties.get('openslide.mpp-x', '0'))
    mpp_y_level_0 = float(image.properties.get('openslide.mpp-y', '0'))

    # Calculate mpp for the specified level
    if level >= len(image.level_downsamples):
        level = len(image.level_downsamples) - 1
    downsample_factor = image.level_downsamples[level]
    mpp_x = mpp_x_level_0 * downsample_factor
    mpp_y = mpp_y_level_0 * downsample_factor

    bbox_info = {
        "x_0": abs_x,
        "y_0": abs_y,
        "width_level": abs_width,
        "height_level": abs_height,
        "slide_width_0": x_dim_0,
        "slide_height_0": y_dim_0,
        "slide_width_level": image.level_dimensions[level][0],
        "slide_height_level": image.level_dimensions[level][1],
        "mpp_x_level": mpp_x,
        "mpp_y_level": mpp_y,
        "mpp_x_0": mpp_x_level_0,
        "mpp_y_0": mpp_y_level_0,
        "level": level
    }

    return save_path, action_message, bbox_info

def draw_bbox_on_overview_roi_all_tasks(overview_image, bbox_info_list, overview_save_path, color_list):
    draw = ImageDraw.Draw(overview_image)
    font_path = os.path.join(cv2.__path__[0],'qt','fonts','DejaVuSans.ttf')
    font = ImageFont.truetype(font_path, size=20)
    for bbox_info, color in zip(bbox_info_list, color_list):
        x_ratio = bbox_info['x_0'] / bbox_info['slide_width_0']
        y_ratio = bbox_info['y_0'] / bbox_info['slide_height_0']
        overview_width, overview_height = overview_image.size
        bbox_x = int(x_ratio * overview_width)
        bbox_y = int(y_ratio * overview_height)
        bbox_width = int(bbox_info['width_level'] * overview_width / bbox_info['slide_width_level'])
        bbox_height = int(bbox_info['height_level'] * overview_height / bbox_info['slide_height_level'])
        # Draw the bounding box
        draw.rectangle([bbox_x, bbox_y, bbox_x + bbox_width, bbox_y + bbox_height], outline=color, width=5)
    # Add text annotations
    draw.text((5, 5), "x=0.0, y=0.0", fill="black", font=font)
    draw.text((5, overview_height - 25), "x=0.0, y=1.0", fill="black", font=font)
    draw.text((overview_width - 140, 5), "x=1.0, y=0.0", fill="black", font=font)
    draw.text((overview_width - 140, overview_height - 25), "x=1.0, y=1.0", fill="black", font=font)
    bbox_center_x = bbox_x
    bbox_center_y = bbox_y - 30
    # draw.text((bbox_center_x, bbox_center_y), f"x={x_ratio:.2f}, y={y_ratio:.2f}", fill="black", font=font)
    # Save the image with the bounding box and annotations
    overview_image.save(overview_save_path)
    return overview_save_path

def draw_bbox_on_overview_roi_only(overview_image, bbox_info, overview_save_path, color):
    draw = ImageDraw.Draw(overview_image)
    font_path = os.path.join(cv2.__path__[0],'qt','fonts','DejaVuSans.ttf')
    font = ImageFont.truetype(font_path, size=20)
    x_ratio = bbox_info['x_0'] / bbox_info['slide_width_0']
    y_ratio = bbox_info['y_0'] / bbox_info['slide_height_0']
    overview_width, overview_height = overview_image.size
    bbox_x = int(x_ratio * overview_width)
    bbox_y = int(y_ratio * overview_height)
    bbox_width = int(bbox_info['width_level'] * overview_width / bbox_info['slide_width_level'])
    bbox_height = int(bbox_info['height_level'] * overview_height / bbox_info['slide_height_level'])
    # Draw the bounding box
    draw.rectangle([bbox_x, bbox_y, bbox_x + bbox_width, bbox_y + bbox_height], outline=color, width=5)
    # Add text annotations
    draw.text((5, 5), "x=0.0, y=0.0", fill="black", font=font)
    draw.text((5, overview_height - 25), "x=0.0, y=1.0", fill="black", font=font)
    draw.text((overview_width - 140, 5), "x=1.0, y=0.0", fill="black", font=font)
    draw.text((overview_width - 140, overview_height - 25), "x=1.0, y=1.0", fill="black", font=font)
    bbox_center_x = bbox_x
    bbox_center_y = bbox_y - 30
    # draw.text((bbox_center_x, bbox_center_y), f"x={x_ratio:.2f}, y={y_ratio:.2f}", fill="black", font=font)
    # Save the image with the bounding box and annotations
    overview_image.save(overview_save_path)
    return overview_save_path

def draw_bbox_on_overview(overview_image, bbox_info, overview_save_path, history_points):
    draw = ImageDraw.Draw(overview_image)
    #font = ImageFont.load_default()
    font_path = os.path.join(cv2.__path__[0],'qt','fonts','DejaVuSans.ttf')
    font = ImageFont.truetype(font_path, size=20)

    # Scale the bounding box coordinates to the overview image
    x_ratio = bbox_info['x_0'] / bbox_info['slide_width_0']
    y_ratio = bbox_info['y_0'] / bbox_info['slide_height_0']
    overview_width, overview_height = overview_image.size
    bbox_x = int(x_ratio * overview_width)
    bbox_y = int(y_ratio * overview_height)
    bbox_width = int(bbox_info['width_level'] * overview_width / bbox_info['slide_width_level'])
    bbox_height = int(bbox_info['height_level'] * overview_height / bbox_info['slide_height_level'])

    # History points
    scaled_history_points = [
        (int(x * overview_image.size[0] / bbox_info['slide_width_0']),
        int(y * overview_image.size[1] / bbox_info['slide_height_0']))
        for x, y in history_points
    ]
    for i in range(1, len(scaled_history_points)):
        start = scaled_history_points[i - 1]
        end = scaled_history_points[i]
        draw.line([start, end], fill="red", width=3)  # Red lines for path
    for i, point in enumerate(scaled_history_points):
        cx, cy = point
        radius = 5
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill="blue", outline="blue")
        draw.text((cx + 10, cy), f"{i+1}", fill="black", font=font)

    left_margin_x = 20
    top_margin_y = 50
    line_spacing = 20
    for i, (original_x, original_y) in enumerate(history_points):
        x_coord = round(original_x / bbox_info['slide_width_0'], 2)
        y_coord = round(original_y / bbox_info['slide_height_0'], 2)
        draw.text((left_margin_x, top_margin_y + i * line_spacing), f"{i+1}: x={x_coord}, y={y_coord}", fill="black", font=font)

    # Draw the bounding box
    draw.rectangle([bbox_x, bbox_y, bbox_x + bbox_width, bbox_y + bbox_height], outline="green", width=5)
    # Add text annotations
    draw.text((5, 5), "x=0.0, y=0.0", fill="black", font=font)
    draw.text((5, overview_height - 25), "x=0.0, y=1.0", fill="black", font=font)
    draw.text((overview_width - 140, 5), "x=1.0, y=0.0", fill="black", font=font)
    draw.text((overview_width - 140, overview_height - 25), "x=1.0, y=1.0", fill="black", font=font)
    bbox_center_x = bbox_x
    bbox_center_y = bbox_y - 30
    draw.text((bbox_center_x, bbox_center_y), f"x={x_ratio:.2f}, y={y_ratio:.2f}", fill="black", font=font)

    # Save the image with the bounding box and annotations
    overview_image.save(overview_save_path)
    return overview_save_path

def concatenate_images(image1_path, image2_path, output_path):
    # Open the images
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)
    # Get the dimensions of the images
    image1_width, image1_height = image1.size
    image2_width, image2_height = image2.size
    # Create a new image with the combined width and max height
    new_width = image1_width + image2_width
    new_height = max(image1_height, image2_height)
    new_image = Image.new('RGB', (new_width, new_height))
    # Paste the images into the new image
    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (image1_width, 0))
    # Save the concatenated image
    new_image.save(output_path)