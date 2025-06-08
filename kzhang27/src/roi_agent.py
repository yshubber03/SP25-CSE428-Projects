import os
import re
import sys
import json
import openai
import random
import numpy as np
import config
import glob
from src.subtyping import subtyping_prompt as prompt

this_file_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.join(this_file_dir, "..", "biomed-0-agent"))

from src.subtyping import slide_utils
import openslide
from autogen import Agent, ConversableAgent, AssistantAgent
from autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent
from PIL import Image, ImageDraw

from skimage import io, color
from skimage.filters import threshold_otsu

from openai_client import azure_config_list, get_openai_response_base64, get_openai_response_base64_with_multiple_images



class ROIAgent(ConversableAgent):
    def __init__(self, image, cancer_type, n_iters=2, mode="single", task="subtyping", to_predict=True, **kwargs):
        super().__init__(**kwargs)
        self.image = image
        self.n_iters = n_iters
        self.working_dir = ""
        self.register_reply([Agent, None], reply_func=ROIAgent._reply_user, position=0)
        self.history_points = []
        self.correct_label = "None"
        self.result = "None"
        self.cancer_type = cancer_type
        self.sample_id = ""
        self.mode = mode
        self.task = task
        self.vqa_msg = None # vqa_question if not None
        self.to_predict = to_predict
        self.final_bbox_info = None
        self.overview_image = None
        self.final_roi = None

        # Add x, y, level attributes with default values
        self.set_roi(0.5, 0.5, 2)
        # self.x = 0.5
        # self.y = 0.5
        # self.level = 2

    def set_roi(self, x, y, level):
        self.x = x
        self.y = y
        self.level = level

    def update_history(self, x, y):
        self.history_points.append((x, y))

    def generate_candidate_rois(self, num_candidates=10):
        candidates = []
        max_level = 0
        while len(candidates) < num_candidates:
            range_min, range_max = 0.1, 0.9
            x = random.uniform(range_min, range_max)
            y = random.uniform(range_min, range_max)
            level = random.randint(0, max_level)
            if 0 <= level <= max_level and slide_utils.is_tissue_region(self.image, level, x, y):
                candidates.append((x, y, level))
                # print(f"Added candidate ROI: (x={x:.2f}, y={y:.2f}, level={level})")
        return candidates

    def get_overview_image(self, image, save_path='overview.png'):
        # use the highest level to get the overview image
        max_level = image.level_count - 1
        x_dim, y_dim = image.level_dimensions[max_level]
        overview = image.read_region((0, 0), max_level, (x_dim, y_dim))
        if x_dim * y_dim > 1024 * 1024:
            overview.thumbnail((1024, 1024))
        overview.save(save_path)
        return save_path

    def _reply_user(self, messages=None, sender=None, config=None):
        os.makedirs(self.working_dir, exist_ok=True)
        if all((messages is None, sender is None)):
            error_msg = f"Either {messages=} or {sender=} must be provided."
            raise AssertionError(error_msg)
        if messages is None:
            messages = self._oai_messages[sender]
        else:
            self.vqa_msg = messages

        user_question = "\n\n".join([msg["content"] for msg in messages])
        # "query <wsi path>", extract wsi_path
        query = user_question.strip()

        commander = AssistantAgent(
            name="Commander",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            system_message="You're a commander to instruct the agent to find the ROI on the whole slide image.",
            is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
        )

        instructor = MultimodalConversableAgent(
            name="Instructor",
            system_message=prompt.get_system_message(),
            llm_config={"config_list": [azure_config_list[0]], "max_tokens": 3000},
            human_input_mode="NEVER",
            max_consecutive_auto_reply=self.n_iters,
        )

        # load the whole slide image
        image = self.image
        available_downsample_levels = {}
        mpp_x_0 = image.properties.get("openslide.mpp-x", None)
        if mpp_x_0 is None:
            raise ValueError("Missing 'openslide.mpp-x' property in slide metadata.")
        for i in range(image.level_count):
            downsample_factor = image.level_downsamples[i]
            mpp = round(float(mpp_x_0) * downsample_factor, 2)
            available_downsample_levels[i] = {
                "downsample_factor": downsample_factor,
                "slide_width": image.level_dimensions[i][0],
                "slide_height": image.level_dimensions[i][1],
                "microns-per-pixel": mpp
            }

        x, y, level = self.x, self.y, self.level
        history_points = []
        overview_img_path = os.path.join(self.working_dir, 'overview.png')
        overview_image, overview_img_path = slide_utils.get_overview_image(image, overview_img_path)
        overview_image_original = overview_image.copy()
        self.overview_image = overview_image.copy()
        roi_img_path = os.path.join(self.working_dir, 'roi_0.png')
        roi_path, action_message, bbox_info = slide_utils.get_image_from_bbox(image, x, y, level, roi_img_path)

        # Record the center
        bbox_center_x = bbox_info["x_0"] + bbox_info["width_level"] // 2
        bbox_center_y = bbox_info["y_0"] + bbox_info["height_level"] // 2
        history_points.append((bbox_center_x, bbox_center_y))

        overview_with_bbox_path = os.path.join(self.working_dir, 'overview_with_bbox_0.png')
        overview_with_bbox_path = slide_utils.draw_bbox_on_overview(overview_image, bbox_info, overview_with_bbox_path, history_points)
        roi_and_overview_img_path = os.path.join(self.working_dir, 'roi_and_overview_0.png')
        slide_utils.concatenate_images(overview_with_bbox_path, roi_img_path, roi_and_overview_img_path)

        final_roi_image_path = ""  # Variable to store the path of the last ROI image
        final_overview_path = ""
        final_bbox_info = None
        final_roi = None

        candidate_rois = self.generate_candidate_rois(num_candidates=20)
        candidate_coords_str = "\n".join(
            [str(f"- Candidate {i+1}: (x={coord[0]:.2f}, y={coord[1]:.2f}, level={coord[2]})")
            for i, coord in enumerate(candidate_rois)]
        )

        for i in range(self.n_iters):
            if i == 0:
                message_content = (
                    f"WSI overview: <img {overview_img_path}>\n"
                    f"Available levels: {available_downsample_levels}\n"
                    f"ROI iteration {i+1}: <img {roi_and_overview_img_path}>\n"
                    f"Please select the best ROI from the following candidates:\n{candidate_coords_str}\n"
                    f"Query: {query}\n"
                    "Choose the most suitable ROI based on the candidate list. Provide the coordinates as: <<x, y, level>>."
                )
                
            else:
                overview_img_path = os.path.join(self.working_dir, 'overview.png')
                overview_image, overview_img_path = slide_utils.get_overview_image(image, overview_img_path)
                roi_img_path = os.path.join(self.working_dir, f'roi_{i}.png')
                roi_path, action_message, bbox_info = slide_utils.get_image_from_bbox(image, x, y, level, roi_img_path)

                bbox_center_x = bbox_info["x_0"] + bbox_info["width_level"] // 2
                bbox_center_y = bbox_info["y_0"] + bbox_info["height_level"] // 2
                history_points.append((bbox_center_x, bbox_center_y))

                overview_with_bbox_path = os.path.join(self.working_dir, f'overview_with_bbox_{i}.png')
                overview_with_bbox_path = slide_utils.draw_bbox_on_overview(overview_image, bbox_info, overview_with_bbox_path, history_points)
                # overview_with_roi_only_path = os.path.join(self.working_dir, f'overview_with_roi_only_{i}.png')
                # overview_clean = overview_image_original.copy()
                # overview_with_roi_only_path = slide_utils.draw_bbox_on_overview_roi_only(overview_clean, bbox_info, overview_with_roi_only_path, "orange")
                roi_and_overview_img_path = os.path.join(self.working_dir, f'roi_and_overview_{i}.png')
                slide_utils.concatenate_images(overview_with_bbox_path, roi_img_path, roi_and_overview_img_path)

                # Update final_roi_image_path each iteration
                final_roi_image_path = roi_img_path
                final_overview_path = roi_and_overview_img_path
                self.final_bbox_info = bbox_info
                

                # For i < 3, include the candidate list for selection
                if i < 3:
                    message_content = (
                        f"ROI iteration {i+1}: <img {roi_and_overview_img_path}>\n"
                        f"Please select the best ROI from the candidates:\n{candidate_coords_str}\n"
                        f"Query: {query}\n"
                        "Provide the coordinates as: <<x, y, level>>."
                    )
                else:
                    # For i >= 3, switch to standard message
                    message_content = (
                        f"ROI iteration {i+1}: <img {roi_and_overview_img_path}>\n"
                        f"ROI coordinates: (x={x}, y={y}, level={level})\n"
                        f"Query: {query}\n"
                        "Think carefully if the current ROI selection is best for answering the user query. Let's try to find a better ROI selection."
                    )
            commander.send(
                message=str(message_content),
                recipient=instructor,
                request_reply=True,
            )

            feedback = commander._oai_messages[instructor][-1]["content"]
            if "TERMINATE".lower() in feedback.lower():
                break
            # parse the feedback to get x, y, level
            matches = re.findall(r"<<x=(.*?), y=(.*?), level=(.*?)>>", feedback)
            if matches:
                new_x, new_y, new_level = matches[-1]
                new_x, new_y, new_level = float(new_x), float(new_y), int(new_level)
                x, y, level = new_x, new_y, new_level
            else:
                print("No new coordinates found in the response; defaulting to last known ROI.")
                break
        
        if self.to_predict:
            num_images_final = 3
            final_prompt = ""
            if self.mode == "single":
                final_prompt = prompt.get_final_prompt(self.cancer_type, self.task, self.vqa_msg)
                response = get_openai_response_base64(final_prompt, final_roi_image_path)
            elif self.mode == "multiple":
                final_prompt = prompt.get_final_prompt_with_multiple_images(self.cancer_type, self.task, self.vqa_msg, num_images_final)
                top_roi_files = slide_utils.select_top_rois(self.working_dir, num_images_final)
                response = get_openai_response_base64_with_multiple_images(final_prompt, top_roi_files)

            if self.task == "subtyping":
                print("Model Response:", response)
                self.result = response
                self.correct_label = slide_utils.get_oncotree_code(self.sample_id)
                print(self.correct_label)
                if response == self.correct_label:
                    print("The model's prediction is correct!")
                else:
                    print("The model's prediction is incorrect!")
            elif self.task == "vqa":
                print("Model Response:", response)
                response = str(response) if response is not None else ""
                print(response)
                self.result = [ans.strip() for ans in response.split(",")]
            
            # Save prediction result
            save_result_path = os.path.join(self.working_dir, "sample_result.json")
            sample_result = {
                "predicted_label": self.result,
                "correct_label": self.correct_label,
                "is_correct": self.result == self.correct_label,
            }
            with open(save_result_path, "w") as f:
                json.dump(sample_result, f, indent=4)
            
            # Delete unnecessary files
            keep_files = []  
            if self.mode == "single":
                keep_files = [final_roi_image_path, final_overview_path]
            elif self.mode == "multiple":
                top_roi_files.append(final_overview_path)
                keep_files = top_roi_files
            for img_file in glob.glob(os.path.join(self.working_dir, '*.png')):
                if img_file not in keep_files:
                    os.remove(img_file)

        # save commander's chat_messages into json file
        chat_messages = commander.chat_messages[instructor]
        save_history_path = os.path.join(self.working_dir, "chat_messages.json")
        with open(save_history_path, "w") as f:
            json.dump(chat_messages, f, indent=4)
        return "Done!"