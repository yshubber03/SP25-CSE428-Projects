import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from segment_anything import SamPredictor, sam_model_registry
from biomedparse import BiomedParseModel  # hypothetical import

# Load SAM model
def load_sam(model_type="vit_h", checkpoint="sam_vit_h.pth"):
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    return SamPredictor(sam)

# Load BiomedParse
def load_biomedparse():
    return BiomedParseModel.from_pretrained("biomedparse_v1")

# Visualize fused mask
def show_mask(mask, title="Mask"):
    plt.imshow(mask, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Hybrid inference: box-guided SAM + prompt-guided BiomedParse
def hybrid_predict(image_path, boxes, prompts):
    image = np.array(Image.open(image_path).convert("RGB"))
    predictor_sam = load_sam()
    predictor_sam.set_image(image)
    predictor_bmp = load_biomedparse()

    fused_masks = []
    for box, prompt in zip(boxes, prompts):
        mask_box, _, _ = predictor_sam.predict(box=np.array(box), multimask_output=False)
        mask_prompt = predictor_bmp.predict(image=image, prompt=prompt)

        
        fused = (mask_box.astype(float) + mask_prompt.astype(float)) / 2
        fused_masks.append(fused)

        print(f"Fused box {box} with prompt: '{prompt}'")
        show_mask(fused, title=f"Fused: {prompt}")

    return fused_masks
