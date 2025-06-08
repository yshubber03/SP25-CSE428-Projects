def get_system_message():
    message = """Find a 1024px x 1024px region of interest (ROI) on the whole slide image (WSI) based on the user's query. Select the most relevant ROI by defining its bounding box and downsample level. The bounding box is determined by its top-left corner (x, y) relative to the top-left corner of the WSI. For example, x=0.5, y=0.5 represents the center of the WSI.

Adjust the downsample level to zoom in or out:

* Zoom in for more detail with a lower level (e.g., level=0 for highest magnification).
* Zoom out for a larger area with a higher level (e.g., level=1 or above).

The maximum downsample level of the WSI will be provided. An overview of the WSI and the current ROI, highlighted by a bounding box, will also be shown. Assess if the current ROI meets the user’s needs. If it does, respond with “TERMINATE.” If not, suggest a new ROI in the format: <<x, y, level>>.

To check different areas, adjust the coordinates. For example:

* To check the left area from the current location (x=0.5, y=0.5), use (x=0.4, y=0.5).
* To check the lower area, use (x=0.5, y=0.6).

Ensure to check multiple areas in the slide to find the best region of interest.

In each response, provide a brief medical reasoning (one sentence) explaining why the selected region is appropriate. Describe any notable cellular or structural features that support your decision.

End your response using the following two-line format:
<reasoning sentence>
<<x=..., y=..., level=...>>

For example:
"This region displays dense nuclear atypia and irregular gland formation, consistent with carcinoma."
<<x=0.43, y=0.62, level=0>>

Make sure the coordinate format exactly matches **<<x=..., y=..., level=...>>** for automatic parsing.
"""
    return message

def generate_prompt_for_coordinates(cancer_type, candidate_coords):
    """
    Generate a text-based prompt for GPT to select the best ROI coordinate.
    """

    # Construct the textual prompt
    prompt = (
        f"This is a histopathological slide of {cancer_type} tissue. "
        f"You are provided with 20 candidate points in the format [(x, y), ...]:\n"
        f"{candidate_coords}\n\n"
        f"Please select the best point for determining the cancer subtype. "
        f"Respond with the coordinates in the format: `x=..., y=...`. Here are rules for the coordinates." + get_system_message()
    )

    return prompt
  

def get_iteration_messages(cancer_type):
    if cancer_type == "BRCA":
        messages = [{"content": "What is the cancer subtype of this slide? Is it Invasive Ductal Carcinoma (IDC) or Invasive Lobular Carcinoma (ILC)?"}]
    elif cancer_type == "LUNG":
        messages = [{"content": "What is the cancer subtype of this slide? Is it Lung Adenocarcinoma (LUAD) or Lung Squamous Cell Carcinoma (LUSC)?"}]
    elif cancer_type == "COLON":
        messages = [{"content": "What is the cancer subtype of this slide? Is it Colon Adenocarcinoma (COAD) or Rectal Adenocarcinoma (READ)?"}]
    elif cancer_type == "RCC":
        messages = [{"content": "What is the cancer subtype of this slide? Is it Clear Cell Renal Cell Carcinoma (CCRCC), or Papillary Renal Cell Carcinoma (PRCC), or Chromophobe Renal Cell Carcinoma (CHRCC)?"}]
    elif cancer_type == "ESO":
        messages = [{"content": "What is the cancer subtype of this slide? Is it Esophageal Adenocarcinoma (ESCA), Esophageal Squamous Cell Carcinoma (ESCC), or Stomach Adenocarcinoma (STAD)?"}]
    elif cancer_type == "HEP":
        messages = [{"content": "What is the cancer subtype of this slide? Is it Cholangiocarcinoma (CHOL) or Hepatocellular Carcinoma (HCC)?"}]
    elif cancer_type == "GLIOMA":
        messages = [{"content": "What is the cancer subtype of this slide? Is it Glioblastoma (GBM) or Oligodendroglioma (ODG)?"}]
    elif cancer_type == "ADREN":
        messages = [{"content": "What is the cancer subtype of this slide? Is it Adrenocortical Carcinoma (ACC) or Pheochromocytoma (PHC)?"}]
    elif cancer_type == "CERVIX":
        messages = [{"content": "What is the cancer subtype of this slide? Is it Cervical Squamous Cell Carcinoma (CESC) or Endocervical Adenocarcinoma (ECAD)?"}]
    elif cancer_type == "PLEURA":
        messages = [{"content": "What is the cancer subtype of this slide? Is it Pleural Mesothelioma, Biphasic Type (PLBMESO), Pleural Mesothelioma, Epithelioid Type (PLEMESO), or Pleural Mesothelioma, Sarcomatoid Type (PLSMESO)?"}]
    elif cancer_type == "SOFT":
        messages = [{"content": "What is the cancer subtype of this slide? Is it Dedifferentiated Liposarcoma (DDLS), Leiomyosarcoma (LMS), Myxofibrosarcoma (MFS), or Undifferentiated Pleomorphic Sarcoma/Malignant Fibrous Histiocytoma/High-Grade Spindle Cell Sarcoma (MFH)?"}]
    elif cancer_type == "TESTIS":
        messages = [{"content": "What is the cancer subtype of this slide? Is it Seminoma (SEM) or Mixed Germ Cell Tumor (MGCT)?"}]
    elif cancer_type == "UTERUS":
        messages = [{"content": "What is the cancer subtype of this slide? Is it Uterine Endometrioid Carcinoma (UEC) or Uterine Serous Carcinoma/Uterine Papillary Serous Carcinoma (USC)?"}]
    else:
        raise ValueError(f"Unknown cancer type: {cancer_type}")
    return messages

def get_final_prompt(cancer_type, task, vqa_msg):
    if task == "subtyping":
        return get_final_prompt_subtyping(cancer_type)
    elif task == "vqa":
        return get_final_prompt_vqa(cancer_type, vqa_msg)
    else:
        raise ValueError(f"Unknown task type: {task}")

def get_final_prompt_subtyping(cancer_type):
    final_prompt = ""
    if cancer_type == "BRCA":
        final_prompt = (
            "This image shows a histological slide from a tissue biopsy with a selected region of interest (ROI) highlighting key cellular structures. "
            "Based on the observed morphological features, determine if the slide is Invasive Ductal Carcinoma (IDC) or Invasive Lobular Carcinoma (ILC). "
            "Use the following criteria to assist in classification:\n\n"
            "- **IDC**: Typically presents with glandular or duct-like structures, clusters of cells forming ducts, and may exhibit central necrosis within ducts. Cells often show significant nuclear pleomorphism, prominent nucleoli, and frequent mitotic figures. Tumor cells are arranged in cohesive nests or sheets.\n"
            "- **ILC**: Generally displays single-file patterns of cells infiltrating the stroma, often in linear or targetoid arrangements around ducts. Lacks duct formation, with cells appearing in strands or individually. E-cadherin expression is typically reduced or absent. Cells have small, uniform nuclei with inconspicuous nucleoli. The stroma reaction is usually minimal, and tumor borders are often ill-defined.\n\n"
            "Considering these characteristics, **provide only the classification word** — either 'IDC' or 'ILC' — without any additional text or explanation. You have to make a decision even though you are unsure."
        )
    elif cancer_type == "LUNG":
        final_prompt = (
            "This image shows a histological slide from a tissue biopsy with a selected region of interest (ROI) highlighting key cellular structures. "
            "Based on the observed morphological features, determine if the slide is Lung Adenocarcinoma (LUAD) or Lung Squamous Cell Carcinoma (LUSC). "
            "Use the following criteria to assist in classification:\n\n"
            "- **LUAD**: Typically shows glandular differentiation with acinar, papillary, or lepidic growth patterns. Cells may contain intracellular mucin, and mucin production is often evident. Nuclei are round with prominent nucleoli. Commonly arises in the peripheral regions of the lung and may be associated with scar tissue. Tumor cells are arranged in loose clusters forming glandular or papillary structures.\n"
            "- **LUSC**: Characterized by squamous differentiation features, including keratinization, intercellular bridges, and squamous pearl formation. Cells are polygonal with abundant eosinophilic cytoplasm. Nuclear pleomorphism and hyperchromasia are common. Typically originates in the central parts of the lung, often associated with bronchial structures. Tumor cells are arranged in cohesive nests or sheets.\n\n"
            "Considering these characteristics, **provide only the classification word** — either 'LUAD' or 'LUSC' — without any additional text or explanation. You have to make a decision even though you are unsure."
        )
    elif cancer_type == "COLON":
        final_prompt = (
            "This image shows a histological slide from a tissue biopsy with a selected region of interest (ROI) highlighting key cellular structures. "
            "Based on the observed morphological features, determine if the slide is Colon Adenocarcinoma (COAD) or Rectal Adenocarcinoma (READ). "
            "Use the following criteria to assist in classification:\n\n"
            "- **COAD**: Arises in the colon and may exhibit glandular or tubular structures with varying degrees of differentiation. Tumor cells often form irregular glands and may produce mucin. Invasion into the muscularis propria and pericolonic fat is common. Lymphovascular invasion may be present. Tumor cells are arranged in loose clusters forming glandular or tubular structures.\n"
            "- **READ**: Originates in the rectum and displays histological features similar to COAD, including gland formation and mucin production. However, READ tends to have circumferential involvement and may more frequently exhibit perineural invasion. Tumor deposits in the mesorectal fat are more commonly observed. Tumor cells are arranged in cohesive clusters forming glandular or tubular structures.\n\n"
            "Considering these characteristics, **provide only the classification word** — either 'COAD' or 'READ' — without any additional text or explanation. You have to make a decision even though you are unsure."
        )
    elif cancer_type == "RCC":
        final_prompt = (
            "This image shows a histological slide from a tissue biopsy with a selected region of interest (ROI) highlighting key cellular structures. "
            "Based on the observed morphological features, determine if the slide is Clear Cell Renal Cell Carcinoma (CCRCC), or Papillary Renal Cell Carcinoma (PRCC), or Chromophobe Renal Cell Carcinoma (CHRCC). "
            "Use the following criteria to assist in classification:\n\n"
            "- **CCRCC (Clear Cell Renal Cell Carcinoma)**: Characterized by clear cytoplasm due to glycogen and lipid accumulation. Cells are arranged in nests or alveolar structures surrounded by a delicate vascular network. The tumor often exhibits a golden-yellow appearance grossly due to lipid content. Nuclei show varying degrees of atypia, with prominent nucleoli in higher grades. Hemorrhage, necrosis, and cystic degeneration are common histological findings.\n"
            "- **PRCC (Papillary Renal Cell Carcinoma)**: Displays papillary or tubular-papillary structures with fibrovascular cores. The tumor cells are cuboidal to columnar and often contain foamy macrophages within the fibrovascular cores. Basophilic, eosinophilic, or oncocytic cytoplasm may be present. Psammoma bodies are frequently observed. The stroma may show edema, and hemorrhage is not uncommon. Multinucleated giant cells may occasionally be identified.\n"
            "- **CHRCC (Chromophobe Renal Cell Carcinoma)**: Composed of large polygonal cells with pale, eosinophilic cytoplasm and distinct cell borders. Tumor cells often demonstrate a perinuclear halo and finely reticulated cytoplasm resembling plant cells. The nuclei are irregular, with perinuclear clearing. The tumor architecture is predominantly solid, with sheets or trabeculae, and may include small, focally arranged tubules. Cytoplasmic granularity is another distinguishing feature.\n\n"
            "Considering these characteristics, **provide only the classification word** — 'CCRCC' or 'PRCC' or 'CHRCC' — without any additional text or explanation. You have to make a decision even though you are unsure."
        )
    elif cancer_type == "ESO":
        final_prompt = (
            "This image shows a histological slide from a tissue biopsy with a selected region of interest (ROI) highlighting key cellular structures. "
            "Based on the observed morphological features, determine if the slide is Esophageal Adenocarcinoma (ESCA), Esophageal Squamous Cell Carcinoma (ESCC), "
            "or Stomach Adenocarcinoma (STAD). "
            "Use the following criteria to assist in classification:\n\n"
            "- **ESCA (Esophageal Adenocarcinoma)**: Typically arises in the lower esophagus and is associated with Barrett's esophagus. Tumors show glandular differentiation, often forming tubular or papillary structures. Cells may exhibit mucin production and have prominent nucleoli.\n"
            "- **ESCC (Esophageal Squamous Cell Carcinoma)**: Occurs in the middle or upper esophagus. Displays squamous differentiation with features like keratin pearls, intercellular bridges, and polygonal cells. Nuclei are hyperchromatic with irregular contours.\n"
            "- **STAD (Stomach Adenocarcinoma)**: Frequently exhibits glandular or tubular structures, with varying degrees of differentiation. Tumors may show mucin production, nuclear pleomorphism, and invasion into the gastric wall. Tumor cells are typically arranged in irregular glands or sheets.\n\n"
            "Considering these characteristics, **provide only the classification word** — 'ESCA', 'ESCC', or 'STAD' — without any additional text or explanation. You have to make a decision even though you are unsure."
        )
    elif cancer_type == "HEP":
        final_prompt = (
            "This image shows a histological slide from a tissue biopsy with a selected region of interest (ROI) highlighting key cellular structures. "
            "Based on the observed morphological features, determine if the slide is Cholangiocarcinoma (CHOL) or Hepatocellular Carcinoma (HCC). "
            "Use the following criteria to assist in classification:\n\n"
            "- **CHOL (Cholangiocarcinoma)**: Arises from the bile ducts and typically forms glandular or tubular structures with dense fibrous stroma. Tumor cells are cuboidal to columnar with prominent nucleoli and may show mucin production. Desmoplastic stromal reaction is a hallmark feature.\n"
            "- **HCC (Hepatocellular Carcinoma)**: Originates in hepatocytes, with tumor cells arranged in trabecular, pseudoacinar, or solid patterns. Cells often have abundant eosinophilic cytoplasm, round nuclei, and prominent nucleoli. Intracytoplasmic bile or Mallory bodies may be present. Endothelial-lined vascular spaces are frequently observed.\n\n"
            "Considering these characteristics, **provide only the classification word** — 'CHOL' or 'HCC' — without any additional text or explanation. You have to make a decision even though you are unsure."
        )
    elif cancer_type == "GLIOMA":
        final_prompt = (
            "This image shows a histological slide from a tissue biopsy with a selected region of interest (ROI) highlighting key cellular structures. "
            "Based on the observed morphological features, determine if the slide is Glioblastoma (GBM) or Oligodendroglioma (ODG). "
            "Use the following criteria to assist in classification:\n\n"
            "- **GBM (Glioblastoma)**: High-grade astrocytic tumor characterized by marked cellular atypia, high mitotic activity, necrosis, and microvascular proliferation. Necrotic areas are often surrounded by hypercellular regions (pseudopalisading necrosis). Cells may appear pleomorphic with hyperchromatic nuclei.\n"
            "- **ODG (Oligodendroglioma)**: Low- to intermediate-grade glioma with tumor cells forming a 'fried-egg' appearance due to clear cytoplasmic halos. Tumors often contain delicate branching capillaries resembling a 'chicken-wire' pattern. Calcifications and uniform, round nuclei are commonly observed.\n\n"
            "Considering these characteristics, **provide only the classification word** — 'GBM' or 'ODG' — without any additional text or explanation. You have to make a decision even though you are unsure."
        )
    elif cancer_type == "ADREN":
        final_prompt = (
            "This image shows a histological slide from a tissue biopsy with a selected region of interest (ROI) highlighting key cellular structures. "
            "Based on the observed morphological features, determine if the slide is Adrenocortical Carcinoma (ACC) or Pheochromocytoma (PHC). "
            "Use the following criteria to assist in classification:\n\n"
            "- **ACC (Adrenocortical Carcinoma)**: Malignant tumor with high nuclear pleomorphism, atypical mitoses, necrosis, and increased mitotic activity. Cells may have eosinophilic or clear cytoplasm, arranged in sheets or nests.\n"
            "- **PHC (Pheochromocytoma)**: Tumor arising from adrenal medulla chromaffin cells, characterized by polygonal cells with abundant granular cytoplasm. A classic 'zellballen' pattern with highly vascularized stroma is common, and nuclei have a salt-and-pepper chromatin pattern.\n\n"
            "Considering these characteristics, **provide only the classification word** — 'ACC' or 'PHC' — without any additional text or explanation. You have to make a decision even though you are unsure."
        )

    elif cancer_type == "CERVIX":
        final_prompt = (
            "This image shows a histological slide from a cervical biopsy with a selected region of interest (ROI) highlighting key cellular structures. "
            "Based on the observed morphological features, determine if the slide is Cervical Squamous Cell Carcinoma (CESC) or Endocervical Adenocarcinoma (ECAD). "
            "Use the following criteria to assist in classification:\n\n"
            "- **CESC (Cervical Squamous Cell Carcinoma)**: Composed of malignant squamous cells with intercellular bridges, keratinization, and pleomorphic nuclei. Tumors may present as well, moderately, or poorly differentiated.\n"
            "- **ECAD (Endocervical Adenocarcinoma)**: Malignant glandular cells forming irregular glands with mucin production. Tumor cells exhibit nuclear atypia, mitotic figures, and may invade cervical stroma.\n\n"
            "Considering these characteristics, **provide only the classification word** — 'CESC' or 'ECAD' — without any additional text or explanation. You have to make a decision even though you are unsure."
        )

    elif cancer_type == "PLEURA":
        final_prompt = (
            "This image shows a histological slide from a pleural biopsy with a selected region of interest (ROI) highlighting key cellular structures. "
            "Based on the observed morphological features, determine if the slide is Pleural Mesothelioma, Biphasic Type (PLBMESO), Pleural Mesothelioma, Epithelioid Type (PLEMESO), or Pleural Mesothelioma, Sarcomatoid Type (PLSMESO). "
            "Use the following criteria to assist in classification:\n\n"
            "- **PLBMESO (Biphasic Mesothelioma)**: Contains both epithelioid and sarcomatoid components, with at least 10% of each. Epithelioid regions appear glandular, while sarcomatoid areas consist of spindle cells.\n"
            "- **PLEMESO (Epithelioid Mesothelioma)**: Composed of uniform, polygonal tumor cells forming tubulopapillary or trabecular patterns. Cells have round nuclei, prominent nucleoli, and eosinophilic cytoplasm.\n"
            "- **PLSMESO (Sarcomatoid Mesothelioma)**: Highly aggressive subtype with elongated spindle cells arranged in fascicles. Lacks glandular differentiation and often resembles fibrosarcoma.\n\n"
            "Considering these characteristics, **provide only the classification word** — 'PLBMESO', 'PLEMESO', or 'PLSMESO' — without any additional text or explanation. You have to make a decision even though you are unsure."
        )

    elif cancer_type == "SOFT":
        final_prompt = (
            "This image shows a histological slide from a soft tissue biopsy with a selected region of interest (ROI) highlighting key cellular structures. "
            "Based on the observed morphological features, determine if the slide is Dedifferentiated Liposarcoma (DDLS), Leiomyosarcoma (LMS), Myxofibrosarcoma (MFS), or Undifferentiated Pleomorphic Sarcoma (MFH). "
            "Use the following criteria to assist in classification:\n\n"
            "- **DDLS (Dedifferentiated Liposarcoma)**: Contains well-differentiated liposarcoma adjacent to high-grade non-lipogenic sarcoma. Dedifferentiated areas show pleomorphic spindle cells and high mitotic activity.\n"
            "- **LMS (Leiomyosarcoma)**: Malignant smooth muscle tumor with elongated spindle cells, cigar-shaped nuclei, and eosinophilic cytoplasm. Tumors show nuclear pleomorphism and frequent mitoses.\n"
            "- **MFS (Myxofibrosarcoma)**: Characterized by myxoid stroma with curvilinear vasculature and scattered pleomorphic tumor cells. Higher-grade tumors show increased cellular atypia.\n"
            "- **MFH (Undifferentiated Pleomorphic Sarcoma)**: Poorly differentiated sarcoma with pleomorphic cells arranged in storiform or haphazard patterns, high mitotic activity, and tumor giant cells.\n\n"
            "Considering these characteristics, **provide only the classification word** — 'DDLS', 'LMS', 'MFS', or 'MFH' — without any additional text or explanation. You have to make a decision even though you are unsure."
        )

    elif cancer_type == "TESTIS":
        final_prompt = (
            "This image shows a histological slide from a testicular biopsy with a selected region of interest (ROI) highlighting key cellular structures. "
            "Based on the observed morphological features, determine if the slide is Seminoma (SEM) or Mixed Germ Cell Tumor (MGCT). "
            "Use the following criteria to assist in classification:\n\n"
            "- **SEM (Seminoma)**: Composed of uniform polygonal tumor cells with clear cytoplasm and centrally located nuclei. Tumors have fibrous septa infiltrated by lymphocytes and may contain granulomas.\n"
            "- **MGCT (Mixed Germ Cell Tumor)**: Contains multiple germ cell tumor components, such as embryonal carcinoma, yolk sac tumor, choriocarcinoma, and teratoma. Each component has distinct histological features, including glandular, solid, or papillary growth patterns.\n\n"
            "Considering these characteristics, **provide only the classification word** — 'SEM' or 'MGCT' — without any additional text or explanation. You have to make a decision even though you are unsure."
        )

    elif cancer_type == "UTERUS":
        final_prompt = (
            "This image shows a histological slide from a uterine biopsy with a selected region of interest (ROI) highlighting key cellular structures. "
            "Based on the observed morphological features, determine if the slide is Uterine Endometrioid Carcinoma (UEC) or Uterine Serous Carcinoma (USC). "
            "Use the following criteria to assist in classification:\n\n"
            "- **UEC (Uterine Endometrioid Carcinoma)**: Common endometrial malignancy with glandular structures resembling normal endometrium. Tumor cells have round to oval nuclei with variable atypia and occasional squamous differentiation.\n"
            "- **USC (Uterine Serous Carcinoma)**: Aggressive high-grade carcinoma with papillary or glandular architecture. Tumor cells exhibit marked pleomorphism, high mitotic index, and prominent nucleoli.\n\n"
            "Considering these characteristics, **provide only the classification word** — 'UEC' or 'USC' — without any additional text or explanation. You have to make a decision even though you are unsure."
        )
    else:
        raise ValueError(f"Unknown cancer type: {cancer_type}")

    return final_prompt

def get_final_prompt_vqa(cancer_type, vqa_msg):
    if not vqa_msg:
        raise ValueError("VQA message list is empty.")

    if cancer_type == "BRCA":
        # Format the prompt with multiple questions
        questions_block = "\n\n".join(
            [msg["content"] for msg in vqa_msg]
        )

        final_prompt = (
            "You are a medical AI assistant trained to analyze pathology slides and answer multiple-choice "
            "questions related to cancer diagnosis. For each question, select the most appropriate answer "
            "from the given choices. If you are uncertain, select the answer that is the closest match based "
            "on available information. Your response must strictly follow the order of the questions, and answers "
            "should be separated by a comma. Do not provide any explanations or additional information.\n\n"
            f"{questions_block}\n\n"
            "Answers:"
        )
    else:
        raise ValueError(f"Unknown cancer type: {cancer_type}")

    return final_prompt

def get_final_prompt_with_multiple_images(cancer_type, task, vqa_msg, num_images):
    if task == "subtyping":
        return get_final_prompt_with_multiple_images_subtyping(cancer_type, num_images)
    elif task == "vqa":
        return get_final_prompt_with_multiple_images_vqa(cancer_type, vqa_msg, num_images)
    else:
        raise ValueError(f"Unknown task type: {task}")

def get_final_prompt_with_multiple_images_subtyping(cancer_type, num_images):
    final_prompt = ""
    if cancer_type == "BRCA":
        final_prompt = (
            f"You are provided {num_images} histological slides from a tissue biopsy. Each slide highlights a specific region of interest (ROI) containing key cellular structures. "
            "Your task is to classify the slides into one of two subtypes: Invasive Ductal Carcinoma (IDC) or Invasive Lobular Carcinoma (ILC). "
            "Use the detailed histological features below to guide your decision-making:\n\n"
            "- **IDC**: Typically presents with glandular or duct-like structures, often forming cohesive nests or sheets of cells. Central necrosis may be evident in ducts. Cells exhibit nuclear pleomorphism, prominent nucleoli, and frequent mitotic figures. Tumor cells are typically arranged in dense clusters, forming clear ductal structures.\n"
            "- **ILC**: Displays single-file infiltration patterns of tumor cells into the stroma. Cells may be arranged in targetoid patterns around ducts and lack duct formation. Reduced or absent E-cadherin expression is a key feature. Tumor cells are generally small with uniform nuclei and inconspicuous nucleoli, showing a diffuse and ill-defined growth pattern.\n\n"
            "Analyze all {num_images} slides provided and determine the most likely subtype based on the consistency of features observed across the regions. Considering these characteristics, **provide only the classification word** — either 'IDC' or 'ILC' — without any additional text or explanation. You have to make a decision even though you are unsure."
        )
    elif cancer_type == "LUNG":
        final_prompt = (
            f"You are provided {num_images} histological slides from a tissue biopsy. Each slide highlights a specific region of interest (ROI) containing key cellular structures. "
            "Your task is to classify the slides into one of two subtypes: Lung Adenocarcinoma (LUAD) or Lung Squamous Cell Carcinoma (LUSC). "
            "Use the detailed histological features below to guide your decision-making:\n\n"
            "- **LUAD**: Typically shows glandular differentiation with acinar, papillary, or lepidic growth patterns. Cells may contain intracellular mucin, and mucin production is often evident. Tumor cells have round nuclei with prominent nucleoli. Commonly arises in the peripheral regions of the lung and may be associated with scar tissue. Tumor cells are arranged in loose clusters forming glandular or papillary structures.\n"
            "- **LUSC**: Exhibits squamous differentiation, including keratin pearls, intercellular bridges, and polygonal cells with abundant eosinophilic cytoplasm. Nuclear pleomorphism and hyperchromasia are common. Tumor cells are arranged in cohesive nests and sheets, often originating in central lung regions and associated with bronchial structures.\n\n"
            "Analyze all {num_images} slides provided and determine the most likely subtype based on the consistency of features observed across the regions. Considering these characteristics, **provide only the classification word** — either 'LUAD' or 'LUSC' — without any additional text or explanation. You have to make a decision even though you are unsure."
        )
    elif cancer_type == "COLON":
        final_prompt = (
            f"You are provided {num_images} histological slides from a tissue biopsy. Each slide highlights a specific region of interest (ROI) containing key cellular structures. "
            "Your task is to classify the slides into one of two subtypes: Colon Adenocarcinoma (COAD) or Rectal Adenocarcinoma (READ). "
            "Use the detailed histological features below to guide your decision-making:\n\n"
            "- **COAD**: Frequently arises in the proximal or distal colon. Tumors exhibit glandular or tubular structures with varying degrees of differentiation. Mucin pools and irregular gland formation are common. Cells exhibit moderate-to-high nuclear atypia, and invasion into the muscularis propria and pericolonic fat is often observed. Tumor cells form loose clusters within the glandular or tubular architecture.\n"
            "- **READ**: Typically originates in the rectum and shares features with COAD, including irregular glands and mucin production. However, READ tends to have circumferential involvement and may more frequently exhibit perineural invasion. Tumor deposits in the mesorectal fat are more commonly observed. Tumor cells are arranged in cohesive clusters forming glandular or tubular structures.\n\n"
            "Analyze all {num_images} slides provided and determine the most likely subtype based on the consistency of features observed across the regions. Considering these characteristics, **provide only the classification word** — either 'COAD' or 'READ' — without any additional text or explanation. You have to make a decision even though you are unsure."
        )
    elif cancer_type == "RCC":
        final_prompt = (
            f"You are provided {num_images} histological slides from a tissue biopsy. Each slide highlights a specific region of interest (ROI) containing key cellular structures. "
            "Your task is to classify the slides into one of three subtypes: Clear Cell Renal Cell Carcinoma (CCRCC), or Papillary Renal Cell Carcinoma (PRCC), or Chromophobe Renal Cell Carcinoma (CHRCC). "
            "Use the detailed histological features below to guide your decision-making:\n\n"
            "- **CCRCC (Clear Cell Renal Cell Carcinoma)**: Characterized by clear cytoplasm due to glycogen and lipid accumulation. Cells are arranged in nests or alveolar structures surrounded by a delicate vascular network. The tumor often exhibits a golden-yellow appearance grossly due to lipid content. Nuclei show varying degrees of atypia, with prominent nucleoli in higher grades. Hemorrhage, necrosis, and cystic degeneration are common histological findings.\n"
            "- **PRCC (Papillary Renal Cell Carcinoma)**: Displays papillary or tubular-papillary structures with fibrovascular cores. The tumor cells are cuboidal to columnar and often contain foamy macrophages within the fibrovascular cores. Basophilic, eosinophilic, or oncocytic cytoplasm may be present. Psammoma bodies are frequently observed. The stroma may show edema, and hemorrhage is not uncommon. Multinucleated giant cells may occasionally be identified.\n"
            "- **CHRCC (Chromophobe Renal Cell Carcinoma)**: Composed of large polygonal cells with pale, eosinophilic cytoplasm and distinct cell borders. Tumor cells often demonstrate a perinuclear halo and finely reticulated cytoplasm resembling plant cells. The nuclei are irregular, with perinuclear clearing. The tumor architecture is predominantly solid, with sheets or trabeculae, and may include small, focally arranged tubules. Cytoplasmic granularity is another distinguishing feature.\n\n"
            "Analyze all {num_images} slides provided and determine the most likely subtype based on the consistency of features observed across the regions. Considering these characteristics, **provide only the classification word** — 'CCRCC' or 'PRCC' or 'CHRCC' — without any additional text or explanation. You have to make a decision even though you are unsure."
        )
    elif cancer_type == "ESO":
        final_prompt = (
            f"You are provided {num_images} histological slides from a tissue biopsy. Each slide highlights a specific region of interest (ROI) containing key cellular structures. "
            "Your task is to classify the slides into one of four subtypes: Esophageal Adenocarcinoma (ESCA), Esophageal Squamous Cell Carcinoma (ESCC), "
            "or Stomach Adenocarcinoma (STAD). "
            "Use the detailed histological features below to guide your decision-making:\n\n"
            "- **ESCA (Esophageal Adenocarcinoma)**: Typically arises in the lower esophagus and is associated with Barrett's esophagus. Tumors show glandular differentiation, often forming tubular or papillary structures. Cells may exhibit mucin production and have prominent nucleoli.\n"
            "- **ESCC (Esophageal Squamous Cell Carcinoma)**: Occurs in the middle or upper esophagus. Displays squamous differentiation with features like keratin pearls, intercellular bridges, and polygonal cells. Nuclei are hyperchromatic with irregular contours.\n"
            "- **STAD (Stomach Adenocarcinoma)**: Frequently exhibits glandular or tubular structures, with varying degrees of differentiation. Tumors may show mucin production, nuclear pleomorphism, and invasion into the gastric wall. Tumor cells are typically arranged in irregular glands or sheets.\n\n"
            "Analyze all {num_images} slides provided and determine the most likely subtype based on the consistency of features observed across the regions. Considering these characteristics, **provide only the classification word** — 'ESCA', 'ESCC', or 'STAD' — without any additional text or explanation. You have to make a decision even though you are unsure."
        )
    elif cancer_type == "HEP":
        final_prompt = (
            f"You are provided {num_images} histological slides from a tissue biopsy. Each slide highlights a specific region of interest (ROI) containing key cellular structures. "
            "Your task is to classify the slides into one of two subtypes: Cholangiocarcinoma (CHOL) or Hepatocellular Carcinoma (HCC). "
            "Use the detailed histological features below to guide your decision-making:\n\n"
            "- **CHOL (Cholangiocarcinoma)**: Arises from the bile ducts and typically forms glandular or tubular structures with dense fibrous stroma. Tumor cells are cuboidal to columnar with prominent nucleoli and may show mucin production. Desmoplastic stromal reaction is a hallmark feature.\n"
            "- **HCC (Hepatocellular Carcinoma)**: Originates in hepatocytes, with tumor cells arranged in trabecular, pseudoacinar, or solid patterns. Cells often have abundant eosinophilic cytoplasm, round nuclei, and prominent nucleoli. Intracytoplasmic bile or Mallory bodies may be present. Endothelial-lined vascular spaces are frequently observed.\n\n"
            "Analyze all {num_images} slides provided and determine the most likely subtype based on the consistency of features observed across the regions. Considering these characteristics, **provide only the classification word** — 'CHOL' or 'HCC' — without any additional text or explanation. You have to make a decision even though you are unsure."
        )
    elif cancer_type == "GLIOMA":
        final_prompt = (
            f"You are provided {num_images} histological slides from a tissue biopsy. Each slide highlights a specific region of interest (ROI) containing key cellular structures. "
            "Your task is to classify the slides into one of two subtypes: Glioblastoma (GBM) or Oligodendroglioma (ODG). "
            "Use the detailed histological features below to guide your decision-making:\n\n"
            "- **GBM (Glioblastoma)**: High-grade astrocytic tumor characterized by marked cellular atypia, high mitotic activity, necrosis, and microvascular proliferation. Necrotic areas are often surrounded by hypercellular regions (pseudopalisading necrosis). Cells may appear pleomorphic with hyperchromatic nuclei.\n"
            "- **ODG (Oligodendroglioma)**: Low- to intermediate-grade glioma with tumor cells forming a 'fried-egg' appearance due to clear cytoplasmic halos. Tumors often contain delicate branching capillaries resembling a 'chicken-wire' pattern. Calcifications and uniform, round nuclei are commonly observed.\n\n"
            "Analyze all {num_images} slides provided and determine the most likely subtype based on the consistency of features observed across the regions. Considering these characteristics, **provide only the classification word** — 'GBM' or 'ODG' — without any additional text or explanation. You have to make a decision even though you are unsure."
        )
    elif cancer_type == "ADREN":
        final_prompt = (
            f"You are provided {num_images} histological slides from a tissue biopsy. Each slide highlights a specific region of interest (ROI) containing key cellular structures. "
            "Your task is to classify the slides into one of two subtypes: Adrenocortical Carcinoma (ACC) or Pheochromocytoma (PHC). "
            "Use the detailed histological features below to guide your decision-making:\n\n"
            "- **ACC (Adrenocortical Carcinoma)**: Malignant tumor of the adrenal cortex, often showing high nuclear pleomorphism, atypical mitoses, necrosis, and increased mitotic rate. Cells may have eosinophilic or clear cytoplasm, arranged in sheets or nests.\n"
            "- **PHC (Pheochromocytoma)**: Tumor arising from chromaffin cells of the adrenal medulla, characterized by large polygonal cells with abundant granular cytoplasm. Zellballen pattern with highly vascularized stroma is common, and nuclei are round with a salt-and-pepper chromatin pattern.\n\n"
            "Analyze all {num_images} slides provided and determine the most likely subtype based on the consistency of features observed across the regions. Considering these characteristics, **provide only the classification word** — 'ACC' or 'PHC' — without any additional text or explanation. You have to make a decision even though you are unsure."
        )

    elif cancer_type == "CERVIX":
        final_prompt = (
            f"You are provided {num_images} histological slides from a cervical biopsy. Each slide highlights a specific region of interest (ROI) containing key cellular structures. "
            "Your task is to classify the slides into one of two subtypes: Cervical Squamous Cell Carcinoma (CESC) or Endocervical Adenocarcinoma (ECAD). "
            "Use the detailed histological features below to guide your decision-making:\n\n"
            "- **CESC (Cervical Squamous Cell Carcinoma)**: Derived from the squamous epithelium, showing nests of malignant cells with keratinization and intercellular bridges. Tumor cells may have pleomorphic, hyperchromatic nuclei and eosinophilic cytoplasm.\n"
            "- **ECAD (Endocervical Adenocarcinoma)**: Glandular malignancy arising from the endocervical canal, characterized by columnar cells forming irregular glands with mucin production. Nuclear atypia and mitotic figures are commonly seen.\n\n"
            "Analyze all {num_images} slides provided and determine the most likely subtype based on the consistency of features observed across the regions. Considering these characteristics, **provide only the classification word** — 'CESC' or 'ECAD' — without any additional text or explanation. You have to make a decision even though you are unsure."
        )

    elif cancer_type == "PLEURA":
        final_prompt = (
            f"You are provided {num_images} histological slides from a pleural biopsy. Each slide highlights a specific region of interest (ROI) containing key cellular structures. "
            "Your task is to classify the slides into one of three subtypes: Pleural Mesothelioma, Biphasic Type (PLBMESO), Pleural Mesothelioma, Epithelioid Type (PLEMESO), or Pleural Mesothelioma, Sarcomatoid Type (PLSMESO). "
            "Use the detailed histological features below to guide your decision-making:\n\n"
            "- **PLBMESO (Biphasic Mesothelioma)**: Contains both epithelioid and sarcomatoid components, with at least 10% of each pattern present. Epithelioid areas appear glandular or tubulopapillary, while sarcomatoid regions consist of spindle cells.\n"
            "- **PLEMESO (Epithelioid Mesothelioma)**: Composed of uniform, polygonal tumor cells forming tubulopapillary, trabecular, or solid patterns. Cells have round nuclei, prominent nucleoli, and abundant eosinophilic cytoplasm.\n"
            "- **PLSMESO (Sarcomatoid Mesothelioma)**: Highly aggressive subtype with elongated spindle cells arranged in fascicles. Lacks glandular differentiation and often resembles fibrosarcoma or other soft tissue sarcomas.\n\n"
            "Analyze all {num_images} slides provided and determine the most likely subtype based on the consistency of features observed across the regions. Considering these characteristics, **provide only the classification word** — 'PLBMESO', 'PLEMESO', or 'PLSMESO' — without any additional text or explanation. You have to make a decision even though you are unsure."
        )

    elif cancer_type == "SOFT":
        final_prompt = (
            f"You are provided {num_images} histological slides from a soft tissue biopsy. Each slide highlights a specific region of interest (ROI) containing key cellular structures. "
            "Your task is to classify the slides into one of four subtypes: Dedifferentiated Liposarcoma (DDLS), Leiomyosarcoma (LMS), Myxofibrosarcoma (MFS), or Undifferentiated Pleomorphic Sarcoma (MFH). "
            "Use the detailed histological features below to guide your decision-making:\n\n"
            "- **DDLS (Dedifferentiated Liposarcoma)**: Displays well-differentiated liposarcoma adjacent to high-grade non-lipogenic sarcoma. The dedifferentiated areas contain pleomorphic spindle cells with high mitotic activity.\n"
            "- **LMS (Leiomyosarcoma)**: Malignant smooth muscle tumor composed of elongated spindle cells with cigar-shaped nuclei. Tumors exhibit eosinophilic cytoplasm, nuclear pleomorphism, and frequent mitoses.\n"
            "- **MFS (Myxofibrosarcoma)**: Characterized by myxoid stroma with curvilinear vasculature and scattered pleomorphic tumor cells. Low-grade tumors have more myxoid matrix, while high-grade variants show increased cellular atypia.\n"
            "- **MFH (Undifferentiated Pleomorphic Sarcoma)**: Poorly differentiated sarcoma with bizarre, pleomorphic cells arranged in storiform or haphazard patterns. Features include high mitotic activity and tumor giant cells.\n\n"
            "Analyze all {num_images} slides provided and determine the most likely subtype based on the consistency of features observed across the regions. Considering these characteristics, **provide only the classification word** — 'DDLS', 'LMS', 'MFS', or 'MFH' — without any additional text or explanation. You have to make a decision even though you are unsure."
        )

    elif cancer_type == "TESTIS":
        final_prompt = (
            f"You are provided {num_images} histological slides from a testicular biopsy. Each slide highlights a specific region of interest (ROI) containing key cellular structures. "
            "Your task is to classify the slides into one of two subtypes: Seminoma (SEM) or Mixed Germ Cell Tumor (MGCT). "
            "Use the detailed histological features below to guide your decision-making:\n\n"
            "- **SEM (Seminoma)**: Composed of uniform polygonal tumor cells with clear cytoplasm and centrally located nuclei. Tumors have fibrous septa infiltrated by lymphocytes and may contain granulomas.\n"
            "- **MGCT (Mixed Germ Cell Tumor)**: Contains multiple germ cell tumor components, such as embryonal carcinoma, yolk sac tumor, choriocarcinoma, and teratoma. Each component has distinct histological features, including glandular, solid, or papillary growth patterns.\n\n"
            "Analyze all {num_images} slides provided and determine the most likely subtype based on the consistency of features observed across the regions. Considering these characteristics, **provide only the classification word** — 'SEM' or 'MGCT' — without any additional text or explanation. You have to make a decision even though you are unsure."
        )

    elif cancer_type == "UTERUS":
        final_prompt = (
            f"You are provided {num_images} histological slides from a uterine biopsy. Each slide highlights a specific region of interest (ROI) containing key cellular structures. "
            "Your task is to classify the slides into one of two subtypes: Uterine Endometrioid Carcinoma (UEC) or Uterine Serous Carcinoma (USC). "
            "Use the detailed histological features below to guide your decision-making:\n\n"
            "- **UEC (Uterine Endometrioid Carcinoma)**: Common endometrial malignancy with glandular structures resembling normal endometrium. Tumor cells have round to oval nuclei with variable atypia and occasional squamous differentiation.\n"
            "- **USC (Uterine Serous Carcinoma)**: Aggressive high-grade carcinoma with papillary or glandular architecture. Tumor cells exhibit marked pleomorphism, high mitotic index, and prominent nucleoli.\n\n"
            "Analyze all {num_images} slides provided and determine the most likely subtype based on the consistency of features observed across the regions. Considering these characteristics, **provide only the classification word** — 'UEC' or 'USC' — without any additional text or explanation. You have to make a decision even though you are unsure."
        )
    else:
        raise ValueError(f"Unknown cancer type: {cancer_type}")

    return final_prompt

def get_final_prompt_with_multiple_images_vqa(cancer_type, vqa_msg, num_images):
    if not vqa_msg:
        raise ValueError("VQA message list is empty.")

    if cancer_type == "BRCA":
        # Format the prompt with multiple questions
        questions_block = "\n\n".join(
            [msg["content"] for msg in vqa_msg]
        )

        final_prompt = (
            f"You are a medical AI assistant trained to analyze pathology slides and answer multiple-choice "
            f"questions related to cancer diagnosis. You are provided {num_images} histological slides from a tissue biopsy."
            "Each slide highlights a specific region of interest (ROI) containing key cellular structures."
            "For each question, select the most appropriate answer "
            "from the given choices. If you are uncertain, select the answer that is the closest match based "
            "on available information. Your response must strictly follow the order of the questions, and answers "
            "should be separated by a comma. Do not provide any explanations or additional information.\n\n"
            f"{questions_block}\n\n"
            f"Analyze all {num_images} slides provided and determine the most likely answer for each question."
            "Answers:"
        )
    else:
        raise ValueError(f"Unknown cancer type: {cancer_type}")
    
    return final_prompt
    