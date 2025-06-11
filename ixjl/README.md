# HiCFoundation with Sequence Data

This repository contains a modified version of the [HiCFoundation model](https://github.com/ma-compbio/HiCFoundation) that incorporates DNA sequence information to enhance Hi-C contact map resolution. The extended architecture integrates a lightweight 1D CNN-based sequence encoder with the original Hi-C encoder. The report is contained in the finalReport.pdf file and the model checkpoints can be found in the following google drive [folder link](https://drive.google.com/drive/folders/1CYCTwA6voRdYyYaCjHhjU5e9atqEcTyC?usp=drive_link). Note: This repository contains code from the HiCFoundation model repo which is not used in the final project.

---

## 🚀 Key Features

- ✅ Dual-encoder architecture: Hi-C + DNA sequence
- ⚡ Lightweight 1D CNN for genomic sequence embedding
- 📈 Loss: Mean Squared Error (MSE) + Structural Similarity Index (SSIM)
- 🧪 Compatible with `.mcool` Hi-C files and `.fa` FASTA genome files
- 📊 Includes training and evaluation scripts

---

## 🏗️ Model Overview

```
[Hi-C Contact Map] --> [Hi-C Encoder] ----\
                                           \
                                            --> [Decoder] --> High-Resolution Prediction
[DNA Sequence] --> [Sequence CNN Encoder]--/
```

---

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## 🧪 Running the Model

### Finetune
This is an example command used to finetune the model. The arguments are the same as in the original HiCFoundation model's finetune.py, but patience and sequence were added. Additionally, since my project used a different kind of loss function, the loss_types now range from 1 to 3.

```bash
python finetune.py \
  --epochs 32 \
  --batch_size 4 \
  --num_worker 2 \
  --patience 5 \
  --accum_iter 4 \
  --pretrain /storage/ixjl/cse428/hicfoundation_model/hicfoundation_pretrain.pth.tar \
  --data_path /storage/ixjl/cse428/processed_finetune_data_multiple_regions \
  --train_config /storage/ixjl/cse428/processed_finetune_data_multiple_regions/train.txt \
  --valid_config /storage/ixjl/cse428/processed_finetune_data_multiple_regions/val.txt \
  --output ./output_lowRes_4DNFI643OYP9 \
  --loss_type=3 \
  --save_freq 8 \
  --tensorboard 1 \
  --sequence 1
```

### Predict
The arguments used by the predict program are modeled after the finetune program. This will generate the high resolution predictions which can then be used for evaluation.

```bash
python predict.py \
  --data_path /storage/ixjl/cse428/processed_finetune_data_multiple_regions \
  --config_file /storage/ixjl/cse428/processed_finetune_data_multiple_regions/eval.txt \
  --checkpoint_path /storage/ixjl/cse428/output_both_files/model/model_best.pth.tar \
  --output_dir ./predictions/model_both_files \
  --input_row_size 224 \
  --input_col_size 224 \
  --patch_size 16 \
  --batch_size 1 \
  --num_workers 1 \
  --sequence 1 \
  --device cuda
```

### Evaluate
To evaluate model performance, the target and predicted matrices are compared. Below is an example command.

```bash
python evaluation/evaluation.py \
  --pred_dir /storage/ixjl/cse428/predictions/model_both_files \
  --data_dir /storage/ixjl/cse428/processed_finetune_data_multiple_regions/evaluation_data \
  --output_json ./evaluation/model_both_files/evaluation_summary.json
```

---

## 📊 Results Summary

| Model | Mean MSE ↓ | Mean PCC ↑ |
|-------|------------|------------|
| Hi-C only (both datasets) | 0.000390 | 0.9038 |
| Hi-C + sequence (both datasets) | **0.000304** | **0.9037** |

---

## 📂 Datasets

You can download the datasets used from the Orca repository under their download section:

🔗 Orca Repository: [[link](https://github.com/jzhoulab/orca/blob/main/README.md)]
