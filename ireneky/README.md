# 🧬 Computational Biology Capstone

# Enhancing scHi-C Data and Clustering with HiCFoundation and Higashi

This repository contains the code and results on the project; exploring how enhancement of sparse single-cell Hi-C data (scHi-C) using **HiCFoundation** affects downstream **embedding and clustering** with **Higashi**.

## Project Overview
HiCFoundation has a couple of applications, this work focuses on the single-cell HiC application. 
The Workflow of this project 

I benchmarked Higashi on:
-  **Baseline dataset** (16k+ cells across 5 human cell types) — reproduced original UMAP and clustering results
-  **Enhanced dataset** — 400 mouse brain scHi-C cells processed with HiCFoundation, then embedded and clustered

##  Key Components

- `baseline_pipeline/`: Runs the original Higashi tutorial on benchmark data
- `enhanced_pipeline/`: Processes HiRES brain data with HiCFoundation, then applies Higashi
- `config/`: JSON config files used for Higashi training
- `higashi_data_v1/`: contains data .txt for 400 .pairs enhanced scHiC data (using HiCFoundation) and label_info.pickle which contains the metadata. **not included due to large file size
- `report/`: Final report (PDF/Word) detailing results and conclusions
- `capstone_project.ipynb`: This is the script for running HiCFoundation on the data and converting to Higashi input format. ps: (may need to be downloaded to view )
- `capstone_project_2.ipynb`: Script for generating embedding for clustering
- `Higashi_eg_data.ipynb`: Script for regenerating results from original paper


##  Main Findings

- Higashi embeddings on enhanced data **retain some biological structure**, but clustering quality is **limited by cell count and training time**.
- Enhancement improves contact matrix density, but **more cells and longer training** are needed to recover meaningful clusters.

---

##  Workflow Summary

```bash
Raw scHi-C (.pairs)
        │
        ▼
[ HiCFoundation ]
Enhanced contact maps (.pkl or .pairs)
        │
        ▼
[ Format Conversion ]
.txt matrix for Higashi
        │
        ▼
[ Higashi ]
UMAP embeddings, clustering, visualization 

```

## Data 

The data run for baseline is the 4DN Human cell line data from [https://noble.gs.washington.edu/proj/schic-topic-model]
The data used for this workflow is the mouse brain HiRES dataset from GEO[accession number: GSE223917]

Input data:

- .pairs
- .txt
- label_info.pickle

##  Dependencies

- Python 3.9+
- PyTorch (CUDA if available)
- HiCFoundation ([repo](https://github.com/Noble-Lab/HiCFoundation))
- Higashi ([repo](https://github.com/ma-compbio/Higashi/tree/main))
- UMAP, seaborn, scikit-learn
