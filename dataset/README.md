# Dataset Description

## 📌 Overview

This folder is intended to store the dataset used for training, validation, and testing of the PCB inspection system. The dataset consists of PCB images and corresponding annotations for defect detection and component recognition tasks.

Due to file size limitations of GitHub, the **full dataset is hosted externally**.

---

## 📥 Dataset Access

The complete dataset is hosted externally and must be downloaded separately.

**Download link:**
https://drive.google.com/file/d/1S9XIyG6gjrY39nwRVd1CESb_zeVqXzXj/view

---

## 📂 Expected Folder Structure

After downloading and extracting the dataset archive, the directory should follow the structure below:

```
dataset/
├── models/            # Model-related files or references (if included)
├── org_images/        # Original, unprocessed PCB images
├── python_scripts/    # Helper scripts for preprocessing and dataset handling
├── runs/              # Training and experiment outputs (auto-generated)
├── train/             # Training images and labels
├── val/               # Validation images and labels
├── class_counter.txt  # Class distribution or count reference
├── classes.txt        # List of defect/component classes
├── extract.txt        # Notes or extraction logs
├── dataset_custom.yaml# Dataset configuration file (e.g., YOLO format)
```

> ⚠️ Some folders (such as `runs/`) may be generated automatically during training. Do not modify the structure unless reflected in the training configuration files.

---

## 🧪 Dataset Usage

The dataset is used for:

* Training defect detection models
* Training component recognition models
* System accuracy evaluation
* Processing time and performance testing

All experiments are conducted under **controlled imaging conditions** to ensure consistency and reliability of results.

---

## 📄 Disclaimer

This dataset is intended for **academic and research purposes only**. Redistribution or commercial use may require additional permissions.
