# Project 2
## Week 05-Scene Classification with Transfer Learning
---



## 📌 Project Overview
This project investigates deep learning applications through convolutional neural networks (CNNs) and transfer learning using PyTorch. The primary goal is to build an end-to-end image classification pipeline that leverages a pretrained model to categorize natural scenes efficiently. By contrasting **Feature Extraction** (frozen backbone) with **Fine-Tuning** (unfreezing deeper layers), this project evaluates how model adaptability scales with domain-specific training.

---

## 📊 Dataset: Intel Image Classification
* **Source:** Kaggle (`puneet6060/intel-image-classification`)
* **Task:** Multi-class natural scene image classification across 6 distinct categories (`buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`)[cite: 1].
  
* **Dataset Scale & Class Distribution:**
  * **Buildings:** 2,191 train images / 437 test images
  * **Forest:** 2,271 train images / 474 test images
  * **Glacier:** 2,404 train images / 553 test images
  * **Mountain:** 2,512 train images / 525 test images
  * **Sea:** 2,274 train images / 510 test images
  * **Street:** 2,382 train images / 501 test images
  * **Total Volume:** 14,034 training images / 3,000 test images
  * **Training / Validation Split:** The training directory is split using an 85/15 ratio, resulting in **11,929 training images** and **2,105 validation images**.

---

## ⚙️ Data Preprocessing & Augmentation Pipeline
To ensure robust model generalization and compatibility with the pretrained architecture, the pipeline applies standardized transformations:
* **Resizing & Cropping:** Images are resized and center-cropped to 224*224 pixels to match the input tensor dimensions expected by ResNet-18.
* **Normalization:** Pixel values are normalized using the standard ImageNet dataset statistics:
  * *Mean:* `[0.485, 0.456, 0.406]`
  * *Standard Deviation:* `[0.229, 0.224, 0.225]`
* **Augmentation (Training Set):** Random horizontal flips and subtle affine transformations are applied to prevent overfitting and improve variance coverage across natural scenery types.

---

## 🛠️ Model Architecture & Transfer Learning Strategy
The project is built on **ResNet-18**, a lightweight and efficient residual convolutional neural network pretrained on ImageNet. Two progressive training stages are executed:

### 1. Phase 1: Feature Extraction (Frozen Backbone)
* **Mechanism:** All convolutional layers in the feature extraction base are frozen (`requires_grad = False`). Only the newly initialized fully connected classification head is updated.
* **Trainable Parameters:** Exactly **3,078 trainable parameters**.
* **Objective:** Assess how well generic ImageNet feature maps separate natural outdoor scenes without altering lower-level edge and texture filters.

### 2. Phase 2: Fine-Tuning (`layer4` + Classifier)
* **Mechanism:** The final residual block (`layer4`) is unfrozen alongside the classification head to allow high-level semantic feature adaptation for the target scene classes.
* **Trainable Parameters:** Approximately **8.4 million trainable parameters**.
* **Objective:** Enable the network to adapt specialized high-level configurations (e.g., distinguishing architectural geometric lines in `buildings`/`street` vs. organic textures in `forest`/`mountain`).

---

## 📈 Results & Experimental Findings

| Training Phase | Trainable Parameters | Validation Accuracy | Key Observations |
| :--- | :--- | :--- | :--- |
| **Phase 1: Feature Extraction** | 3,078 | **90.8%** | Fast convergence; establishes a strong baseline using frozen pre-learned weights. |
| **Phase 2: Fine-Tuning (`layer4`)** | ~8.4 Million | **94.1%** | Yields higher overall accuracy by tuning high-level features for specific scene structures. |

* **Test Accuracy (Evaluated Once):** **93.8%** *(Locked evaluation on the 3,000-image test set)*
* **Weakest Class:** `buildings` (frequently misclassified or confused with `street` due to overlapping urban architectural elements and asphalt textures).

---

## 🔍 Comprehensive Error Analysis
* **Most Instructive Error:** A street-level urban photograph featuring prominent building facades and storefront windows was incorrectly categorized as a `building` class due to the dense structural geometry occupying the primary visual frame rather than the visible road perspective below.
* **Takeaway:** The model demonstrates robust feature retention across natural landscapes, though structural similarities between urban settings (`buildings` vs. `street`) and elevated topologies (`glacier` vs. `mountain`) present the primary source of classification confusion.

---


## 🚀 Week 6 Proposal: AI-Based Manufacturing Defect Classification
Building directly upon the transfer learning methodology mastered in Week 5, the upcoming week pivots from natural scene identification toward industrial automation and quality assurance:

* **Objective:** Develop an end-to-end image classification model that identifies manufacturing defects using transfer learning and a pretrained CNN backbone. The system will categorize product visual data into distinct defect classes to flag structural anomalies and support automated factory inspection pipelines.
* **Data Source:** A structured, labeled image dataset sourced from Kaggle containing multiple defect classes with balanced splits. Dataset license validation, direct links, specific class labels, and total sample quantities will be documented prior to pipeline execution.
* **Methodology:** Implement a comparative framework analyzing frozen feature extraction against targeted fine-tuning. Performance will be rigorously benchmarked using overall accuracy, precision, recall, F1-scores, and normalized confusion matrices.
* **Application Value:** Establishes a reliable framework for computer-vision-based quality control, minimizing manual oversight in industrial manufacturing lines.

---


# Week 6 — Cotton Disease Detection with ResNet18

## Overview
Week 6 file contains the implementation and evaluation of a deep learning pipeline designed to classify cotton crop health from leaf and plant images using a pre-trained **ResNet18** vision backbone. Accurate and automated detection of cotton plant diseases enables timely intervention, preventing catastrophic yield losses for farmers while minimizing unnecessary pesticide usage.

---

## Technical Summary Dashboard

| Metric / Item | Detail |
| :--- | :--- |
| **Who Benefits** | Cotton farmers and agricultural extension workers in field and remote settings. |
| **Why It Matters** | Early disease diagnosis prevents severe yield loss, financial distress, and targeted chemical application. |
| **Dataset** | **Cotton Disease Dataset** (`janmejaybhoi` on Kaggle), Open Dataset; 1,951 images across 4 classes. |
| **Data Audit** | Imbalance ratio: **2.8** (815/288); Exact duplicates: **34**; Near-duplicates (train/test): **4**; Grouping: 1.00 img/group (`GROUP_PATTERN = None`). |
| **Imbalance Handling** | Handled via Stratified Group Split / Stratified Sampling and Weighted Cross-Entropy Loss (weights: 1.693, 0.599, 1.143, 1.157). |
| **Phase 1 Validation F1** | **0.870** (Head only @ Epoch 2) |
| **Phase 2 Validation F1** | **0.982** (Fine-tune `layer4` @ Epoch 3) |
| **Test Set Macro-F1** | **0.102** (Model collapse detected on test set; weak classes F1 = 0.000) |
| **Grad-CAM Verdict** | **No shortcut found**; attention localized on leaf lesions, spots, and structural veins. |
| **Limitations** | Unseen field conditions, extreme weather, camera variations. Human expert oversight required. |

---

## Dataset Details & Class Breakdown

The dataset contains **1,951 images** categorized into 4 distinct classes:

| Class Label | Image Count | Proportion (%) | Class Weight |
| :--- | :--- | :--- | :--- |
| **diseased cotton leaf** | 288 | 14.76% | 1.693 |
| **diseased cotton plant** | 815 | 41.77% | 0.599 |
| **fresh cotton leaf** | 427 | 21.89% | 1.143 |
| **fresh cotton plant** | 421 | 21.58% | 1.157 |
| **Total** | **1,951** | **100.0%** | — |

* **Imbalance Ratio**: 815/288 = 2.8
* **Images Per Group Ratio**: 1.00 (Filenames do not contain structured plant IDs).

---

## Class Weight Arithmetic

Class weights are computed using the balanced weight formula:
Weight_c = N/K * n_c

Where:
* N = 1463 (total training samples)
* K = 4 (number of classes)
* n_c = training sample count for class c

**Example calculation for `diseased cotton leaf`:**

Weight = 1463/4 * 216 = 1.693

---

## Pipeline Architecture & Configuration
### Key Pipeline Constants
| Constant | Value | Description |
| :--- | :--- | :--- |
| **DATA_ROOT** | `'/kaggle/input/datasets/janmejaybhoi/cotton-disease-dataset/Cotton Disease'` | Dataset root directory |
| **TEST_DIR** | `None` | Carved from TRAIN_DIR via Stratified Group Split |
| **GRAYSCALE** | `False` | RGB color photos |
| **IMG_SIZE** | `224` | Input resolution (224x224) |
| **BACKBONE** | `'resnet18'` | Pre-trained ResNet-18 |
| **POSITIVE** | `None` | Multi-class setup (4 classes)|
| **TARGET_RECALL** | `0.95` | Thresholding target|


* **Preprocessing & Transforms**: Images are resized to $224 \times 224$ pixels and normalized according to ImageNet standards[cite: 1].
* **Backbone**: Transfer learning with PyTorch `models.resnet18(pretrained=True)` with a custom fully connected head adjusted for 4 output classes[cite: 1].
* **Training & Optimization**: Cross-Entropy loss with Stratified Group K-Fold / Train-Test splitting to preserve target class ratios[cite: 1].

---

## Experimental Workflow

### 1. Dataset Indexing & Audit
* Built a custom file indexer scanning subdirectories for valid image extensions (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`)[cite: 1].
* Performed integrity checks for missing/corrupted files and image dimensions[cite: 1].

### 2. Model Training & Evaluation
* Trained ResNet18 using PyTorch on CUDA acceleration[cite: 1].
* Monitored loss and Macro-F1 across validation splits[cite: 1].
* Computed Confusion Matrix and Per-Class Precision/Recall metrics[cite: 1].
