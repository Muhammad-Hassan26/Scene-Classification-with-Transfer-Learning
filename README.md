# Project 2: Scene Classification with Transfer Learning
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
