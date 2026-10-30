# Project 2
## Week 05-Scene Classification with Transfer Learning
---



## 📌 Project Overview
This project investigates deep learning applications through convolutional neural networks (CNNs) and transfer learning using PyTorch. The primary goal is to build an end-to-end image classification pipeline that leverages a pretrained model to categorize natural scenes efficiently. By contrasting **Feature Extraction** (frozen backbone) with **Fine-Tuning** (unfreezing deeper layers), this project evaluates how model adaptability scales with domain-specific training.

---

## 📊 Dataset: Intel Image Classification
* **Source:** Kaggle (`puneet6060/intel-image-classification`)
* **Task:** Multi-class natural scene image classification across 6 distinct categories (`buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`).
  
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


* **Preprocessing & Transforms**: Images are resized to $224 \times 224$ pixels and normalized according to ImageNet standards.
* **Backbone**: Transfer learning with PyTorch `models.resnet18(pretrained=True)` with a custom fully connected head adjusted for 4 output classes.
* **Training & Optimization**: Cross-Entropy loss with Stratified Group K-Fold / Train-Test splitting to preserve target class ratios.

---

## Experimental Workflow

### 1. Dataset Indexing & Audit
* Built a custom file indexer scanning subdirectories for valid image extensions (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`).
* Performed integrity checks for missing/corrupted files and image dimensions.

### 2. Model Training & Evaluation
* Trained ResNet18 using PyTorch on CUDA acceleration.
* Monitored loss and Macro-F1 across validation splits.
* Computed Confusion Matrix and Per-Class Precision/Recall metrics.

---


# Week 7: Text and Multimodal AI - Project Repository

The project covers manual TF-IDF implementation and vector similarity, robust text classification for spam detection, domain-adapted Named Entity Recognition (NER) for agriculture, regular expression extraction with negative assertions, OCR error correction, and multimodal case record building with automated PII masking.

---

## TF-IDF by Hand

Manual calculation of TF-IDF representations and cosine similarities, verified against `scikit-learn`'s `TfidfVectorizer` over a 3-document corpus:
* $d_1$: `"win free cash now"`
* $d_2$: `"call me now"`
* $d_3$: `"free call now"`

### Key Analytical Findings
* **Smoothed IDF Formula:** Utilizes $\text{tfidf} = \ln\left(\frac{1 + N}{1 + df}\right) + 1$ (where $N = 3$).
  * For `"free"` ($df = 2$): $\ln(4 / 3) + 1 \approx \mathbf{1.288}$
  * For `"now"` ($df = 3$): $\ln(4 / 4) + 1 = \mathbf{1.000}$
* **Discriminative Weighting:** The term `"now"` receives the baseline score of $1.000$ because it appears across all documents ($df = 3$), nullifying its inverse document frequency component ($\ln(1) = 0$) and rendering it entirely non-discriminative.
* **Cosine Similarity:** The similarity score between $d_2$ and $d_3$ computes to **$0.54$**, perfectly matching `scikit-learn` outputs.

---

## Text Classification (SMS Spam Collection)

Trained and evaluated on the **SMS Spam Collection Dataset** containing `5,572` total messages (`4,825` ham, `747` spam; spam proportion: `13.4%`, baseline "always ham" accuracy: `86.6%`).

### Model Performance Summary
* **Word 1-2 Grams Classifier:** Macro-F1: **`0.974`** | Feature Space: `12,128` | Spam Precision: `0.979` | Spam Recall: `0.933`
* **Character 2-5 Grams Classifier (`char_wb`):** Macro-F1: **`0.982`** | Feature Space: `36,449`
* **Obfuscation Robustness:** Tested against adversarial strings such as `"W1N a FR33 pr1ze!! txt CLAIM now"`, achieving a confident spam probability score of **`0.85`**.

### Top Predictive Features
* **Spam Indicators(Top Positive Features):** `['call', 'txt', 'free', 'text', 'reply', 'uk', 'to', 'www', 'mobile', 'claim', 'stop', 'your']`
* **Ham Indicators(Top Negative Features):** `['my', 'me', 'ok', 'that', 'it', 'but', 'll', 'gt', 'lt', 'can', 'what', 'da']`

---

## Named Entity Recognition (NER), Measured

Domain-adapted entity extraction evaluated on specialized sentences concerning **Agriculture (Cotton Disease)** using `spaCy` (`en_core_web_sm`) augmented with a custom `EntityRuler`.

### Evaluation Results (Held-Out Test Set: 10 Sentences / 35 Total Gold Entities)
* **spaCy Out-of-the-Box:** Precision: `0.778` | Recall: `0.700` | **F1: 0.737**
* **spaCy + Custom Entity Ruler & Title Stripping:** Precision: `0.947` | Recall: `0.900` | **F1: 0.923** (representing an absolute **+0.186** F1 gain)

---

## Regex Fields with Negative Tests

Designed rigorous regular expression patterns for agricultural domain extraction, backed by strict negative assertions to prevent false positives over unformatted text or non-target entities.

| Pattern Name | Target Entity | Example Match | Test Status |
| :--- | :--- | :--- | :--- |
| `disease_pest` | Specific cotton diseases, pests, and treatments | `"cotton leaf curl virus"`, `"whitefly"` | ✅ Passed |
| `location` | Key cotton-belt districts (e.g., Multan, Bahawalpur) | `"Multan"`, `"Vehari"` | ✅ Passed (Rejects non-target cities) |
| `amount` | Monetary values with currency prefixes | `"Rs. 15,000"`, `"PKR 28,000"` | ✅ Passed (Rejects bare integers) |
| `date` | Numeric and written calendar dates | `"14 August 2026"`, `"02/09/2026"` | ✅ Passed |
| `metric` | Survey and impact metrics | `"35 fields"`, `"120 farmers"` | ✅ Passed |

---

## Character Error Rate (CER) and OCR Repair

Evaluated optical character recognition pipelines and implemented deterministic context post-processing rules to correct frequent character substitutions (such as letter/digit confusions like $\text{O} \leftrightarrow 0$ and $\text{l/I} \leftrightarrow 1$).

* **Synthetic Invoice OCR Extraction:** Achieved a clean Character Error Rate (CER) of **`0.000`** on high-contrast renders.
* **Context Rules Post-Processor (`fix_ocr`):** Automatically standardizes damaged tokens (e.g., mapping `Invoice N0. 4417` $\rightarrow$ `Invoice No. 4417` and `12,5O0` $\rightarrow$ `12,500`), effectively mitigating real-world scan anomalies.

---

## Multimodal Case Record & Privacy Masking

Integrated the **ResNet-18** image classification backbone from Week 6 (`['diseased cotton leaf', 'diseased cotton plant', 'fresh cotton leaf', 'fresh cotton plant']`) with text evidence extraction and stringent PII masking routines.

### Privacy Compliance & Scrubbing
Before logging or serializing records, sensitive personally identifiable information (PII) including CNICs, phone numbers, and email addresses undergoes automated masking:
```python
sample = "Farmer Imran Sheikh (CNIC: 35202-1234567-1) called 0300-1234567 via info@engro.com.pk"
# Output -> "Farmer Imran Sheikh (CNIC: [CNIC]) called [PHONE] via [EMAIL]"
```

### Sample Output JSON Record
```json
{
  "image_evidence": {
    "diseased cotton leaf": 0.892,
    "diseased cotton plant": 0.081,
    "fresh cotton leaf": 0.021,
    "fresh cotton plant": 0.006
  },
  "text_evidence": {
    "entities": [
      ["Imran Sheikh", "PERSON"],
      ["Bahawalpur", "GPE"],
      ["14 August 2026", "DATE"],
      ["cotton leaf curl virus", "DISEASE"]
    ],
    "fields": {
      "disease_pest": ["cotton leaf curl virus"],
      "location": ["Bahawalpur"],
      "date": ["14 August 2026"]
    },
    "urgent_terms": [
      "outbreak",
      "virus"
    ]
  }
}
---



