# 📌 Melanoma-Detection-Using-ResNet50-Apache-Spark-Grad-CAM-Explainability
This project builds a deep-learning pipeline for early melanoma detection using transfer learning, distributed preprocessing with Apache Spark, and Grad-CAM explainability to evaluate model reasoning. The system is trained using the HAM10000 dermoscopic image dataset and fine-tuned to classify melanoma vs. benign lesions.

---

## 🧠 Project Overview

This work combines:

- **ResNet50 transfer learning** (ImageNet-pretrained)
- **Distributed preprocessing** using Apache Spark on multiple virtual machines
- **Threshold-dependent evaluation** to address class imbalance
- **Explainability** using Gradient-weighted Class Activation Mapping (Grad-CAM)
- **Comprehensive performance reporting** (ROC, PR, confusion matrices, clinical metrics)

---

## 📂 Dataset

**HAM10000: Human Against Machine for Skin Lesion Analysis**  
- 10,015 dermoscopic images  
- 7 original diagnostic categories  
- Converted to **binary classification**: *melanoma* vs. *benign*  
- Highly imbalanced (≈11% melanoma)

Dataset link:  
🔗 https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

---

## ⚙️ Preprocessing Pipeline

### 🔸 Techniques Used  
- Resizing to 224×224  
- Normalization  
- Class-balanced augmentation  
- Distributed processing on **2-node Apache Spark cluster**

### 🔸 Spark Benefits  
- Preprocessing reduced from **~2 hours → ~30 minutes**  
- Enables scaling to larger datasets  
- Fault-tolerant and parallel by design

---

## 🏗️ Model Architecture

- **Backbone:** ResNet50 (frozen during initial training)  
- **Custom head layers:**  
  - GlobalAveragePooling2D  
  - Dense (256 units, ReLU)  
  - Dropout (0.5)  
  - Sigmoid output layer  
- **Total parameters:** 24.14M  
- **Trainable parameters:** 524K

---

## 📊 Evaluation Metrics

The model is evaluated using:

- Accuracy  
- AUROC  
- AUPRC  
- Sensitivity / Specificity  
- PPV / NPV  
- Confusion matrices  
- Threshold analysis  
- ROC curve & PR curve

---

## 🧪 Performance Summary

### **1️⃣ Default Threshold (0.50)**  
| Metric | Value |
|--------|--------|
| Accuracy | **0.3392** |
| AUROC | **0.8152** |
| AUPRC | **0.3326** |
| Sensitivity | **0.9936** |
| Specificity | **0.2624** |
| PPV | **0.1364** |
| NPV | **0.9971** |

**Confusion Matrix:**  
```
TN = 349,  FP = 981  
FN = 1,    TP = 155
```

---

### **2️⃣ Optimal Threshold (0.6747, Youden’s J)**  
| Metric | Value |
|--------|--------|
| Accuracy | **0.7167** |
| Sensitivity | **0.8077** |
| Specificity | **0.7060** |
| PPV | **0.2437** |
| NPV | **0.9690** |

**Confusion Matrix:**  
```
TN = 939,  FP = 391  
FN = 30,   TP = 126
```

---

## 🔍 Explainability: Grad-CAM Visualizations

Grad-CAM heatmaps revealed:

- Correct predictions sometimes relied on **background artifacts**  
- Some melanoma cases were misclassified due to **non-lesion attention**  
- Resolution limitations restricted lesion-level interpretation  
- Highlights the need for segmentation or lesion-centered cropping

---

## ⚠️ Limitations

- Single dataset (HAM10000) → limited generalization  
- Binary classification removes valuable diagnostic detail  
- Background-focused Grad-CAM patterns  
- Class imbalance remains a challenge  
- Transfer learning architecture not designed for dermoscopy images

---

## 🔭 Future Work

- Train and evaluate on **ISIC 2019** and **BCN20000**  
- Add **lesion segmentation** (U-Net / Mask R-CNN)  
- Explore **focal loss**, **class weighting**, or **GAN-based augmentation**  
- Fine-tune deeper ResNet layers  
- Collaborate with dermatologists for expert evaluation  
- Deploy in a clinical decision-support interface

---

## 📁 Repository Structure

```
Melanoma-Detection/
│── data/
│── preprocessing/
│── spark_cluster/
│── model/
│── evaluations/
│── gradcam_outputs/
│── notebooks/
│── README.md
```

---

## ▶️ Running the Project

### **1. Install requirements**
```bash
pip install -r requirements.txt
```

### **2. Start Spark cluster**
```bash
start-master.sh
start-worker.sh spark://<master-ip>:7077
```

### **3. Train model**
```bash
python train_model.py
```

### **4. Evaluate**
```bash
python evaluate_model.py
```

### **5. Generate Grad-CAM**
```bash
python grad_cam.py
```

---

## 👨🏽‍💻 Author

**Dennis Owusu**  
Master of Science in Health Informatics  
Michigan Technological University  
