# Lightweight Scene Text Detection for Indian Street Images

## 📌 Overview
This repository contains the implementation and experimental study of a **lightweight scene text detection system** based on the EAST architecture, developed as part of a **Major Technical Project (MTP)**.

The work focuses on:
- Efficient text detection in real-world street images
- Reducing model size and inference latency
- Benchmarking on **ICDAR 2015**
- Laying the foundation for **multilingual Indian scene text recognition**

> 📄 Planned paper title:  
> **"Lightweight Multilingual Scene Text Detection and Recognition for Indian Street Images"**

---

## 🎯 Objectives
- Build a **lightweight yet accurate** scene text detector
- Study the impact of **pretraining vs scratch training**
- Analyze **accuracy–efficiency trade-offs**
- Prepare a system suitable for **edge / low-resource deployment**

---

## 🧠 Model Architecture
- Base Model: **EAST (Efficient and Accurate Scene Text Detector)**
- Backbone: **VGG16-BN**
- Training Modes:
  - From scratch
  - ImageNet pretrained
- Loss:
  - Dice Loss (classification)
  - Geometry + angle loss (localization)

---

## 📊 Experiments

| Experiment | Description | Epochs | F-measure |
|----------|------------|--------|-----------|
| Exp-1 | Scratch training (VGG16) | 20 | 0.044 |
| Exp-2 | ImageNet pretrained VGG16 | 20 | 0.143 |
| Exp-3 | Long training (ImageNet VGG16) | 60 | **0.711** |

Detailed metrics are available in:
experiments/*/metrics.json

---

## ⚙️ Metrics Reported
- Precision / Recall / F-measure (ICDAR protocol)
- Inference latency (ms)
- FPS
- GPU memory usage
- Model parameters (Millions)
- GFLOPs

---

## 📁 Repository Structure



---

## ⚙️ Metrics Reported
- Precision / Recall / F-measure (ICDAR protocol)
- Inference latency (ms)
- FPS
- GPU memory usage
- Model parameters (Millions)
- GFLOPs

---

## 📁 Repository Structure
.
├── src/ # Core model, loss, training, inference
├── notebooks/ # Experiment notebooks (Exp-1, Exp-2, Exp-3)
├── experiments/ # Metrics and logs per experiment
├── evaluate/ # Official ICDAR evaluation scripts
├── assets/ # Visual results
├── requirements.txt
└── README.md


---

## 🚀 Running Experiments

### Environment
```bash
conda create -n PytorchEAST python=3.9
conda activate PytorchEAST
pip install -r requirements.txt


Training

Use the notebooks:

notebooks/01_train_from_scratch.ipynb
notebooks/02_train_imagenet_vgg16.ipynb
notebooks/03_train_imagenet_vgg16_long.ipynb


📌 Current Status

✅ Scene text detection completed

⏳ Lightweight backbone experiments (planned)

⏳ Multilingual text recognition (planned)

⏳ Indian street text dataset experiments (planned)




👤 Author

Akash Khairal
M.Tech (Artificial Intelligence)
Indian Institute of Technology, Jodhpur


📜 License

This project is for academic and research purposes.



Save and exit nano.

---

## PART 3 — Commit YOUR README

```bash
git add README.md
git commit -m "Add project README for MTP and research work"
git push

