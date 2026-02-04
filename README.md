# YOLO11 Document Segmentation

A YOLO11-based segmentation project for detecting and segmenting document regions in images and videos.  
This repository includes training and inference scripts, pretrained weights, Jupyter notebooks, and example results.

---

## ✨ Features

- YOLO11 segmentation model
- Image, batch, and video inference
- Segmentation mask extraction
- Custom annotation rendering
- Pretrained weights included (~45 MB)
- Jupyter notebooks for interactive demos

---

## 📂 Project Structure

```text
yolo11-document-segmentation/
│
├── src/
│   ├── train.py
│   └── inference.py
│
├── weights/
│   └── best.pt
│
├── dataset/
│   ├── data.yaml
│   └── README.md
│
├── examples/
│   ├── input_1.jpg
│   └── output_1.jpg
│
├── docs/
│   └── QUICKSTART.md
│
├── requirements.txt
└── README.md

---

## ⚙️ Installation

```bash
pip install -r requirements.txt

## 🚀 Quick Inference

```bash
python src/inference.py

## 🧠 Training

```bash
python src/test.py

