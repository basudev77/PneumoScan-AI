# 🫁 PneumoScan AI

**Computer Vision and Deep Learning for Early Disease Diagnosis and Prediction**
Pneumonia detection from chest X-ray images using a DenseNet121 transfer-learning model with Grad-CAM explainability, deployed as an interactive Streamlit web app.

> Final Year Project — B.Tech CSE (AI), Budge Budge Institute of Technology
> Project Code: **PROJ-AI881** · **Project II** (8th Semester) · Project No. **60**
> Supervisor: **Prof. Subhadeep Majumdar**

🔗 **Live App:** [basudevdas-pneumoscan-ai.hf.space](https://basudevdas-pneumoscan-ai.hf.space/)

📦 **Repository:** [github.com/basudev77/PneumoScan-AI](https://github.com/basudev77/PneumoScan-AI)

---

## 📌 Overview

Pneumonia remains one of the leading causes of mortality worldwide, and manual chest X-ray interpretation is time-consuming, subjective, and prone to error — especially where access to experienced radiologists is limited. **PneumoScan AI** automates preliminary pneumonia screening from chest X-ray images and explains *why* it made each prediction, using a Grad-CAM heatmap overlay.

This is the **8th-semester (Phase II)** version of the project. In the 7th semester (Phase I), a shallow custom CNN was trained from scratch as a baseline. This phase upgrades the classification backbone to **DenseNet121** (transfer learning, pretrained on ImageNet), expands/augments the training data, and adds an explainability layer plus a significantly richer web application.

| | Phase I — 7th Sem (Baseline) | Phase II — 8th Sem (Current) |
|---|---|---|
| Backbone | Custom 5-block CNN, trained from scratch | DenseNet121, pretrained on ImageNet, transfer learning |
| Input size | 120 × 120 × 3 | 224 × 224 × 3 |
| Explainability | None | Grad-CAM heatmap |
| Deployment | Basic single-image Streamlit page | Multi-tab app: single/batch analysis, history & analytics, model details, about |

---

## ✨ Features

- 🧠 **DenseNet121 Transfer Learning** — ImageNet-pretrained backbone fine-tuned for binary pneumonia classification
- 🔥 **Grad-CAM Explainability** — visual heatmap overlay showing which lung regions influenced each prediction
- 🎚️ **Adjustable Decision Threshold** — tune the probability cutoff used to classify Pneumonia vs Normal
- 🟢🟡🔴 **Confidence Scoring** — every prediction is tagged Low / Moderate / High confidence
- 📦 **Batch Analysis** — upload and analyze multiple X-rays at once, with CSV export
- 📊 **History & Analytics Dashboard** — logs every prediction with timestamped history, distribution charts, and CSV export
- 🌓 **Light/Dark Mode** — accessible, theme-aware UI built with custom CSS
- ⚡ **Lightweight Deployment** — runs as a Streamlit app, no GPU required for inference

---

## 🏗️ Model Architecture

```
Input (224×224×3)
      │
      ▼
DenseNet121 backbone (ImageNet-pretrained, frozen)
  4 Dense Blocks — 6, 12, 24, 16 layers
      │
      ▼
GlobalAveragePooling2D            → (1024,)
      │
      ▼
Dense(256, activation="relu")
      │
      ▼
BatchNormalization
      │
      ▼
Dropout(0.5)
      │
      ▼
Dense(1, activation="sigmoid")    → P(Pneumonia)
```

**Training configuration**

| Parameter | Value |
|---|---|
| Optimizer | Adam (learning rate = 0.0001) |
| Loss | Binary Cross-Entropy |
| Metrics | Accuracy, Precision, Recall, AUC |
| Batch size | 8 |
| Epochs | Up to 20, with `EarlyStopping` (patience = 3, restore best weights) |
| Augmentation | Rotation ±10°, Zoom 0.1, Horizontal Flip (training set only) |
| Grad-CAM layer | `conv5_block16_concat` (final DenseNet121 dense-block output, 7×7 feature map) |

---

## 📂 Dataset

[Kaggle — Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

Real clinical chest X-rays, organized into `train/`, `val/`, and `test/` splits, each containing `NORMAL/` and `PNEUMONIA/` subfolders. The dataset exhibits class imbalance (more pneumonia images than normal), which is mitigated using training-time data augmentation.

```
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

---

## 🗂️ Project Structure

```
.
├── app.py                       # Streamlit web application (PneumoScan AI)
├── pneumonialModelv2_export.py  # Model training / export script (DenseNet121)
├── cnn_latest_model.keras       # Trained model weights (not tracked in git — see below)
├── prediction_history.csv       # Auto-generated prediction log (created at runtime)
├── requirements.txt
└── README.md
```

> ⚠️ The trained model file (`cnn_latest_model.keras`) is not included in this repository due to size. See [Setup](#-setup--installation) below for how to train it yourself, or download a pretrained copy if provided separately.

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/basudev77/PneumoScan-AI.git
cd PneumoScan-AI
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the model (optional — skip if you already have `cnn_latest_model.keras`)
Download the [Kaggle Chest X-Ray dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) as `chest_xray.zip`, place it in the project root, then run:
```bash
python pneumonialModelv2_export.py
```
This extracts the dataset, trains the DenseNet121-based model, evaluates it on the test set, and saves the trained model as `cnn_latest_model.keras`.

### 5. Run the web app
```bash
streamlit run app.py
```
The app will open at `http://localhost:8501`.

---

## 🖥️ Usage

1. **X-Ray Analysis** — upload a single chest X-ray (JPG/JPEG/PNG), click *Run Analysis*, and view the prediction, probability bars, confidence badge, and Grad-CAM overlay.
2. **Batch Analysis** — upload multiple X-rays at once for bulk screening; export results as CSV.
3. **History & Analytics** — review all past predictions with charts and export the full log.
4. **Model Details** — view the live model architecture summary and dataset information.
5. **About** — project overview, objectives, limitations, and team details.

Use the sidebar to toggle **light/dark mode**, adjust the **decision threshold**, and configure **Grad-CAM display options**.

---

## 📈 Results

Final evaluation on the held-out test set (`model.evaluate(data_test)`):

| Metric | Phase I — Baseline CNN | Phase II — DenseNet121 |
|---|---|---|
| Accuracy | 0.9827 | **0.8667** |
| Precision | 0.9828 | **0.8273** |
| Recall | 0.9827 | **0.8938** |
| F1-Score | 0.9827 | **0.8593** |

> F1-Score computed as the harmonic mean of Precision and Recall: 2 × (0.8273 × 0.8938) / (0.8273 + 0.8938) ≈ **0.8593**.

**Note:** The Phase II DenseNet121 model shows a notably higher Recall (0.8938) than Precision (0.8273), meaning it is somewhat better at catching actual pneumonia cases (fewer false negatives) than at avoiding false alarms — a reasonable trade-off for a medical screening tool, though it also means its raw accuracy is lower than the Phase I baseline's reported 0.9827. This is likely influenced by dataset class imbalance and the backbone currently being trained with the DenseNet121 layers frozen (see [Future Work](#-future-work) — fine-tuning deeper layers is expected to improve this further).

---

## 🛠️ Tech Stack

`Python 3` · `TensorFlow / Keras` · `DenseNet121` · `OpenCV` · `Streamlit` · `NumPy` · `Pandas` · `Matplotlib` · `Pillow`

---

## 🚀 Future Work

- **Multi-disease detection** — extend beyond binary Normal/Pneumonia to also flag COVID-19, tuberculosis, and pleural effusion.
- **Backbone benchmarking** — compare DenseNet121 against EfficientNet and ResNet50 to confirm the strongest architecture for this task.
- **Fine-tuning deeper DenseNet121 layers** — currently the backbone is frozen (only the classification head is trained); unfreezing later layers at a low learning rate is expected to improve accuracy/precision further.
- **DICOM support** — accept native hospital X-ray formats (`.dcm`) instead of requiring JPG/PNG conversion.
- **REST API backend** — expose the model via an API for integration with external systems (e.g., hospital record software).
- **Confidence calibration** — apply techniques like temperature scaling so the reported probability better reflects true reliability.

---

## ⚠️ Disclaimer

This project is developed **for academic and research purposes only**. It is **not a certified medical device** and must **not** be used as a substitute for professional medical diagnosis. Always consult a qualified healthcare provider for medical decisions.

---

## 👥 Team

| Name | University Roll No. |
|---|---|
| Sandipan Dutta | 27630822001 |
| Rehan Hezazi | 27630822008 |
| Basudev Das | 27630822018 |
| Rishi Raj | 27630822023 |
| Dhriti Sundar Manik | 27630822027 |

**Department of Computer Science & Engineering (AI)**
Budge Budge Institute of Technology, Kolkata
Under the guidance of **Prof. Subhadeep Majumdar**

---

## 📄 License

This project is submitted as part of academic coursework. Add a license (e.g., MIT) here if you intend to open-source it.

---

## 🙏 Acknowledgements

- [Kaggle — Chest X-Ray Images (Pneumonia) Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- Rajpurkar et al., *CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning*
- Huang et al., *Densely Connected Convolutional Networks (DenseNet)*
- Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*