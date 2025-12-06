import os
import io
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="PneumoScan - Pneumonia Detection",
    page_icon="🩻",
    layout="wide",
)

# -------------------- PERSISTENT HISTORY FILE --------------------
HISTORY_PATH = "prediction_history.csv"


# -------------------- HELPER: LOAD HISTORY FROM CSV --------------------
def load_history_df():
    if os.path.exists(HISTORY_PATH):
        try:
            df = pd.read_csv(HISTORY_PATH)
        except Exception:
            df = pd.DataFrame(
                columns=[
                    "timestamp",
                    "filename",
                    "pneumonia_prob",
                    "normal_prob",
                    "threshold",
                    "prediction",
                ]
            )
    else:
        df = pd.DataFrame(
            columns=[
                "timestamp",
                "filename",
                "pneumonia_prob",
                "normal_prob",
                "threshold",
                "prediction",
            ]
        )
    return df


# -------------------- HELPER: APPEND SINGLE ENTRY TO CSV --------------------
def append_history_entry(entry: dict):
    df = load_history_df()
    df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    df.to_csv(HISTORY_PATH, index=False)


# -------------------- CACHED MODEL LOADING --------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("cnn_model.keras")
    return model


model = load_model()


# -------------------- IMAGE PREPROCESSING --------------------
def preprocess_image(image: np.ndarray, target_size=(120, 120)):
    """
    image: RGB numpy array
    returns: preprocessed array ready for model (1, H, W, C)
    """
    img_resized = cv2.resize(image, target_size)
    img_arr = img_resized.astype("float32") / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)
    return img_resized, img_arr


# -------------------- PREDICTION FUNCTION --------------------
def predict_pneumonia(img_array):
    """
    img_array: (1, H, W, C)
    returns: probability of pneumonia (float)
    """
    pred = model.predict(img_array, verbose=0)
    prob = float(pred[0][0])  # assuming single output neuron with sigmoid
    return prob


# -------------------- SIDEBAR --------------------
st.sidebar.title("🩻 PneumoScan")
st.sidebar.markdown("Deep learning based **Pneumonia Detection** from chest X-ray.")

st.sidebar.markdown("---")
threshold = st.sidebar.slider(
    "Decision threshold (for Pneumonia)",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.01,
    help="If probability ≥ threshold ⇒ Pneumonia, else Normal",
)

show_preprocessed = st.sidebar.checkbox(
    "Show preprocessed (120×120) image", value=True
)

st.sidebar.markdown("---")
st.sidebar.markdown("👨‍💻 **Project Type**: Final Year (CSE - AI)\n\n📚 Model: CNN")

st.sidebar.info(
    "⚠️ **Disclaimer**: This tool is for educational & research purposes only and "
    "must **not** be used for real medical diagnosis."
)


# -------------------- HEADER --------------------
st.title("🩻 PneumoScan – Pneumonia Detection from Chest X-ray")
st.caption(
    "Upload a chest X-ray image. The model predicts the probability of **Pneumonia vs Normal**.\n"
)

# -------------------- TABS --------------------
tab_predict, tab_history, tab_model, tab_about = st.tabs(
    ["🧠 X-ray Analysis", "📊 Prediction History", "📈 Model & Dataset", "ℹ️ About Project"]
)

# -------------------- TAB 1: PREDICTION --------------------
with tab_predict:
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("1️⃣ Upload & Analyze")

        uploaded_file = st.file_uploader(
            "Choose an X-ray image (JPG / JPEG / PNG)",
            type=["jpg", "jpeg", "png"],
            help="Upload a frontal chest X-ray image.",
        )

        analyze_clicked = st.button(
            "🔍 Run Analysis",
            use_container_width=True,
            help="Click after uploading an image to run the model.",
        )

        img_placeholder = st.empty()
        preprocessed_placeholder = st.empty()

    with col_right:
        st.subheader("2️⃣ Result")
        result_placeholder = st.empty()
        prob_placeholder = st.empty()
        threshold_info_placeholder = st.empty()

    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file).convert("RGB")
        img_rgb = np.array(pil_image)

        img_placeholder.image(
            img_rgb,
            caption="Original Uploaded X-ray",
            use_container_width=True,
        )

        img_resized, img_array = preprocess_image(img_rgb, target_size=(120, 120))

        if show_preprocessed:
            preprocessed_placeholder.markdown("**Preprocessed (120×120) input to model:**")
            preprocessed_placeholder.image(
                img_resized,
                caption="Preprocessed Image",
                width=200,
            )

        if analyze_clicked:
            with st.spinner("Running model prediction..."):
                pneu_prob = predict_pneumonia(img_array)
                normal_prob = 1.0 - pneu_prob

            is_pneumonia = pneu_prob >= threshold
            label = "Pneumonia" if is_pneumonia else "Normal"

            # ---------- SAVE TO PERSISTENT HISTORY (CSV ONLY) ----------
            new_entry = {
                "timestamp": datetime.now().isoformat(sep=" ", timespec="seconds"),
                "filename": uploaded_file.name,
                "pneumonia_prob": round(pneu_prob * 100, 2),
                "normal_prob": round(normal_prob * 100, 2),
                "threshold": threshold,
                "prediction": label,
            }
            append_history_entry(new_entry)
            # -----------------------------------------------------------
            if is_pneumonia:
                result_placeholder.error(
                    f"🫁 **Pneumonia likely**\n\n"
                    f"Model confidence (Pneumonia): **{pneu_prob*100:.2f}%**"
                )
            else:
                result_placeholder.success(
                    f"✅ **Normal likely**\n\n"
                    f"Model confidence (Normal): **{normal_prob*100:.2f}%**"
                )

            # Probabilities
            with prob_placeholder:
                st.markdown("#### Probability Breakdown")
                prog_col1, prog_col2 = st.columns(2)
                with prog_col1:
                    st.write(f"**Pneumonia:** {pneu_prob*100:.2f}%")
                    st.progress(pneu_prob)
                with prog_col2:
                    st.write(f"**Normal:** {normal_prob*100:.2f}%")
                    st.progress(normal_prob)

            threshold_info_placeholder.caption(
                f"Decision threshold: **{threshold:.2f}**  (≥ threshold ⇒ Pneumonia)"
            )

    else:
        with col_right:
            result_placeholder.info("Upload a chest X-ray on the left, then click **Run Analysis**.")


# -------------------- TAB 2: HISTORY --------------------
with tab_history:
    st.subheader("📊 Prediction History")

    df_history = load_history_df()

    if df_history.empty:
        st.info("No predictions yet. Run at least one X-ray analysis first.")
    else:
        ctrl_col1, ctrl_col2 = st.columns([1, 1])

        with ctrl_col1:
            csv = df_history.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download history as CSV",
                data=csv,
                file_name="pneumoscan_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with ctrl_col2:
            if st.button("🗑️ Clear all history", use_container_width=True):

                empty_df = pd.DataFrame(columns=df_history.columns)
                try:
                    empty_df.to_csv(HISTORY_PATH, index=False)
                    st.success("History cleared.")
                except Exception as e:
                    st.error(f"Could not clear CSV file: {e}")

                st.rerun()

        st.markdown("### Records")

        # -------- Scrollable, full-width, clean + dark-mode table ----------
        table_html = df_history.to_html(
            index=False,
            justify="center",
            classes="fullwidth-table"
        )

        st.markdown(
            """
            <style>
                .history-table-wrapper {
                    width: 100%;
                    margin: 0;
                    padding: 0;
                }

                .history-scroll-container {
                    max-height: 350px;
                    overflow-y: auto;
                    border: 1px solid var(--border-color, #5a5a5a);
                    border-radius: 8px;
                    padding: 0;
                    margin: 0;
                    background-color: var(--background-color, #ffffff);
                    width: 100%;
                    box-sizing: border-box;
                }

                /* Table full width */
                .fullwidth-table {
                    width: 100% !important;
                    border-collapse: collapse;
                    table-layout: fixed;
                    font-size: 0.9rem;
                    color: var(--text-color, #e6e6e6);
                }

                /* Header styling */
                .fullwidth-table th {
                    background-color: var(--header-bg, #2b2b2b);
                    color: var(--header-text, #ffffff);
                    padding: 8px;
                    text-align: center;
                    border-bottom: 1px solid var(--border-color, #666);
                    position: sticky;
                    top: 0;
                    z-index: 1;
                    white-space: nowrap;
                }

                /* Table cell styling */
                .fullwidth-table td {
                    padding: 8px;
                    text-align: center;
                    border-bottom: 1px solid var(--border-color, #444);
                    color: var(--text-color, #e6e6e6);
                    word-wrap: break-word;
                }

                /* Hover effect */
                .fullwidth-table tr:hover td {
                    background-color: var(--hover-bg, #333333);
                }

                /* LIGHT MODE VARIABLES */
                @media (prefers-color-scheme: light) {
                    .history-scroll-container {
                        border-color: #ddd;
                        background-color: #ffffff;
                    }
                    .fullwidth-table th {
                        background-color: #f2f2f2;
                        color: #000;
                        border-bottom: 1px solid #ddd;
                    }
                    .fullwidth-table td {
                        color: #000;
                        border-bottom: 1px solid #eee;
                    }
                    .fullwidth-table tr:hover td {
                        background-color: #fafafa;
                    }
                }

                /* DARK MODE VARIABLES */
                @media (prefers-color-scheme: dark) {
                    .history-scroll-container {
                        border-color: #444;
                        background-color: #1e1e1e;
                    }
                    .fullwidth-table th {
                        background-color: #333;
                        color: #fff;
                        border-bottom: 1px solid #555;
                    }
                    .fullwidth-table td {
                        color: #ddd;
                        border-bottom: 1px solid #444;
                    }
                    .fullwidth-table tr:hover td {
                        background-color: #2a2a2a;
                    }
                }
            </style>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="history-table-wrapper">
                <div class="history-scroll-container">
                    {table_html}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        # --------------------------------------------------------

        # ---- Summary Metrics ----
        st.markdown("### Summary")

        total = len(df_history)
        pneu_count = (df_history["prediction"] == "Pneumonia").sum()
        normal_count = (df_history["prediction"] == "Normal").sum()

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Total Predictions", total)
        col_b.metric("Predicted Pneumonia", pneu_count)
        col_c.metric("Predicted Normal", normal_count)


# -------------------- TAB 3: MODEL & DATASET --------------------
with tab_model:
    st.subheader("📈 Model & Dataset Details")

    col1, col2 = st.columns(2)

    # ---------------- MODEL DETAILS ----------------
    with col1:
        st.markdown("### 🧠 Model Architecture (Custom CNN)")
        st.markdown(
            """
            This project uses a **custom Convolutional Neural Network (CNN)** for binary
            classification of chest X-ray images into **Normal** and **Pneumonia**.

            #### ⭐ Key Model Details
            - **Task:** Pneumonia detection from chest X-rays  
            - **Input size:** 120 × 120 × 3 (RGB)  
            - **Type:** Binary image classification  
            - **Output:** Sigmoid neuron → `P(Pneumonia)`  
            - **Loss:** Binary Cross-Entropy  
            - **Optimizer:** Adam (learning rate = 0.001)  
            - **Epochs trained:** 10 (with EarlyStopping)  
            - **Trainable parameters:** 1,700,161  

            #### 🏗️ CNN Architecture Overview
            - **Conv Block 1:** Conv2D(32) → MaxPooling  
            - **Conv Block 2:** Conv2D(64) → MaxPooling  
            - **Conv Block 3:** Conv2D(128) → MaxPooling  
            - **Conv Block 4:** Conv2D(256) → MaxPooling  
            - **Conv Block 5:** Conv2D(512) → MaxPooling  

            - **Fully Connected Layers:**  
              - Flatten  
              - Dense(256)  
              - Dense(1, sigmoid)

            #### 📌 How the Model Works
            The final sigmoid layer outputs a **continuous probability (0–1)**:
            - If probability ≥ threshold → **Pneumonia**  
            - If probability < threshold → **Normal**  

            Even though the output is continuous, applying the threshold makes this a  
            **binary classification model**.
            """
        )

        # ---------------- DATASET DETAILS ----------------
        st.markdown("### 📁 Dataset Used (Kaggle)")
        st.markdown(
            """
            The model is trained on the public **Chest X-Ray Images (Pneumonia)** dataset
            from Kaggle — a widely used benchmark in medical imaging research.

            #### 📂 Dataset Directory Structure
            ```
            chest_xray/
                ├── train/
                │     ├── NORMAL/
                │     └── PNEUMONIA/
                ├── val/
                │     ├── NORMAL/
                │     └── PNEUMONIA/
                └── test/
                      ├── NORMAL/
                      └── PNEUMONIA/
            ```

            #### 🎯 Why This Dataset?
            - Contains real clinical chest X-ray images  
            - Clear separation between Normal and Pneumonia classes  
            - Ideal for training CNN-based disease classification models  
            - Standard benchmark for research and performance comparison  
            """
        )

    # ---------------- MODEL SUMMARY PANEL ----------------
    with col2:
        st.markdown("### 🧾 Model Summary")
        
        summary_buf = io.StringIO()
        model.summary(print_fn=lambda x: summary_buf.write(x + "\n"))
        st.text(summary_buf.getvalue())

        st.markdown("---")


# -------------------- TAB 4: ABOUT --------------------
with tab_about:
    st.subheader("ℹ️ About This Project")

    st.markdown("""
    ### Project Title  
    **Computer Vision and Deep Learning for Early Disease Diagnosis and Prediction**  
    """)

    st.markdown("### Problem Statement")
    st.markdown("""
    Pneumonia is a major respiratory disease that requires rapid detection.  
    Manual X-ray interpretation is time-consuming and prone to human error.  
    The aim is to develop an AI-based system that can assist in early and  
    reliable identification of pneumonia from chest X-ray images.
    """)

    st.markdown("### Motivation")
    st.markdown("""
    - Pneumonia affects millions annually, especially children and the elderly.  
    - Early diagnosis reduces mortality and improves treatment outcomes.  
    - Deep learning provides automated, fast, and reliable screening support.  
    - Helps reduce radiologist workload in high-volume healthcare settings.
    """)

    st.markdown("### Objective")
    st.markdown("""
    - Build a **CNN-based automated pneumonia detection system**.  
    - Train model using the **Kaggle Chest X-ray Pneumonia Dataset**.  
    - Provide an easy-to-use **Streamlit interface** for real-time prediction.  
    - Assist healthcare professionals in faster screening (non-clinical use).  
    """)

    st.markdown("### Scope of Project")
    st.markdown("""
    - Detect pneumonia from chest X-ray images (binary classification).  
    - Provide interpretable probability-based outputs.  
    - Maintain **prediction history** for analysis.  
    - Web-based interface makes the model easy to access.  
    """)

    st.markdown("### Technical Overview")
    st.markdown("""
    - **Domain:** Computer Vision, Deep Learning  
    - **Model:** Custom CNN (5 Conv blocks + Dense layers)  
    - **Loss:** Binary Crossentropy  
    - **Epochs Trained:** 10  
    - **Optimizer:** Adam (LR = 0.001)  
    - **Dataset:** Kaggle Chest X-Ray Pneumonia  
    - **Preprocessing:** Resize (120×120), Normalize [0–1]
    """)

    st.markdown("### Limitations")
    st.markdown("""
    - Cannot be used for clinical diagnosis.  
    - Dataset imbalance may affect generalization.  
    - No localization (heatmap) of infected areas.  
    - Model performance may drop on unseen hospital X-ray formats.
    """)

    st.markdown("### Future Enhancements")
    st.markdown("""
    - Grad-CAM heatmaps for explainability  
    - Multi-disease detection (COVID-19, Tuberculosis, etc.)  
    - Advanced models (ResNet, DenseNet, CheXNet)  
    - Cloud deployment with API backend  
    """)

    st.markdown("---")
    st.markdown("""
    ### Team & Academic Details  
    - **Institute:** Budge Budge Institute of Technology (BBIT), Kolkata  
    - **Department:** CSE (AI)  
    - **Course:** PROJ-AI781 – Project 1  
    - **Project Group:** Group 2  

    **Team Members**  
    - Basudev Das – 27630822018  
    - Rishi Raj – 27630822023  
    - Rehan Hejazi – 27630822008  
    - Sandipan Dutta – 27630822001  
    - Dhriti Sundar Manik – 27630822027  

    **Supervisor:** Prof. Subhadeep Mazumdar  
    """)
