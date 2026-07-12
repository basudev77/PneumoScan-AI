"""
PneumoScan AI — v2.1 (8th Semester · Project II · PROJ-AI881 · Project No. 60)
Clean, professional medical UI with light/dark mode toggle.

7th-sem (Phase I) baseline: custom shallow CNN trained from scratch (120x120 input).
8th-sem (Phase II, this version): classification backbone upgraded to a
DenseNet121 transfer-learning model (ImageNet-pretrained, 224x224 input) with
an expanded/augmented training set. Grad-CAM now reads from DenseNet121's
final dense-block output (conv5_block16_concat).

Removed: image filters, layer name display, PDF download (broken).
Added: light/dark toggle, cleaner layout, non-AI-aesthetic design.
"""

import os
import io
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="PneumoScan AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme toggle (must come before CSS injection) ──────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

# ── CSS ────────────────────────────────────────────────────
def inject_css(dark: bool):
    if dark:
        bg        = "#0b0f1a"
        bg2       = "#111827"
        bg3       = "#1a2235"
        border    = "rgba(148,163,184,0.1)"
        border2   = "rgba(148,163,184,0.18)"
        text      = "#f1f5f9"
        text2     = "#94a3b8"
        text3     = "#64748b"
        accent    = "#3b82f6"
        accent2   = "#0ea5e9"
        sidebar_bg= "#0d1424"
        card_bg   = "#131c2e"
        input_bg  = "#0d1424"
        code_bg   = "#0d1424"
        tab_sel   = "#1d4ed8"
    else:
        bg        = "#f8fafc"
        bg2       = "#ffffff"
        bg3       = "#f1f5f9"
        border    = "rgba(15,23,42,0.08)"
        border2   = "rgba(15,23,42,0.14)"
        text      = "#0f172a"
        text2     = "#475569"
        text3     = "#94a3b8"
        accent    = "#2563eb"
        accent2   = "#0284c7"
        sidebar_bg= "#f0f4f8"
        card_bg   = "#ffffff"
        input_bg  = "#ffffff"
        code_bg   = "#f1f5f9"
        tab_sel   = "#1d4ed8"

    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

*, *::before, *::after {{ box-sizing: border-box; }}
html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif !important;
    color: {text} !important;
}}
[data-testid="stAppViewContainer"], [data-testid="stApp"], .main {{
    color: {text} !important;
    font-family: 'DM Sans', sans-serif !important;
}}
.stApp {{ background: {bg} !important; }}
.block-container {{ padding: 1.8rem 2.4rem 4rem !important; max-width: 1400px !important; }}
#MainMenu, footer, header {{ visibility: hidden; }}

/* ─ Sidebar ─ */
section[data-testid="stSidebar"] {{
    background: {sidebar_bg} !important;
    border-right: 1px solid {border} !important;
}}
section[data-testid="stSidebar"] * {{ color: {text} !important; }}
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stCheckbox label {{
    color: {text2} !important;
    font-size: 0.83rem !important;
}}

/* ─ Navbar ─ */
.nav {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 1rem 1.6rem;
    background: {card_bg};
    border: 1px solid {border2};
    border-radius: 14px;
    margin-bottom: 1.8rem;
}}
.nav-left {{ display: flex; align-items: center; gap: 14px; }}
.nav-logo {{ font-size: 2rem; line-height: 1; }}
.nav-name {{ font-size: 1.25rem; font-weight: 700; color: {text}; letter-spacing: -0.4px; }}
.nav-tagline {{ font-size: 0.71rem; color: {text3}; letter-spacing: 0.8px; text-transform: uppercase; margin-top: 1px; }}
.nav-badge {{
    background: {bg3}; border: 1px solid {border2};
    color: {text2}; font-size: 0.71rem; font-weight: 600;
    padding: 4px 11px; border-radius: 20px; letter-spacing: 0.3px;
}}
.nav-badge-blue {{
    background: rgba(59,130,246,0.1); border: 1px solid rgba(59,130,246,0.25);
    color: {accent};
}}

/* ─ Section label ─ */
.sec-label {{
    font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1.5px; color: {text3}; margin-bottom: 0.6rem;
}}

/* ─ Cards ─ */
.card {{
    background: {card_bg};
    border: 1px solid {border};
    border-radius: 12px;
    padding: 1.4rem;
    margin-bottom: 0.9rem;
}}

/* ─ Stat row ─ */
.stat-grid {{ display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; margin-bottom: 1.4rem; }}
.stat-card {{
    background: {card_bg}; border: 1px solid {border};
    border-radius: 12px; padding: 1.1rem; text-align: center;
}}
.stat-num {{ font-size: 1.9rem; font-weight: 700; line-height: 1; margin-bottom: 3px; }}
.stat-desc {{ font-size: 0.68rem; color: {text3}; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }}

/* ─ Result boxes ─ */
.res-pneu {{
    background: rgba(239,68,68,0.06); border: 1px solid rgba(239,68,68,0.28);
    border-left: 3px solid #ef4444;
    border-radius: 10px; padding: 1.1rem 1.4rem; margin: 0.8rem 0;
}}
.res-norm {{
    background: rgba(16,185,129,0.06); border: 1px solid rgba(16,185,129,0.28);
    border-left: 3px solid #10b981;
    border-radius: 10px; padding: 1.1rem 1.4rem; margin: 0.8rem 0;
}}
.res-title {{ font-size: 1.05rem; font-weight: 700; color: {text}; margin: 0 0 4px; }}
.res-sub {{ font-size: 0.8rem; color: {text2}; margin: 0; }}

/* ─ Prob bars ─ */
.pb-row {{ margin: 7px 0; }}
.pb-head {{ display: flex; justify-content: space-between; font-size: 0.79rem; margin-bottom: 4px; }}
.pb-label {{ color: {text2}; font-weight: 500; }}
.pb-val {{ color: {text}; font-weight: 700; font-family: 'DM Mono', monospace; }}
.pb-track {{ background: {bg3}; border-radius: 99px; height: 8px; overflow: hidden; }}
.pb-red   {{ background: #ef4444; border-radius: 99px; height: 100%; transition: width 0.4s ease; }}
.pb-green {{ background: #10b981; border-radius: 99px; height: 100%; transition: width 0.4s ease; }}

/* ─ Badges ─ */
.badge {{
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: 0.69rem; font-weight: 700; letter-spacing: 0.3px;
}}
.b-pneu {{ background: rgba(239,68,68,0.12); color: #ef4444; }}
.b-norm {{ background: rgba(16,185,129,0.12); color: #10b981; }}
.b-hi   {{ background: rgba(16,185,129,0.12); color: #10b981; }}
.b-mod  {{ background: rgba(245,158,11,0.12); color: #f59e0b; }}
.b-low  {{ background: rgba(239,68,68,0.12);  color: #ef4444; }}

/* ─ Heatmap legend ─ */
.hm-leg {{ display:flex; align-items:center; gap:8px; font-size:0.71rem; color:{text3}; margin-top:5px; }}
.hm-bar {{ height:6px; flex:1; border-radius:99px;
           background: linear-gradient(90deg,#00008b,#0000ff,#00ffff,#00ff00,#ffff00,#ff0000); }}

/* ─ History table ─ */
.ht-wrap {{ border:1px solid {border}; border-radius:10px; overflow:hidden; max-height:300px; overflow-y:auto; }}
.ht {{ width:100%; border-collapse:collapse; font-size:0.79rem; }}
.ht thead th {{
    background:{bg3}; color:{text3}; padding:9px 13px;
    text-align:left; font-weight:700; font-size:0.67rem;
    text-transform:uppercase; letter-spacing:0.8px;
    border-bottom:1px solid {border}; position:sticky; top:0;
}}
.ht tbody tr {{ border-bottom:1px solid {border}; }}
.ht tbody tr:hover {{ background:{bg3}; }}
.ht tbody td {{ padding:9px 13px; color:{text2}; }}
.ht tbody td.hi {{ color:{text}; font-weight:600; }}

/* ─ About blocks ─ */
.ab {{ border-left:2px solid {accent}; padding-left:1rem; margin-bottom:1.3rem; }}
.ab h4 {{ color:{accent}; margin:0 0 5px; font-size:0.85rem; font-weight:700; }}
.ab p {{ color:{text2}; font-size:0.82rem; line-height:1.7; margin:0; }}
.team-row {{ display:flex; align-items:center; gap:11px; padding:9px 0; border-bottom:1px solid {border}; }}
.team-av {{
    width:34px; height:34px; border-radius:50%;
    background:{accent}; color:#fff; font-weight:700;
    font-size:0.75rem; display:flex; align-items:center; justify-content:center; flex-shrink:0;
}}
.team-n {{ color:{text}; font-weight:600; font-size:0.82rem; }}
.team-r {{ color:{text3}; font-size:0.7rem; }}
.chip {{
    background:{bg3}; color:{text2}; border:1px solid {border};
    padding:3px 10px; border-radius:20px; font-size:0.7rem;
    font-weight:600; display:inline-block; margin:2px;
}}

/* ─ Tabs ─ */
.stTabs [data-baseweb="tab-list"] {{
    background:{bg3}; border-radius:10px; padding:4px; gap:3px;
    border:1px solid {border}; margin-bottom:1.4rem;
}}
.stTabs [data-baseweb="tab"] {{
    background:transparent !important; color:{text3} !important;
    border-radius:8px !important; font-weight:600 !important;
    font-size:0.82rem !important; padding:7px 16px !important;
}}
.stTabs [aria-selected="true"] {{
    background:{tab_sel} !important; color:#fff !important;
}}

/* ─ Buttons ─ */
.stButton > button {{
    background:{accent} !important; color:#fff !important;
    border:none !important; border-radius:9px !important;
    font-weight:600 !important; font-size:0.84rem !important;
    padding:0.5rem 1.2rem !important;
    transition: opacity 0.15s, transform 0.15s !important;
}}
.stButton > button:hover {{
    opacity:0.88 !important; transform:translateY(-1px) !important;
}}

/* ─ File uploader ─ */
[data-testid="stFileUploader"] {{
    background:{input_bg} !important;
    border:1.5px dashed {border2} !important;
    border-radius:10px !important;
}}

/* ─ Inputs & sliders ─ */
.stSlider [data-baseweb="slider"] {{ padding:0 !important; }}
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {{
    background:{input_bg} !important;
    border:1px solid {border2} !important;
    color:{text} !important;
    border-radius:8px !important;
}}

/* ─ Code block ─ */
.stCodeBlock pre {{
    background:{code_bg} !important;
    font-family: 'DM Mono', monospace !important;
    font-size:0.75rem !important;
}}

/* ─ Divider ─ */
hr {{ border-color:{border} !important; margin:1rem 0 !important; }}

/* ─ Scrollbar ─ */
::-webkit-scrollbar {{ width:4px; height:4px; }}
::-webkit-scrollbar-track {{ background:transparent; }}
::-webkit-scrollbar-thumb {{ background:{border2}; border-radius:99px; }}

/* ─ Metric widget ─ */
[data-testid="stMetric"] {{
    background:{card_bg}; border:1px solid {border};
    border-radius:10px; padding:0.9rem 1rem;
}}
[data-testid="stMetricValue"] {{ color:{text} !important; font-family:'DM Sans',sans-serif !important; }}
[data-testid="stMetricLabel"] {{ color:{text3} !important; font-size:0.72rem !important; }}

/* ─ Selectbox / multiselect ─ */
[data-baseweb="select"] > div {{
    background:{input_bg} !important;
    border:1px solid {border2} !important;
    border-radius:8px !important;
}}

/* ─ FIX: generic st.markdown text (headings, paragraphs, lists) ─
   Modern Streamlit no longer exposes [class*="css"] hooks, so plain
   markdown/caption/code/file-uploader text was falling back to the
   browser default instead of the theme color. Target elements by tag
   (not *) so custom-colored spans/badges/divs are left untouched. */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4,
[data-testid="stMarkdownContainer"] h5,
[data-testid="stMarkdownContainer"] h6,
[data-testid="stMarkdownContainer"] strong,
[data-testid="stMarkdownContainer"] em,
[data-testid="stMarkdownContainer"] td,
[data-testid="stMarkdownContainer"] th,
[data-testid="stMarkdownContainer"] blockquote {{
    color: {text} !important;
}}
[data-testid="stMarkdownContainer"] a {{ color: {accent} !important; }}

/* ─ Captions (st.caption) ─ */
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] * {{
    color: {text2} !important;
    opacity: 1 !important;
}}

/* ─ File uploader — inner text, icon, and Browse button ─ */
[data-testid="stFileUploaderDropzone"] {{
    background: {input_bg} !important;
}}
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] div {{
    color: {text} !important;
}}
[data-testid="stFileUploaderDropzoneInstructions"] small {{
    color: {text3} !important;
}}
[data-testid="stFileUploaderDropzoneInstructions"] svg {{
    fill: {text3} !important;
}}
[data-testid="stFileUploader"] section button,
[data-testid="stBaseButton-secondary"] {{
    background: {bg3} !important;
    color: {text} !important;
    border: 1px solid {border2} !important;
}}
[data-testid="stFileUploaderFile"],
[data-testid="stFileUploaderFileName"] {{
    color: {text} !important;
}}
[data-testid="stFileUploaderFileErrorMessage"] {{
    color: #ef4444 !important;
}}

/* ─ Download button ─ */
[data-testid="stDownloadButton"] > button {{
    background:{accent} !important; color:#fff !important;
    border:none !important; border-radius:9px !important;
    font-weight:600 !important;
}}

/* ─ st.code / model summary block ─ */
[data-testid="stCodeBlock"] pre,
[data-testid="stCodeBlock"] code,
[data-testid="stCodeBlock"] span {{
    color: {text} !important;
    background: transparent !important;
}}
[data-testid="stCodeBlock"] {{
    background: {code_bg} !important;
    border: 1px solid {border} !important;
    border-radius: 8px !important;
}}

/* ─ Alerts (st.info / st.warning / st.error) ─ */
[data-testid="stAlert"] {{
    background: {card_bg} !important;
    border: 1px solid {border2} !important;
}}
[data-testid="stAlert"] p,
[data-testid="stAlert"] span,
[data-testid="stAlert"] div {{
    color: {text} !important;
}}

/* ─ Progress bar caption text ─ */
[data-testid="stProgress"] + div, .stProgress p {{
    color: {text2} !important;
}}
</style>
""", unsafe_allow_html=True)

inject_css(st.session_state.dark_mode)

# ── Constants ──────────────────────────────────────────────
HISTORY_PATH = "prediction_history.csv"
IMG_SIZE      = (224, 224)
# Final dense-block output of the DenseNet121 backbone (7x7 feature map) —
# matches the layer used in the training/export notebook's Grad-CAM routine.
GRADCAM_LAYER = "conv5_block16_concat"
HISTORY_COLS = ["timestamp","filename","pneumonia_prob","normal_prob",
                "threshold","prediction","confidence_level"]

# ── Helpers ────────────────────────────────────────────────
def load_history_df():
    if os.path.exists(HISTORY_PATH):
        try:
            df = pd.read_csv(HISTORY_PATH)
            for c in HISTORY_COLS:
                if c not in df.columns:
                    df[c] = ""
            return df
        except Exception:
            pass
    return pd.DataFrame(columns=HISTORY_COLS)

def append_history_entry(entry: dict):
    df = load_history_df()
    df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    df.to_csv(HISTORY_PATH, index=False)

def confidence_level(prob, threshold):
    d = abs(prob - threshold)
    if d < 0.10: return "Low"
    if d < 0.25: return "Moderate"
    return "High"

def conf_badge(level):
    cls = {"High":"b-hi","Moderate":"b-mod","Low":"b-low"}.get(level,"b-low")
    return f'<span class="badge {cls}">{level} Confidence</span>'

# ── Model ──────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model("cnn_latest_model.keras")
    except Exception as e:
        st.error(f"Model not found: {e}")
        return None

model = load_model()

# ── Preprocessing ──────────────────────────────────────────
def preprocess_image(image: np.ndarray):
    img_resized = cv2.resize(image, IMG_SIZE)
    img_arr = img_resized.astype("float32") / 255.0
    return img_resized, np.expand_dims(img_arr, axis=0)

def predict_pneumonia(img_array):
    pred = model.predict(img_array, verbose=0)
    return float(pred[0][0])

# ── Grad-CAM ───────────────────────────────────────────────
def generate_gradcam(model, img_array, layer_name=GRADCAM_LAYER):
    try:
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(layer_name).output, model.output]
        )
        img_tensor = tf.cast(img_array, tf.float32)
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_tensor)
            tape.watch(conv_outputs)
            loss = predictions[:, 0] if len(predictions.shape) > 1 else predictions[0]
        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            return None, "Gradients returned None"
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_out = conv_outputs[0]
        heatmap = conv_out @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.math.reduce_max(heatmap)
        if float(max_val) == 0:
            return None, "Heatmap is all zeros"
        heatmap = (heatmap / max_val).numpy()
        return heatmap, None
    except Exception as e:
        return None, str(e)

def overlay_gradcam(original_img, heatmap, alpha=0.45):
    h, w = original_img.shape[:2]
    hm_resized = cv2.resize(heatmap, (w, h))
    hm_uint8 = np.uint8(255 * hm_resized)
    hm_colored = cv2.cvtColor(cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    overlay = (1 - alpha) * original_img.astype(np.float32) + alpha * hm_colored.astype(np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    # Logo
    st.markdown(f"""
    <div style="padding:1.2rem 0 1rem; text-align:center;">
        <div style="font-size:2.6rem; line-height:1;">🫁</div>
        <div style="font-size:1.05rem; font-weight:700; margin-top:8px;">PneumoScan AI</div>
        <div style="font-size:0.67rem; letter-spacing:1.5px; text-transform:uppercase; opacity:0.45; margin-top:3px;">
            Chest X-Ray Analysis
        </div>
    </div>
    <hr>
    """, unsafe_allow_html=True)

    # Theme toggle
    st.markdown('<div class="sec-label">Appearance</div>', unsafe_allow_html=True)
    mode_label = "☀️  Switch to Light Mode" if st.session_state.dark_mode else "🌙  Switch to Dark Mode"
    if st.button(mode_label, use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

    st.markdown('<hr>', unsafe_allow_html=True)

    # Detection settings
    st.markdown('<div class="sec-label">Detection Settings</div>', unsafe_allow_html=True)
    threshold = st.slider(
        "Decision Threshold",
        min_value=0.10, max_value=0.90, value=0.50, step=0.01,
        help="Probability ≥ threshold → Pneumonia"
    )

    st.markdown('<div class="sec-label" style="margin-top:1rem;">Display Options</div>', unsafe_allow_html=True)
    show_gradcam     = st.checkbox("Show Grad-CAM heatmap", value=True)
    show_preprocessed = st.checkbox("Show preprocessed image", value=False)
    if show_gradcam:
        gradcam_alpha = st.slider("Heatmap opacity", 0.2, 0.8, 0.45, 0.05)
    else:
        gradcam_alpha = 0.45

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:0.72rem; opacity:0.5; line-height:1.8; text-align:center;">
        BBIT, Kolkata &nbsp;·&nbsp; CSE (AI)<br>
        PROJ-AI881 &nbsp;·&nbsp; Project II &nbsp;·&nbsp; Project No. 60<br>
        Prof. Subhadeep Majumdar
    </div>
    <div style="margin-top:1rem; background:rgba(239,68,68,0.08); border:1px solid rgba(239,68,68,0.2);
                border-radius:8px; padding:9px 11px; font-size:0.72rem; color:#f87171; line-height:1.6;">
        ⚠️ Research &amp; education only.<br>Not for clinical diagnosis.
    </div>
    """, unsafe_allow_html=True)

# ── Top Navbar ─────────────────────────────────────────────
st.markdown(f"""
<div class="nav">
    <div class="nav-left">
        <span class="nav-logo">🫁</span>
        <div>
            <div class="nav-name">PneumoScan AI</div>
            <div class="nav-tagline">Pneumonia Detection from Chest X-Ray</div>
        </div>
    </div>
    <div style="display:flex; gap:8px; align-items:center;">
        <span class="nav-badge nav-badge-blue">DenseNet121 · Transfer Learning</span>
        <span class="nav-badge">Grad-CAM</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🧠 X-Ray Analysis",
    "📦 Batch Analysis",
    "📊 History & Analytics",
    "📈 Model Details",
    "ℹ️ About",
])

# ══════════════════════════════════════════════════════════
#  TAB 1 — SINGLE ANALYSIS
# ══════════════════════════════════════════════════════════
with tab1:
    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown('<div class="sec-label">Step 1 — Upload Image</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Choose a chest X-ray (JPG / JPEG / PNG)",
            type=["jpg","jpeg","png"],
            label_visibility="collapsed"
        )

        if uploaded:
            pil_img  = Image.open(uploaded).convert("RGB")
            orig_rgb = np.array(pil_img)

            st.image(orig_rgb, caption=uploaded.name, use_container_width=True)

            img_resized, img_array = preprocess_image(orig_rgb)
            if show_preprocessed:
                st.image(img_resized, caption=f"Preprocessed ({IMG_SIZE[0]}×{IMG_SIZE[1]})", width=180)

            st.markdown("")
            run_btn = st.button("Run Analysis", use_container_width=True)
        else:
            st.markdown("""
            <div style="text-align:center; padding:3rem 1rem;
                        border:1.5px dashed rgba(148,163,184,0.2);
                        border-radius:10px; margin-top:0.5rem;">
                <div style="font-size:2.8rem; margin-bottom:0.8rem; opacity:0.4;">📤</div>
                <div style="font-size:0.85rem; opacity:0.5;">Upload a chest X-ray image to begin</div>
                <div style="font-size:0.72rem; opacity:0.3; margin-top:4px;">JPG · JPEG · PNG</div>
            </div>
            """, unsafe_allow_html=True)
            run_btn = False

    with col_r:
        st.markdown('<div class="sec-label">Step 2 — Results</div>', unsafe_allow_html=True)

        if uploaded and run_btn and model:
            with st.spinner("Analyzing X-ray..."):
                pneu_prob = predict_pneumonia(img_array)
                norm_prob = 1.0 - pneu_prob
                is_pneu   = pneu_prob >= threshold
                label     = "Pneumonia" if is_pneu else "Normal"
                conf      = confidence_level(pneu_prob, threshold)

                heatmap, hm_err = (generate_gradcam(model, img_array)
                                   if show_gradcam else (None, None))
                heatmap_overlay = (overlay_gradcam(orig_rgb, heatmap, gradcam_alpha)
                                   if heatmap is not None else None)

            # Persist history
            append_history_entry({
                "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "filename":         uploaded.name,
                "pneumonia_prob":   round(pneu_prob * 100, 2),
                "normal_prob":      round(norm_prob * 100, 2),
                "threshold":        threshold,
                "prediction":       label,
                "confidence_level": conf,
            })

            # Result banner
            if is_pneu:
                st.markdown(f"""
                <div class="res-pneu">
                    <p class="res-title">🔴 Pneumonia Detected</p>
                    <p class="res-sub">
                        Confidence: <strong>{pneu_prob*100:.1f}%</strong>
                        &nbsp;·&nbsp; {conf_badge(conf)}
                    </p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="res-norm">
                    <p class="res-title">🟢 Normal — No Signs of Pneumonia</p>
                    <p class="res-sub">
                        Confidence: <strong>{norm_prob*100:.1f}%</strong>
                        &nbsp;·&nbsp; {conf_badge(conf)}
                    </p>
                </div>""", unsafe_allow_html=True)

            # Probability bars
            st.markdown(f"""
            <div class="pb-row">
                <div class="pb-head">
                    <span class="pb-label">Pneumonia</span>
                    <span class="pb-val">{pneu_prob*100:.1f}%</span>
                </div>
                <div class="pb-track">
                    <div class="pb-red" style="width:{pneu_prob*100:.1f}%"></div>
                </div>
            </div>
            <div class="pb-row" style="margin-top:10px;">
                <div class="pb-head">
                    <span class="pb-label">Normal</span>
                    <span class="pb-val">{norm_prob*100:.1f}%</span>
                </div>
                <div class="pb-track">
                    <div class="pb-green" style="width:{norm_prob*100:.1f}%"></div>
                </div>
            </div>
            <div style="font-size:0.71rem; opacity:0.45; margin-top:8px;">
                Threshold: {threshold:.2f} &nbsp;·&nbsp; {'≥ threshold → Pneumonia'}
            </div>
            """, unsafe_allow_html=True)

            # Grad-CAM
            if show_gradcam:
                st.markdown("<hr>", unsafe_allow_html=True)
                if heatmap_overlay is not None:
                    st.markdown('<div class="sec-label">Grad-CAM Activation Map</div>', unsafe_allow_html=True)
                    gc1, gc2 = st.columns(2)
                    with gc1:
                        st.image(orig_rgb, caption="Original", use_container_width=True)
                    with gc2:
                        st.image(heatmap_overlay, caption="Activation Overlay", use_container_width=True)
                    st.markdown("""
                    <div class="hm-leg">
                        <span>Low</span>
                        <div class="hm-bar"></div>
                        <span>High activation</span>
                    </div>
                    <div style="font-size:0.71rem; opacity:0.4; margin-top:5px;">
                        Highlighted regions influenced the model's prediction most strongly.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning(f"Grad-CAM unavailable: {hm_err}")

        elif not uploaded:
            st.markdown("""
            <div style="text-align:center; padding:5rem 1rem; opacity:0.35;">
                <div style="font-size:2.5rem; margin-bottom:0.8rem;">⬅️</div>
                <div style="font-size:0.85rem;">Upload an X-ray and click Run Analysis</div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  TAB 2 — BATCH
# ══════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="sec-label">Batch X-Ray Analysis</div>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.83rem; opacity:0.55; margin-bottom:1rem;">Upload multiple images to analyze all at once.</p>', unsafe_allow_html=True)

    batch_files = st.file_uploader(
        "Upload X-ray images",
        type=["jpg","jpeg","png"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if batch_files:
        if st.button("Analyze All Images", use_container_width=True) and model:
            results = []
            prog    = st.progress(0)
            status  = st.empty()

            for i, bf in enumerate(batch_files):
                status.caption(f"Analyzing {bf.name}  ({i+1}/{len(batch_files)})")
                arr = np.array(Image.open(bf).convert("RGB"))
                _, proc = preprocess_image(arr)
                prob = predict_pneumonia(proc)
                lbl  = "Pneumonia" if prob >= threshold else "Normal"
                conf_b = confidence_level(prob, threshold)
                results.append({
                    "filename":      bf.name,
                    "pneumonia_prob": round(prob*100, 2),
                    "normal_prob":    round((1-prob)*100, 2),
                    "prediction":     lbl,
                    "confidence":     conf_b,
                    "_img":           arr,
                })
                append_history_entry({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "filename": bf.name,
                    "pneumonia_prob": round(prob*100, 2),
                    "normal_prob":    round((1-prob)*100, 2),
                    "threshold":      threshold,
                    "prediction":     lbl,
                    "confidence_level": conf_b,
                })
                prog.progress((i+1) / len(batch_files))

            status.empty(); prog.empty()
            total_b = len(results)
            pneu_b  = sum(1 for r in results if r["prediction"] == "Pneumonia")

            st.markdown(f"""
            <div class="stat-grid">
                <div class="stat-card">
                    <div class="stat-num" style="color:#3b82f6">{total_b}</div>
                    <div class="stat-desc">Total Scanned</div>
                </div>
                <div class="stat-card">
                    <div class="stat-num" style="color:#ef4444">{pneu_b}</div>
                    <div class="stat-desc">Pneumonia</div>
                </div>
                <div class="stat-card">
                    <div class="stat-num" style="color:#10b981">{total_b - pneu_b}</div>
                    <div class="stat-desc">Normal</div>
                </div>
                <div class="stat-card">
                    <div class="stat-num" style="color:#f59e0b">{round(pneu_b/total_b*100)}%</div>
                    <div class="stat-desc">Detection Rate</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            cols_b = st.columns(min(3, total_b))
            for idx, res in enumerate(results):
                with cols_b[idx % 3]:
                    st.image(res["_img"], use_container_width=True)
                    bc   = "b-pneu" if res["prediction"] == "Pneumonia" else "b-norm"
                    icon = "🔴" if res["prediction"] == "Pneumonia" else "🟢"
                    st.markdown(f"""
                    <div style="padding:8px 0 12px;">
                        <div style="font-size:0.77rem; font-weight:600; margin-bottom:5px;
                                    overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">
                            {res['filename']}
                        </div>
                        <span class="badge {bc}">{icon} {res['prediction']}</span>
                        <span style="font-size:0.71rem; opacity:0.5; margin-left:6px;">{res['confidence']}</span>
                        <div style="font-size:0.71rem; opacity:0.4; margin-top:3px;">
                            P(Pneumonia): {res['pneumonia_prob']}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("")
            batch_df = pd.DataFrame([{k: v for k, v in r.items() if k != "_img"} for r in results])
            st.download_button(
                "Download Results as CSV",
                batch_df.to_csv(index=False).encode(),
                f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
            )
    else:
        st.markdown("""
        <div style="text-align:center; padding:2.5rem;
                    border:1.5px dashed rgba(148,163,184,0.18);
                    border-radius:10px; opacity:0.45;">
            <div style="font-size:2.5rem; margin-bottom:0.8rem;">📦</div>
            <div style="font-size:0.85rem;">Upload multiple X-ray images above</div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  TAB 3 — HISTORY & ANALYTICS
# ══════════════════════════════════════════════════════════
with tab3:
    df_h = load_history_df()
    st.markdown('<div class="sec-label">Prediction History</div>', unsafe_allow_html=True)

    if df_h.empty:
        st.info("No predictions yet. Run at least one analysis first.")
    else:
        total_h = len(df_h)
        pneu_h  = (df_h["prediction"] == "Pneumonia").sum()
        avg_p   = df_h["pneumonia_prob"].mean()

        # Stat row
        st.markdown(f"""
        <div class="stat-grid">
            <div class="stat-card">
                <div class="stat-num" style="color:#3b82f6">{total_h}</div>
                <div class="stat-desc">Total Scans</div>
            </div>
            <div class="stat-card">
                <div class="stat-num" style="color:#ef4444">{pneu_h}</div>
                <div class="stat-desc">Pneumonia</div>
            </div>
            <div class="stat-card">
                <div class="stat-num" style="color:#10b981">{total_h - pneu_h}</div>
                <div class="stat-desc">Normal</div>
            </div>
            <div class="stat-card">
                <div class="stat-num" style="color:#f59e0b">{avg_p:.1f}%</div>
                <div class="stat-desc">Avg P(Pneumonia)</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Charts
        is_dark = st.session_state.dark_mode
        fig_bg  = "#131c2e" if is_dark else "#ffffff"
        txt_col = "#f1f5f9" if is_dark else "#0f172a"
        sub_col = "#94a3b8" if is_dark else "#64748b"
        grid_col= "#1a2235" if is_dark else "#f1f5f9"

        ch1, ch2 = st.columns(2)
        with ch1:
            fig, ax = plt.subplots(figsize=(4, 3.5), facecolor=fig_bg)
            ax.set_facecolor(fig_bg)
            wedges, texts, autotexts = ax.pie(
                [pneu_h, total_h - pneu_h],
                labels=["Pneumonia", "Normal"],
                colors=["#ef4444", "#10b981"],
                autopct="%1.1f%%",
                startangle=90,
                wedgeprops={"linewidth": 0},
            )
            for t in texts:     t.set_color(txt_col); t.set_fontsize(9)
            for a in autotexts: a.set_color("#ffffff"); a.set_fontsize(8); a.set_fontweight("bold")
            ax.set_title("Prediction Distribution", color=txt_col, fontsize=10, pad=10, fontweight="600")
            fig.tight_layout()
            st.pyplot(fig); plt.close(fig)

        with ch2:
            fig2, ax2 = plt.subplots(figsize=(5, 3.5), facecolor=fig_bg)
            ax2.set_facecolor(fig_bg)
            ax2.hist(df_h["pneumonia_prob"].values, bins=15,
                     color="#3b82f6", edgecolor=fig_bg, alpha=0.85, rwidth=0.85)
            ax2.axvline(threshold * 100, color="#ef4444", linestyle="--",
                        linewidth=1.5, label=f"Threshold ({threshold*100:.0f}%)")
            ax2.set_title("Probability Distribution", color=txt_col, fontsize=10, fontweight="600")
            ax2.set_xlabel("P(Pneumonia) %", color=sub_col, fontsize=8)
            ax2.set_ylabel("Count",          color=sub_col, fontsize=8)
            ax2.tick_params(colors=sub_col, labelsize=8)
            for s in ax2.spines.values(): s.set_visible(False)
            ax2.yaxis.grid(True, color=grid_col, linewidth=0.6)
            ax2.set_axisbelow(True)
            ax2.legend(fontsize=8, facecolor=fig_bg, labelcolor=txt_col, framealpha=0.5)
            fig2.tight_layout()
            st.pyplot(fig2); plt.close(fig2)

        # Table
        rows = ""
        for _, row in df_h.iloc[::-1].iterrows():
            pred = row.get("prediction", "")
            tag  = (f'<span class="badge b-pneu">🔴 Pneumonia</span>'
                    if pred == "Pneumonia"
                    else f'<span class="badge b-norm">🟢 Normal</span>')
            rows += (
                f'<tr>'
                f'<td>{row.get("timestamp","")}</td>'
                f'<td class="hi">{row.get("filename","")}</td>'
                f'<td>{row.get("pneumonia_prob","")}%</td>'
                f'<td>{row.get("normal_prob","")}%</td>'
                f'<td>{row.get("threshold","")}</td>'
                f'<td>{tag}</td>'
                f'<td>{row.get("confidence_level","—")}</td>'
                f'</tr>'
            )

        st.markdown(f"""
        <div class="ht-wrap">
            <table class="ht">
                <thead><tr>
                    <th>Timestamp</th><th>File</th>
                    <th>P(Pneumonia)</th><th>P(Normal)</th>
                    <th>Threshold</th><th>Result</th><th>Confidence</th>
                </tr></thead>
                <tbody>{rows}</tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")
        hc1, hc2 = st.columns(2)
        with hc1:
            st.download_button(
                "Export History as CSV",
                df_h.to_csv(index=False).encode(),
                "pneumoscan_history.csv",
                "text/csv",
                use_container_width=True,
            )
        with hc2:
            if st.button("Clear All History", use_container_width=True):
                pd.DataFrame(columns=HISTORY_COLS).to_csv(HISTORY_PATH, index=False)
                st.rerun()

# ══════════════════════════════════════════════════════════
#  TAB 4 — MODEL DETAILS
# ══════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="sec-label">Model & Dataset Information</div>', unsafe_allow_html=True)
    m1, m2 = st.columns(2, gap="large")

    with m1:
        st.markdown("#### Architecture — DenseNet121 (Transfer Learning)")
        st.caption("Upgraded in the 8th-semester phase from the custom, shallow CNN used in the 7th-semester baseline.")
        arch_rows = [
            ("Base Model",       "DenseNet121 (ImageNet-pretrained, include_top=False)"),
            ("Backbone Weights", "Frozen during head training (feature extractor)"),
            ("Dense Blocks",     "4 blocks (6, 12, 24, 16 layers) with dense connectivity"),
            ("Global AvgPool",   "GlobalAveragePooling2D → (None, 1024)"),
            ("Dense Head",       "Dense(256, ReLU) → BatchNorm → Dropout(0.5) → Dense(1, Sigmoid)"),
            ("Input Size",       "224 × 224 × 3 RGB"),
            ("Output",           "P(Pneumonia) via Sigmoid"),
            ("Loss",             "Binary Cross-Entropy"),
            ("Optimizer",        "Adam (LR = 0.0001)"),
            ("Metrics Tracked",  "Accuracy, Precision, Recall, AUC"),
            ("Data Augmentation","Rotation ±10°, Zoom 0.1, Horizontal Flip"),
            ("Batch Size",       "8"),
            ("Training",         "Up to 20 epochs with EarlyStopping (patience = 3, restore best weights)"),
            ("Grad-CAM Target",  "conv5_block16_concat (7×7 feature map)"),
        ]
        for k, v in arch_rows:
            st.markdown(
                f"<div style='display:flex; justify-content:space-between; padding:6px 0;"
                f"border-bottom:1px solid rgba(148,163,184,0.08); font-size:0.81rem;'>"
                f"<span style='opacity:0.5;'>{k}</span>"
                f"<span style='font-weight:600;'>{v}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("")
        st.markdown("#### Dataset")
        st.markdown("""
        **Kaggle — Chest X-Ray Images (Pneumonia)**

        Real clinical X-rays organised into train / val / test splits,
        each with NORMAL and PNEUMONIA sub-folders. For this phase, the
        training set is passed through an augmentation pipeline (rotation,
        zoom, horizontal flip) to increase effective sample diversity and
        reduce the impact of class imbalance between the Normal and
        Pneumonia categories. Widely used benchmark for medical imaging
        classification research.
        """)

    with m2:
        st.markdown("#### How Grad-CAM Works")
        st.markdown("""
        `conv5_block16_concat` — the output of DenseNet121's final dense
        block — produces a **7×7 feature map**, where each cell covers roughly
        a 32×32 pixel area of the original 224×224 X-ray.

        Grad-CAM computes the gradient of the output neuron with respect to
        each spatial location in this map, averages the gradients per channel,
        and creates a weighted activation map. This highlights which regions
        most strongly drove the model's decision.

        - 🔴 **Warm/red** — high model attention
        - 🔵 **Cool/blue** — low model attention
        """)

        if model:
            st.markdown("#### Live Model Summary")
            buf = io.StringIO()
            model.summary(print_fn=lambda x: buf.write(x + "\n"))
            st.code(buf.getvalue(), language="text")

# ══════════════════════════════════════════════════════════
#  TAB 5 — ABOUT
# ══════════════════════════════════════════════════════════
with tab5:
    a1, a2 = st.columns([3, 2], gap="large")

    with a1:
        st.markdown('<div class="sec-label">Project Overview</div>', unsafe_allow_html=True)
        sections = {
            "Problem Statement":
                "Computer Vision and Deep Learning for Early Disease Diagnosis and "
                "Prediction.<br><br>"
                "Pneumonia is a leading cause of mortality worldwide, and "
                "manual X-ray interpretation is time-intensive and prone to variability. "
                "This project automates preliminary screening using a DenseNet121-based "
                "transfer-learning model, upgraded in this phase from the shallow custom "
                "CNN used in the 7th-semester baseline.",
            "Motivation":
                "Deep learning provides fast, consistent screening support. This system "
                "reduces diagnostic delays in high-volume settings and serves as an "
                "educational research tool for medical AI and explainable deep learning.",
            "Objective":
                "Fine-tune a DenseNet121 backbone (pretrained on ImageNet) for binary "
                "pneumonia classification, deploy it via an accessible web interface, and "
                "provide interpretable Grad-CAM heatmaps pinpointing regions of interest "
                "in each X-ray.",
            "Key Features":
                "DenseNet121 transfer-learning model (224×224 input) · Grad-CAM heatmap "
                "visualisation · Batch multi-image analysis · Prediction history with "
                "analytics · Confidence level scoring · Light and dark mode · CSV export.",
            "Limitations":
                "Not validated for clinical use. Dataset imbalance may affect rare "
                "presentations. Grad-CAM provides approximate, not pixel-precise, "
                "localisation.",
            "Future Work":
                "Multi-disease detection (COVID-19, TB, pleural effusion) · benchmarking "
                "against EfficientNet/ResNet50 backbones · fine-tuning deeper DenseNet121 "
                "layers · DICOM support · REST API backend · confidence calibration.",
        }
        for title, content in sections.items():
            st.markdown(
                f'<div class="ab"><h4>{title}</h4><p>{content}</p></div>',
                unsafe_allow_html=True,
            )

    with a2:
        # Academic card
        st.markdown("""
        <div class="card">
            <div class="sec-label">Academic Details</div>
            <table style="width:100%; font-size:0.81rem; border-collapse:collapse;">
        """, unsafe_allow_html=True)
        for k, v in [("Institute","BBIT, Kolkata"),("Department","CSE — Artificial Intelligence"),
                      ("Project Code","PROJ-AI881"),("Phase","Project II · 8th Semester"),
                      ("Project No.","60"),
                      ("Supervisor","Prof. Subhadeep Majumdar")]:
            st.markdown(
                f"<tr><td style='padding:5px 0; opacity:0.45; width:45%;'>{k}</td>"
                f"<td style='padding:5px 0; font-weight:600;'>{v}</td></tr>",
                unsafe_allow_html=True,
            )
        st.markdown("</table></div>", unsafe_allow_html=True)

        # Team card
        st.markdown('<div class="card" style="margin-top:0.9rem;"><div class="sec-label">Team Members</div>', unsafe_allow_html=True)
        for initials, name, roll in [
            ("BD", "Basudev Das",         "27630822018"),
            ("RR", "Rishi Raj",           "27630822023"),
            ("RH", "Rehan Hejazi",        "27630822008"),
            ("SD", "Sandipan Dutta",      "27630822001"),
            ("DM", "Dhriti Sundar Manik", "27630822027"),
        ]:
            st.markdown(
                f'<div class="team-row">'
                f'<div class="team-av">{initials}</div>'
                f'<div><div class="team-n">{name}</div>'
                f'<div class="team-r">{roll}</div></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

        # Tech stack
        st.markdown('<div class="card" style="margin-top:0.9rem;"><div class="sec-label">Tech Stack</div><div style="margin-top:4px;">', unsafe_allow_html=True)
        for t in ["TensorFlow 2.x","DenseNet121","Transfer Learning","Streamlit","OpenCV",
                  "Grad-CAM","NumPy","Pandas","Matplotlib","Pillow"]:
            st.markdown(f'<span class="chip">{t}</span>', unsafe_allow_html=True)
        st.markdown("</div></div>", unsafe_allow_html=True)