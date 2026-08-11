"""
GulfBite — Streamlit app
Photo -> dish recognition (3-tier: CNN -> YOLOv8 -> user confirm) -> portion size -> calorie/macro estimate.

BEFORE RUNNING:
  Place these four files (copied from your Google Drive GulfBite-Models/ folder) into a
  local `models/` folder next to this script:
    models/MobileNetV2_best.keras
    models/class_indices.json
    models/yolov8_ingredient_detector-4/weights/best.pt
    models/ingredient_nutrition_cache.json
  Or edit MODELS_DIR below to point elsewhere.

RUN LOCALLY:
    pip install -r requirements.txt
    streamlit run app.py

DEPLOY TO HUGGING FACE SPACES (Streamlit SDK):
    Push this file + requirements.txt + the models/ folder to your Space repo.
    Large model files may need Git LFS — check HF Spaces storage limits before pushing.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import json
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

# ============================================================================
# CONFIG — unchanged from the validated Colab pipeline
# ============================================================================

MODELS_DIR = "models"
CNN_MODEL_PATH = os.path.join(MODELS_DIR, "MobileNetV2_best.keras")
CLASS_INDICES_PATH = os.path.join(MODELS_DIR, "class_indices.json")
YOLO_WEIGHTS_PATH = os.path.join(MODELS_DIR, "yolov8_ingredient_detector-4", "weights", "best.pt")
INGREDIENT_CACHE_PATH = os.path.join(MODELS_DIR, "ingredient_nutrition_cache.json")

TRIGGER_SET = {'07_ouzi', '01_machboos', '09_jisheed', '02_kabsa', '03_biryani', '06_saloona'}
WRAP_TRIGGER_SET = {'10_shawarma', '11_falafel_wrap'}
CONFIDENCE_THRESHOLD = 0.7

YOLO_FEATURE_MAP = {
    '01_machboos':     'loomi',
    '07_ouzi':         'whole_shank',
    '08_samak_mashwi': 'whole_fish',
    '02_kabsa':        'whole_chicken_piece',
    '10_shawarma':     'shawarma_meat',
    '11_falafel_wrap': 'falafel_ball',
}
FEATURE_TO_DISH = {v: k for k, v in YOLO_FEATURE_MAP.items()}

CONFUSION_GROUPS = {
    'rice_cluster': {'01_machboos', '02_kabsa', '03_biryani', '07_ouzi', '09_jisheed', '06_saloona'},
    'wrap_cluster': {'10_shawarma', '11_falafel_wrap', '12_falafel'},
}

GROUP_REASONS = {
    'rice_cluster': "Rice dishes like Machboos, Kabsa, and Biryani can look very similar, so we like to double-check.",
    'wrap_cluster': "Wrapped dishes can hide their filling, so we like to double-check.",
}

def get_group_reason(cnn_class):
    for group_name, group_set in CONFUSION_GROUPS.items():
        if cnn_class in group_set:
            return GROUP_REASONS.get(group_name)
    return None

FEATURE_RELIABILITY = {
    'loomi':               {'status': 'reliable'},
    'whole_chicken_piece': {'status': 'unreliable'},
    'whole_shank':         {'status': 'insufficient_evidence'},
    'whole_fish':          {'status': 'insufficient_evidence'},
    'shawarma_meat':       {'status': 'reliable'},
    'falafel_ball':        {'status': 'reliable'},
}

DISH_RECIPES = {
    '01_machboos':     [('rice', 150), ('chicken', 130), ('olive_oil', 15), ('onion', 20), ('tomato', 15)],
    '02_kabsa':        [('rice', 150), ('chicken', 130), ('olive_oil', 15), ('tomato', 20), ('onion', 15)],
    '03_biryani':      [('rice', 160), ('chicken', 140), ('olive_oil', 15), ('yogurt', 20), ('onion', 20)],
    '04_harees':       [('bulgur', 100), ('lamb', 100), ('ghee', 15)],
    '05_thareed':      [('pita_bread', 80), ('lamb', 120), ('mixed_vegetables', 60)],
    '06_saloona':      [('lamb', 120), ('mixed_vegetables', 100), ('tomato', 40), ('olive_oil', 15)],
    '07_ouzi':         [('rice', 150), ('lamb', 180), ('mixed_nuts', 15), ('olive_oil', 15)],
    '08_samak_mashwi': [('fish', 200), ('olive_oil', 10)],
    '09_jisheed':      [('rice', 150), ('fish', 100), ('olive_oil', 10)],
    '10_shawarma':     [('pita_bread', 80), ('chicken', 100), ('garlic_sauce', 20), ('pickles', 10)],
    '11_falafel_wrap': [('pita_bread', 80), ('falafel', 90), ('tahini', 15), ('mixed_vegetables', 30)],
    '12_falafel':      [('falafel', 120), ('olive_oil', 10)],
    '13_samboosa':     [('pastry_dough', 60), ('ground_meat', 60), ('olive_oil', 10)],
    '14_mutabbaq':     [('pastry_dough', 100), ('ground_meat', 80), ('olive_oil', 15)],
    '15_hummus':       [('chickpeas', 80), ('tahini', 15), ('olive_oil', 10)],
    '16_fattoush':     [('mixed_vegetables', 150), ('pita_bread', 20), ('olive_oil', 10)],
    '17_tabbouleh':    [('parsley', 80), ('bulgur', 20), ('tomato', 30), ('olive_oil', 15)],
    '18_foul_medames': [('fava_beans', 150), ('olive_oil', 15)],
    '19_shakshuka':    [('eggs', 100), ('tomato_sauce', 150), ('olive_oil', 10)],
    '20_balaleet':     [('vermicelli', 80), ('sugar', 15), ('eggs', 50)],
    '21_khameer':      [('bread_wheat', 80)],
    '22_chebab':       [('pancake_batter', 100)],
    '23_luqaimat':     [('fried_dough', 100), ('date_syrup', 30)],
    '24_knafeh':       [('kunafa_dough', 80), ('soft_cheese', 60), ('sugar_syrup', 40), ('ghee', 15)],
    '25_karak_chai':   [('milk', 100), ('black_tea', 100), ('sugar', 10)],
}

DISH_BLURBS = {
    '01_machboos':     "A spiced rice dish with meat or chicken, flavoured with dried lime (loomi) — a Bahraini and Kuwaiti staple.",
    '02_kabsa':        "Saudi Arabia's best-known dish: spiced rice with meat, often finished with saffron and tomato.",
    '03_biryani':      "A layered spiced rice dish with South Asian roots, now a Gulf favourite thanks to centuries of trade.",
    '04_harees':       "A slow-cooked wheat and meat porridge, traditionally eaten during Ramadan and Eid across the Gulf.",
    '05_thareed':      "Bread soaked in a rich meat and vegetable stew — an Emirati dish often said to have been a favourite of the Prophet Muhammad.",
    '06_saloona':      "A everyday spiced stew of meat and vegetables, found in home kitchens across the Gulf.",
    '07_ouzi':         "Whole roasted lamb served over rice, traditionally prepared for celebrations and large gatherings.",
    '08_samak_mashwi': "Grilled fish, simply prepared — a reflection of the Gulf's long fishing heritage.",
    '09_jisheed':      "An Emirati dish of shredded fish mixed with rice.",
    '10_shawarma':     "Spit-roasted meat wrapped in bread — originally Levantine, now a Middle East-wide street food staple.",
    '11_falafel_wrap': "Fried chickpea or fava bean balls in a wrap, a Levantine and Egyptian vegetarian favourite.",
    '12_falafel':      "Deep-fried balls of chickpeas or fava beans, a staple across the Gulf.",
    '13_samboosa':     "A fried or baked pastry with a savoury filling, especially popular during Ramadan.",
    '14_mutabbaq':     "A folded, stuffed pastry with Yemeni roots, filled with either savoury or sweet fillings.",
    '15_hummus':       "A creamy chickpea and tahini dip, a Levantine staple found on tables across the Gulf.",
    '16_fattoush':     "A Levantine bread salad with crisp vegetables and toasted pita, dressed with sumac.",
    '17_tabbouleh':    "A Levantine salad of finely chopped parsley, bulgur, tomato, and lemon.",
    '18_foul_medames': "A stewed fava bean dish of Egyptian origin, a common Gulf breakfast staple.",
    '19_shakshuka':    "Eggs poached in a spiced tomato sauce, of North African and Levantine origin.",
    '20_balaleet':     "Sweet saffron-spiced vermicelli topped with a savoury omelette — a distinctly Emirati breakfast pairing.",
    '21_khameer':      "A traditional Emirati sweet leavened bread, often spiced with cardamom or saffron.",
    '22_chebab':       "An Emirati pancake flavoured with cardamom and saffron, popular at breakfast.",
    '23_luqaimat':     "Sweet fried dough balls drizzled with date syrup, a classic Ramadan and Eid treat across the Gulf.",
    '24_knafeh':       "A cheese pastry soaked in sweet syrup, with roots in the Levant, beloved across the Gulf.",
    '25_karak_chai':   "Spiced milk tea with South Asian influence, now an everyday favourite across the Gulf.",
}

PORTION_MULTIPLIERS = {'S': 0.7, 'M': 1.0, 'L': 1.4}
PORTION_LABELS = {'S': 'Small', 'M': 'Medium', 'L': 'Large'}
CALORIE_RANGE_PCT = 0.15


def display_name(cls: str) -> str:
    """'01_machboos' -> 'Machboos'"""
    return cls.split('_', 1)[1].replace('_', ' ').title()


def get_candidate_group(cnn_class: str) -> set:
    for group in CONFUSION_GROUPS.values():
        if cnn_class in group:
            return group
    return {cnn_class}


# ============================================================================
# MODEL LOADING — cached so this only happens once per session, not per rerun
# ============================================================================

@st.cache_resource
def load_models():
    import tensorflow as tf
    from ultralytics import YOLO

    missing = [p for p in (CNN_MODEL_PATH, CLASS_INDICES_PATH, YOLO_WEIGHTS_PATH, INGREDIENT_CACHE_PATH)
               if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing model file(s):\n" + "\n".join(missing) +
            "\n\nCopy these from your Google Drive GulfBite-Models/ folder into the models/ "
            "directory next to app.py — see the top of app.py for the exact expected layout."
        )

    cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
    with open(CLASS_INDICES_PATH) as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}

    yolo_model = YOLO(YOLO_WEIGHTS_PATH)

    with open(INGREDIENT_CACHE_PATH) as f:
        ingredient_cache = json.load(f)

    return cnn_model, idx_to_class, yolo_model, ingredient_cache


# ============================================================================
# PIPELINE FUNCTIONS — identical logic to the validated Colab version
# ============================================================================

def run_cnn(pil_image, model, idx_to_class, img_size=(224, 224)):
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    img = pil_image.convert('RGB').resize(img_size)
    arr = np.array(img).astype('float32')
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr, verbose=0)[0]
    top_idx = int(np.argmax(preds))
    confidence = float(preds[top_idx])
    predicted_class = idx_to_class[top_idx]
    return predicted_class, confidence


def run_yolov8(pil_image, yolo_model, conf_threshold=0.25):
    results = yolo_model.predict(np.array(pil_image.convert('RGB')), conf=conf_threshold, verbose=False)
    detections = []
    r = results[0]
    for box in r.boxes:
        cls_id = int(box.cls[0])
        cls_name = yolo_model.names[cls_id]
        box_conf = float(box.conf[0])
        detections.append((cls_name, box_conf))
    return detections


def map_detections_to_suggestion(detections, candidates):
    if not detections:
        return None, None, 'no_detection'
    valid = [
        (FEATURE_TO_DISH[feat], conf, feat)
        for feat, conf in detections
        if feat in FEATURE_TO_DISH and FEATURE_TO_DISH[feat] in candidates
    ]
    if not valid:
        return None, None, 'no_detection'
    valid.sort(key=lambda x: x[1], reverse=True)
    dish, conf, feature = valid[0]
    status = FEATURE_RELIABILITY.get(feature, {'status': 'insufficient_evidence'})['status']
    gated = (dish, conf) if status == 'reliable' else None
    return (dish, conf), gated, status


def estimate_nutrition(dish_class, portion_size, ingredient_cache):
    multiplier = PORTION_MULTIPLIERS[portion_size]
    totals = {'calories': 0.0, 'protein': 0.0, 'carbs': 0.0, 'fat': 0.0}
    missing = []
    for ingredient_key, base_grams in DISH_RECIPES[dish_class]:
        info = ingredient_cache.get(ingredient_key)
        if info is None or info.get('source') == 'NONE':
            missing.append(ingredient_key)
            continue
        grams = base_grams * multiplier
        for macro in totals:
            totals[macro] += info[macro] * (grams / 100)
    cal_low = totals['calories'] * (1 - CALORIE_RANGE_PCT)
    cal_high = totals['calories'] * (1 + CALORIE_RANGE_PCT)
    return {
        'calories_range': (round(cal_low), round(cal_high)),
        'protein_g': round(totals['protein'], 1),
        'carbs_g': round(totals['carbs'], 1),
        'fat_g': round(totals['fat'], 1),
        'missing_ingredients': missing,
    }


# ============================================================================
# VISUAL THEME — "Arabian Gulf twilight": deep teal background, saffron accent,
# warm sand text. Fraunces for display type, Inter for body, JetBrains Mono for
# numbers (calories/macros read like a nutrition label, not a UI label).
# ============================================================================

def inject_theme():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');

    :root {
    --gb-bg: #151107;
    --gb-surface: #1F1A0E;
    --gb-surface-2: #2A2313;
    --gb-accent: #C9982E;
    --gb-accent-dim: #A67D22;
    --gb-text: #F0E9D8;
    --gb-muted: #A69874;
    --gb-border: #3A3018;
}

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: var(--gb-bg); }

    #MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
    .block-container { max-width: 620px; padding-top: 2rem; }

    .gb-header { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.15rem; }
    .gb-title {
        font-family: 'Inter', sans-serif; font-weight: 800; font-size: 1.7rem;
        letter-spacing: -0.02em; color: var(--gb-text); margin: 0;
    }
    .gb-subtitle { font-family: 'Inter', sans-serif; color: var(--gb-muted); font-size: 0.92rem; margin: 0 0 1rem 2.65rem; }

    .gb-stepper { display: flex; align-items: center; margin: 0.25rem 0 1.75rem 0; }
    .gb-step { display: flex; align-items: center; gap: 0.5rem; flex-shrink: 0; }
    .gb-step-dot {
        width: 1.5rem; height: 1.5rem; border-radius: 6px; display: flex;
        align-items: center; justify-content: center; font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem; font-weight: 600; border: 1.5px solid var(--gb-border);
        color: var(--gb-muted); background: var(--gb-surface); flex-shrink: 0;
    }
    .gb-step-label { font-size: 0.76rem; color: var(--gb-muted); white-space: nowrap; }
    .gb-step.done .gb-step-dot { background: var(--gb-accent); border-color: var(--gb-accent); color: #FFFFFF; }
    .gb-step.done .gb-step-label { color: var(--gb-text); }
    .gb-step.active .gb-step-dot {
        border-color: var(--gb-accent); color: var(--gb-accent);
        box-shadow: 0 0 0 3px rgba(31,158,110,0.15);
    }
    .gb-step.active .gb-step-label { color: var(--gb-text); font-weight: 600; }
    .gb-step-line { flex: 1; height: 1.5px; background: var(--gb-border); margin: 0 0.5rem; min-width: 0.75rem; }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--gb-surface); border: 1px solid var(--gb-border) !important;
        border-radius: 12px !important; padding: 0.25rem;
    }

    [data-testid="stFileUploaderDropzone"] { background: var(--gb-surface-2); border: 1.5px dashed var(--gb-border); border-radius: 8px; }
    
    [data-testid="stFileUploader"] button {
        border-radius: 8px; border: 1px solid var(--gb-accent); background: transparent;
        color: var(--gb-accent); font-family: 'Inter', sans-serif; font-weight: 600;
    }
    [data-testid="stFileUploader"] button:hover { background: rgba(201,152,46,0.12); }

    div.stButton > button, div.stDownloadButton > button {
        border-radius: 8px; border: 1px solid var(--gb-accent); background: transparent;
        color: var(--gb-accent); font-family: 'Inter', sans-serif; font-weight: 600;
        padding: 0.5rem 1.1rem; transition: all 0.15s ease; width: 100%;
    }
    div.stButton > button:hover { background: rgba(201,152,46,0.12); color: var(--gb-accent); }
    div.stButton > button[kind="primary"] { background: var(--gb-accent); color: #FFFFFF; border-color: var(--gb-accent); }
    div.stButton > button[kind="primary"]:hover { background: var(--gb-accent-dim); }

    [data-testid*="Expander"] {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    box-shadow: none !important;
}
[data-testid="stExpander"] {
    border-top: 1px solid rgba(201,152,46,0.25) !important;
    margin-top: 0.75rem;
    padding-top: 0.25rem;
}
details {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
}
details > summary {
    list-style: none;
}
    [data-testid="stAlert"] { background: var(--gb-surface-2); border-radius: 8px; }

    .gb-dish-name { font-family: 'Inter', sans-serif; font-weight: 700; font-size: 1.4rem; color: var(--gb-text); margin-bottom: 0.15rem; }
    .gb-dish-meta { font-size: 0.85rem; color: var(--gb-muted); margin-bottom: 1rem; }

    .gb-range-numbers {
        display: flex; align-items: baseline; gap: 0.4rem; font-family: 'JetBrains Mono', monospace;
        font-size: 1.9rem; font-weight: 600; color: var(--gb-text); margin-bottom: 0.4rem;
    }
    .gb-range-unit { font-size: 0.85rem; color: var(--gb-muted); font-weight: 500; }
    .gb-range-track { height: 8px; border-radius: 4px; background: var(--gb-surface-2); position: relative; overflow: hidden; }
    .gb-range-fill {
        position: absolute; top: 0; bottom: 0; left: 15%; right: 15%;
        background: var(--gb-accent); border-radius: 4px;
    }
    .gb-range-caption { font-size: 0.76rem; color: var(--gb-muted); margin-top: 0.45rem; margin-bottom: 1.1rem; }

    .gb-stat-grid { display: flex; gap: 0.6rem; margin-top: 1.1rem; }
    .gb-stat { flex: 1; background: var(--gb-surface-2); border-radius: 10px; padding: 0.65rem 0.4rem; text-align: center; }
    .gb-stat-value { font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; font-weight: 600; color: var(--gb-text); }
    .gb-stat-label { font-size: 0.68rem; color: var(--gb-muted); text-transform: uppercase; letter-spacing: 0.04em; margin-top: 0.15rem; }

    /* Subtle background texture — breaks up the flat single-color background */
.stApp {
    background-color: var(--gb-bg);
   background-image: radial-gradient(circle at 1px 1px, rgba(201,152,46,0.06) 1px, transparent 0);
    background-size: 24px 24px;
}

/* Hero number treatment for the calorie figure */
.gb-range-numbers {
    font-size: 3.4rem !important;
    line-height: 1;
    letter-spacing: -0.02em;
    margin-bottom: 0.5rem !important;
}
.gb-range-unit { font-size: 1rem !important; }

/* Photo framing — turns a plain upload into a presented result */
[data-testid="stImage"] {
    border-radius: 4px;
    overflow: hidden;
    box-shadow: 0 10px 28px rgba(0,0,0,0.45);
    border: 2px solid var(--gb-muted);
}
[data-testid="stImage"]::before {
    content: "YOUR PHOTO";
    position: absolute;
    top: 10px; left: 10px;
    background: rgba(11,21,18,0.75);
    color: var(--gb-text);
    font-family: 'Inter', sans-serif;
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    padding: 3px 8px;
    border-radius: 6px;
    z-index: 2;
}

/* Radio options restyled as selectable rows, not plain dots */
[data-testid="stRadio"] > div {
    gap: 0.5rem;
}
[data-testid="stRadio"] label {
    background: var(--gb-surface-2);
    border: 1px solid var(--gb-border);
    border-radius: 10px;
    padding: 0.6rem 0.9rem !important;
    margin: 0 !important;
    transition: border-color 0.15s ease, background 0.15s ease;
    width: 100%;
}
[data-testid="stRadio"] label:hover {
    border-color: var(--gb-accent);
    background: rgba(201,152,46,0.08);
}

/* Soft fade-in on each stage container */
[data-testid="stVerticalBlockBorderWrapper"] {
    animation: gb-fade-in 0.35s ease;
}
@keyframes gb-fade-in {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Guard against body text ever inheriting accent color */
[data-testid="stMarkdownContainer"] p { color: var(--gb-text); }

/* Macro composition bar — replaces the 3-box stat grid */
.gb-macro-bar-track { display: flex; height: 10px; border-radius: 6px; overflow: hidden; margin-top: 1.1rem; background: var(--gb-surface-2); }
.gb-macro-legend { display: flex; justify-content: space-around; margin-top: 0.7rem; }
.gb-macro-item { text-align: center; }
.gb-macro-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 4px; }
.gb-macro-value { font-family: 'JetBrains Mono', monospace; font-weight: 600; font-size: 1rem; color: var(--gb-text); }
.gb-macro-label { font-size: 0.68rem; color: var(--gb-muted); text-transform: uppercase; letter-spacing: 0.04em; }
.gb-mono-inline { font-family: 'JetBrains Mono', monospace; font-weight: 600; color: var(--gb-accent); }

.gb-dish-grid { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.5rem; }
.gb-dish-chip { background: var(--gb-surface-2); border-radius: 999px; padding: 0.35rem 0.75rem; font-size: 0.76rem; color: var(--gb-text); white-space: nowrap; }

.gb-confidence-track { height: 8px; border-radius: 4px; background: var(--gb-surface-2); position: relative; overflow: hidden; margin: 0.4rem 0; }
.gb-confidence-fill { position: absolute; top:0; bottom:0; left:0; background: var(--gb-accent); border-radius: 4px; }
.gb-confidence-label { font-size: 0.78rem; color: var(--gb-muted); }
.gb-caption-note { font-size: 0.8rem; color: var(--gb-muted); margin: 0.3rem 0 0.6rem 0; line-height: 1.4; }

/* --- Loading spinner --- */
[data-testid="stSpinner"] {
    color: var(--gb-muted);
    font-family: 'Inter', sans-serif;
}
[data-testid="stSpinner"] > div {
    border-top-color: var(--gb-accent) !important;
}

/* --- Upload dropzone icon --- */
[data-testid="stFileUploaderDropzone"] svg {
    fill: var(--gb-accent);
    color: var(--gb-accent);
}

/* --- Button press feedback (mobile-first, since that's your primary use case) --- */
div.stButton > button:active,
div.stDownloadButton > button:active,
[data-testid="stFileUploader"] button:active {
    transform: scale(0.96);
    transition: transform 0.08s ease;
}
div.stButton > button[kind="primary"]:active {
    background: var(--gb-accent-dim);
}

/* --- Subtle depth --- */
[data-testid="stVerticalBlockBorderWrapper"] {
    box-shadow: 0 6px 20px rgba(0,0,0,0.35);
}
div.stButton > button[kind="primary"] {
    box-shadow: 0 4px 14px rgba(201,152,46,0.35);
}

.gb-divider { border-top: 1px solid rgba(201,152,46,0.25); margin: 1.1rem 0; }
.gb-section-label { font-size: 0.7rem; color: var(--gb-accent); text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.5rem; font-weight: 600; }
    </style>
    """, unsafe_allow_html=True)


def render_header():
    st.markdown("""
    <div class="gb-header">
        <svg width="34" height="34" viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg">
    <rect width="40" height="40" rx="10" fill="#C9982E"/>
    <circle cx="16" cy="16" r="9" fill="none" stroke="#FFFFFF" stroke-width="3.4"/>
    <ellipse cx="13.5" cy="17.5" rx="1.5" ry="2.3" transform="rotate(-25 13.5 17.5)" fill="#FFFFFF"/>
    <ellipse cx="17.5" cy="17.0" rx="1.5" ry="2.3" transform="rotate(15 17.5 17.0)" fill="#FFFFFF"/>
    <ellipse cx="16.0" cy="13.2" rx="1.5" ry="2.3" transform="rotate(-5 16.0 13.2)" fill="#FFFFFF"/>
    <line x1="22.36" y1="22.36" x2="32" y2="32" stroke="#FFFFFF" stroke-width="4.6" stroke-linecap="round"/>
</svg>
        <p class="gb-title">GulfBite</p>
    </div>
    <p class="gb-subtitle">Nutrition insights for Gulf cuisine</p>
    """, unsafe_allow_html=True)


def render_dish_list():
    dishes = sorted(DISH_RECIPES.keys(), key=display_name)
    chips = "".join(f'<div class="gb-dish-chip">{display_name(d)}</div>' for d in dishes)
    st.markdown(f'<div class="gb-dish-grid">{chips}</div>', unsafe_allow_html=True)
  

def render_stepper(current_stage: str, triggered: bool):
    """Reflects the app's REAL stages — the 'Confirm dish' step only appears
    once we actually know it's needed, since that mirrors how the pipeline behaves."""
    steps = [("upload", "Upload photo")]
    if triggered:
        steps.append(("confirm_dish", "Confirm dish"))
    steps.append(("select_portion", "Choose portion"))
    steps.append(("result", "Result"))

    keys = [s[0] for s in steps]
    active_idx = keys.index(current_stage) if current_stage in keys else 0

    html = ['<div class="gb-stepper">']
    for i, (key, label) in enumerate(steps):
        if i < active_idx:
            cls, mark = "done", str(i + 1)
        elif i == active_idx:
            cls, mark = "active", str(i + 1)
        else:
            cls, mark = "", str(i + 1)
        html.append(
            f'<div class="gb-step {cls}"><div class="gb-step-dot">{mark}</div>'
            f'<div class="gb-step-label">{label}</div></div>'
        )
        if i < len(steps) - 1:
            html.append('<div class="gb-step-line"></div>')
    html.append('</div>')
    st.markdown(''.join(html), unsafe_allow_html=True)


def render_calorie_range(lo: int, hi: int):
    st.markdown(f"""
    <div class="gb-range-numbers">{lo}&ndash;{hi} <span class="gb-range-unit">kcal</span></div>
    <div class="gb-range-track"><div class="gb-range-fill"></div></div>
    <div class="gb-range-caption">This range reflects typical variation in recipes and portion preparation.</div>
    """, unsafe_allow_html=True)

def render_confidence_bar(confidence):
    pct = confidence * 100
    st.markdown(f"""
    <div class="gb-confidence-label">Confidence: <span class="gb-mono-inline">{pct:.0f}%</span></div>
    <div class="gb-confidence-track"><div class="gb-confidence-fill" style="width:{pct:.1f}%;"></div></div>
    """, unsafe_allow_html=True)

def render_macro_bar(protein_g, carbs_g, fat_g):
    protein_kcal = protein_g * 4
    carbs_kcal = carbs_g * 4
    fat_kcal = fat_g * 9
    total = max(protein_kcal + carbs_kcal + fat_kcal, 1)
    p_pct, c_pct, f_pct = protein_kcal/total*100, carbs_kcal/total*100, fat_kcal/total*100

    st.markdown('<div class="gb-section-label">Macronutrient breakdown</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="gb-macro-bar-track">
        <div style="width:{p_pct:.1f}%; background:var(--gb-accent);"></div>
        <div style="width:{c_pct:.1f}%; background:#C9A227;"></div>
        <div style="width:{f_pct:.1f}%; background:#8B97AC;"></div>
    </div>
    <div class="gb-macro-legend">
        <div class="gb-macro-item"><span class="gb-macro-dot" style="background:var(--gb-accent)"></span><span class="gb-macro-value">{protein_g}g</span><br><span class="gb-macro-label">Protein</span></div>
        <div class="gb-macro-item"><span class="gb-macro-dot" style="background:#C9A227"></span><span class="gb-macro-value">{carbs_g}g</span><br><span class="gb-macro-label">Carbs</span></div>
        <div class="gb-macro-item"><span class="gb-macro-dot" style="background:#8B97AC"></span><span class="gb-macro-value">{fat_g}g</span><br><span class="gb-macro-label">Fat</span></div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# STREAMLIT UI
# ============================================================================

st.set_page_config(page_title="GulfBite", page_icon="favicon.png", layout="centered")
inject_theme()

if "stage" not in st.session_state:
    st.session_state.stage = "upload"          # upload -> confirm_dish -> select_portion -> result
    st.session_state.triggered = False
    st.session_state.image = None
    st.session_state.cnn_class = None
    st.session_state.cnn_confidence = None
    st.session_state.candidates = None
    st.session_state.yolo_suggestion = None
    st.session_state.yolo_gate_status = None
    st.session_state.tier_used = None
    st.session_state.final_dish = None
    st.session_state.portion_size = None


def reset():
    for key in list(st.session_state.keys()):
        del st.session_state[key]


render_header()

with st.expander("How this GulfBite app works?"):
    st.write(
        "Snap a photo of your meal. If it looks like something else, we'll double-check with "
        "you. Calories come from real ingredient data — shown as a range, since no two plates "
        "are exactly alike."
    )

with st.expander("What foods can GulfBite recognise?"):
    st.write("GulfBite currently recognises 25 Gulf dishes:")
    render_dish_list()
  
try:
    cnn_model, idx_to_class, yolo_model, ingredient_cache = load_models()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# ---------------------------------------------------------------- STAGE: upload
if st.session_state.stage == "upload":
    render_stepper("upload", st.session_state.triggered)

    with st.container(border=True):
        uploaded = st.file_uploader("Upload a photo of your meal", type=["jpg", "jpeg", "png", "heic", "heif"])
        if uploaded is not None:
            image = ImageOps.exif_transpose(Image.open(uploaded))
            st.session_state.image = image
            st.image(image, caption="Uploaded photo", use_column_width=True)

            with st.spinner("Analyzing photo..."):
                cnn_class, cnn_confidence = run_cnn(image, cnn_model, idx_to_class)

                triggered = (
                    cnn_confidence < CONFIDENCE_THRESHOLD
                    or cnn_class in TRIGGER_SET
                    or cnn_class in WRAP_TRIGGER_SET
                )

                st.session_state.cnn_class = cnn_class
                st.session_state.cnn_confidence = cnn_confidence
                st.session_state.triggered = triggered

                if not triggered:
                    st.session_state.final_dish = cnn_class
                    st.session_state.tier_used = "CNN only"
                    st.session_state.stage = "select_portion"
                    st.rerun()
                else:
                    candidates = get_candidate_group(cnn_class)
                    run_yolo_here = (cnn_class in YOLO_FEATURE_MAP) or (cnn_class == '03_biryani')
                    yolo_suggestion, gate_status = None, None
                    if run_yolo_here:
                        detections = run_yolov8(image, yolo_model)
                        _, gated, gate_status = map_detections_to_suggestion(detections, candidates)
                        yolo_suggestion = gated[0] if gated else None

                    st.session_state.candidates = sorted(candidates)
                    st.session_state.yolo_suggestion = yolo_suggestion
                    st.session_state.yolo_gate_status = gate_status
                    st.session_state.tier_used = (
                        "CNN + YOLO + user confirm" if yolo_suggestion else "CNN + user confirm"
                    )
                    st.session_state.stage = "confirm_dish"
                    st.rerun()

# ---------------------------------------------------------------- STAGE: confirm_dish
elif st.session_state.stage == "confirm_dish":
    render_stepper("confirm_dish", True)

    with st.container(border=True):
        st.image(st.session_state.image, caption="Uploaded photo", use_column_width=True)

        cnn_class = st.session_state.cnn_class
        cnn_conf = st.session_state.cnn_confidence
        candidates = st.session_state.candidates
        yolo_suggestion = st.session_state.yolo_suggestion

        st.write(f"**Detected dish:** {display_name(cnn_class)}")
        render_confidence_bar(cnn_conf)

        reason = get_group_reason(cnn_class)
        if reason:
            st.markdown(f'<p class="gb-caption-note">{reason}</p>', unsafe_allow_html=True)

        if yolo_suggestion:
            st.info(f"Additional analysis suggests this may be **{display_name(yolo_suggestion)}**.")
        else:
            st.write("Please confirm the correct match:")

        default_choice = yolo_suggestion if yolo_suggestion else cnn_class
        default_idx = candidates.index(default_choice) if default_choice in candidates else 0

        choice = st.radio(
            "Choose the correct dish:",
            options=candidates,
            format_func=display_name,
            index=default_idx,
        )

        if st.button("Confirm dish", type="primary"):
            st.session_state.final_dish = choice
            st.session_state.stage = "select_portion"
            st.rerun()

# ---------------------------------------------------------------- STAGE: select_portion
elif st.session_state.stage == "select_portion":
    render_stepper("select_portion", st.session_state.triggered)

    with st.container(border=True):
        st.image(st.session_state.image, caption="Uploaded photo", use_column_width=True)
        st.markdown(f'<p class="gb-dish-name">{display_name(st.session_state.final_dish)}</p>',
                    unsafe_allow_html=True)
        st.write("Select a portion size:")

        col1, col2, col3 = st.columns(3)
        if col1.button("· Small", use_container_width=True):
            st.session_state.portion_size = "S"
            st.session_state.stage = "result"
            st.rerun()
        if col2.button("● Medium", use_container_width=True):
            st.session_state.portion_size = "M"
            st.session_state.stage = "result"
            st.rerun()
        if col3.button("⬤ Large", use_container_width=True):
            st.session_state.portion_size = "L"
            st.session_state.stage = "result"
            st.rerun()

# ---------------------------------------------------------------- STAGE: result
elif st.session_state.stage == "result":
    render_stepper("result", st.session_state.triggered)

    with st.container(border=True):
        st.image(st.session_state.image, caption="Uploaded photo", use_column_width=True)

        dish = st.session_state.final_dish
        nutrition = estimate_nutrition(dish, st.session_state.portion_size, ingredient_cache)
        lo, hi = nutrition['calories_range']

        st.markdown(f'<p class="gb-dish-name">{display_name(dish)}</p>', unsafe_allow_html=True)
        st.markdown(
            f'<p class="gb-dish-meta">Portion: {PORTION_LABELS[st.session_state.portion_size]}</p>',
            unsafe_allow_html=True,
        )

        blurb = DISH_BLURBS.get(dish)
        if blurb:
            st.markdown(f'<p class="gb-caption-note">{blurb}</p>', unsafe_allow_html=True)

        st.markdown('<div class="gb-divider"></div>', unsafe_allow_html=True)

        render_calorie_range(lo, hi)
        render_macro_bar(nutrition['protein_g'], nutrition['carbs_g'], nutrition['fat_g'])

        if nutrition['missing_ingredients']:
            st.warning(f"Nutrition data was unavailable for the following ingredients, which are excluded "
                       f"from this estimate: {', '.join(nutrition['missing_ingredients'])}.")

        st.markdown('<div class="gb-divider"></div>', unsafe_allow_html=True)

        with st.expander("Wrong dish?"):
            all_dishes = sorted(DISH_RECIPES.keys(), key=display_name)
            corrected = st.selectbox(
                "Which dish is this?",
                options=all_dishes,
                format_func=display_name,
                index=all_dishes.index(dish) if dish in all_dishes else 0,
            )
            if st.button("Update", type="primary"):
                st.session_state.final_dish = corrected
                st.session_state.tier_used = "User correction"
                st.rerun()

        with st.expander("How we got this"):
            detail_lines = [f"First guess: {display_name(st.session_state.cnn_class)} "
                 f"(<span class='gb-mono-inline'>{st.session_state.cnn_confidence:.0%}</span> match)"]
            if st.session_state.get('yolo_suggestion'):
                detail_lines.append(f"Visual check pointed to: {display_name(st.session_state.yolo_suggestion)}")
            detail_lines.append(f"You confirmed: {display_name(dish)}")
            st.markdown("<br>".join(detail_lines), unsafe_allow_html=True)

    st.button("Try another photo", on_click=reset)
