"""
GulfBite — Streamlit app
Photo -> dish recognition (3-tier: CNN -> YOLOv8 -> user confirm) -> portion size -> calorie/macro estimate.

BEFORE RUNNING:
  Place these four files into a local `models/` folder:
    models/MobileNetV2_best.keras
    models/class_indices.json
    models/yolov8_ingredient_detector-4/weights/best.pt
    models/ingredient_nutrition_cache.json
"""

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps
import streamlit as st

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:
    pass

# ============================================================================
# CONFIG
# ============================================================================

MODELS_DIR = "models"
CNN_MODEL_PATH = os.path.join(MODELS_DIR, "MobileNetV2_best.keras")
CLASS_INDICES_PATH = os.path.join(MODELS_DIR, "class_indices.json")
YOLO_WEIGHTS_PATH = os.path.join(
    MODELS_DIR, "yolov8_ingredient_detector-4", "weights", "best.pt"
)
INGREDIENT_CACHE_PATH = os.path.join(
    MODELS_DIR, "ingredient_nutrition_cache.json"
)

TRIGGER_SET = {
    "07_ouzi",
    "01_machboos",
    "09_jisheed",
    "02_kabsa",
    "03_biryani",
    "06_saloona",
}
WRAP_TRIGGER_SET = {"10_shawarma", "11_falafel_wrap"}
CONFIDENCE_THRESHOLD = 0.7

YOLO_FEATURE_MAP = {
    "01_machboos": "loomi",
    "07_ouzi": "whole_shank",
    "08_samak_mashwi": "whole_fish",
    "02_kabsa": "whole_chicken_piece",
    "10_shawarma": "shawarma_meat",
    "11_falafel_wrap": "falafel_ball",
}
FEATURE_TO_DISH = {v: k for k, v in YOLO_FEATURE_MAP.items()}

CONFUSION_GROUPS = {
    "rice_cluster": {
        "01_machboos",
        "02_kabsa",
        "03_biryani",
        "07_ouzi",
        "09_jisheed",
        "06_saloona",
    },
    "wrap_cluster": {"10_shawarma", "11_falafel_wrap", "12_falafel"},
}

GROUP_REASONS = {
    "rice_cluster": "Rice dishes like Machboos, Kabsa, and Biryani can look very similar, so we like to double-check.",
    "wrap_cluster": "Wrapped dishes can hide their filling, so we like to double-check.",
}


def get_group_reason(cnn_class):
    for group_name, group_set in CONFUSION_GROUPS.items():
        if cnn_class in group_set:
            return GROUP_REASONS.get(group_name)
    return None


FEATURE_RELIABILITY = {
    "loomi": {"status": "reliable"},
    "whole_chicken_piece": {"status": "unreliable"},
    "whole_shank": {"status": "insufficient_evidence"},
    "whole_fish": {"status": "insufficient_evidence"},
    "shawarma_meat": {"status": "reliable"},
    "falafel_ball": {"status": "reliable"},
}

DISH_RECIPES = {
    "01_machboos": [
        ("rice", 150),
        ("chicken", 130),
        ("olive_oil", 15),
        ("onion", 20),
        ("tomato", 15),
    ],
    "02_kabsa": [
        ("rice", 150),
        ("chicken", 130),
        ("olive_oil", 15),
        ("tomato", 20),
        ("onion", 15),
    ],
    "03_biryani": [
        ("rice", 160),
        ("chicken", 140),
        ("olive_oil", 15),
        ("yogurt", 20),
        ("onion", 20),
    ],
    "04_harees": [("bulgur", 100), ("lamb", 100), ("ghee", 15)],
    "05_thareed": [
        ("pita_bread", 80),
        ("lamb", 120),
        ("mixed_vegetables", 60),
    ],
    "06_saloona": [
        ("lamb", 120),
        ("mixed_vegetables", 100),
        ("tomato", 40),
        ("olive_oil", 15),
    ],
    "07_ouzi": [
        ("rice", 150),
        ("lamb", 180),
        ("mixed_nuts", 15),
        ("olive_oil", 15),
    ],
    "08_samak_mashwi": [("fish", 200), ("olive_oil", 10)],
    "09_jisheed": [("rice", 150), ("fish", 100), ("olive_oil", 10)],
    "10_shawarma": [
        ("pita_bread", 80),
        ("chicken", 100),
        ("garlic_sauce", 20),
        ("pickles", 10),
    ],
    "11_falafel_wrap": [
        ("pita_bread", 80),
        ("falafel", 90),
        ("tahini", 15),
        ("mixed_vegetables", 30),
    ],
    "12_falafel": [("falafel", 120), ("olive_oil", 10)],
    "13_samboosa": [
        ("pastry_dough", 60),
        ("ground_meat", 60),
        ("olive_oil", 10),
    ],
    "14_mutabbaq": [
        ("pastry_dough", 100),
        ("ground_meat", 80),
        ("olive_oil", 15),
    ],
    "15_hummus": [("chickpeas", 80), ("tahini", 15), ("olive_oil", 10)],
    "16_fattoush": [
        ("mixed_vegetables", 150),
        ("pita_bread", 20),
        ("olive_oil", 10),
    ],
    "17_tabbouleh": [
        ("parsley", 80),
        ("bulgur", 20),
        ("tomato", 30),
        ("olive_oil", 15),
    ],
    "18_foul_medames": [("fava_beans", 150), ("olive_oil", 15)],
    "19_shakshuka": [("eggs", 100), ("tomato_sauce", 150), ("olive_oil", 10)],
    "20_balaleet": [("vermicelli", 80), ("sugar", 15), ("eggs", 50)],
    "21_khameer": [("bread_wheat", 80)],
    "22_chebab": [("pancake_batter", 100)],
    "23_luqaimat": [("fried_dough", 100), ("date_syrup", 30)],
    "24_knafeh": [
        ("kunafa_dough", 80),
        ("soft_cheese", 60),
        ("sugar_syrup", 40),
        ("ghee", 15),
    ],
    "25_karak_chai": [("milk", 100), ("black_tea", 100), ("sugar", 10)],
}

DISH_BLURBS = {
    "01_machboos": "A spiced rice dish with meat or chicken, flavoured with dried lime (loomi) — a Bahraini and Kuwaiti staple.",
    "02_kabsa": "Saudi Arabia's best-known dish: spiced rice with meat, often finished with saffron and tomato.",
    "03_biryani": "A layered spiced rice dish with South Asian roots, now a Gulf favourite thanks to centuries of trade.",
    "04_harees": "A slow-cooked wheat and meat porridge, traditionally eaten during Ramadan and Eid across the Gulf.",
    "05_thareed": "Bread soaked in a rich meat and vegetable stew — an Emirati dish often said to have been a favourite of the Prophet Muhammad.",
    "06_saloona": "A everyday spiced stew of meat and vegetables, found in home kitchens across the Gulf.",
    "07_ouzi": "Whole roasted lamb served over rice, traditionally prepared for celebrations and large gatherings.",
    "08_samak_mashwi": "Grilled fish, simply prepared — a reflection of the Gulf's long fishing heritage.",
    "09_jisheed": "An Emirati dish of shredded fish mixed with rice.",
    "10_shawarma": "Spit-roasted meat wrapped in bread — originally Levantine, now a Middle East-wide street food staple.",
    "11_falafel_wrap": "Fried chickpea or fava bean balls in a wrap, a Levantine and Egyptian vegetarian favourite.",
    "12_falafel": "Deep-fried balls of chickpeas or fava beans, a staple across the Gulf.",
    "13_samboosa": "A fried or baked pastry with a savoury filling, especially popular during Ramadan.",
    "14_mutabbaq": "A folded, stuffed pastry with Yemeni roots, filled with either savoury or sweet fillings.",
    "15_hummus": "A creamy chickpea and tahini dip, a Levantine staple found on tables across the Gulf.",
    "16_fattoush": "A Levantine bread salad with crisp vegetables and toasted pita, dressed with sumac.",
    "17_tabbouleh": "A Levantine salad of finely chopped parsley, bulgur, tomato, and lemon.",
    "18_foul_medames": "A stewed fava bean dish of Egyptian origin, a common Gulf breakfast staple.",
    "19_shakshuka": "Eggs poached in a spiced tomato sauce, of North African and Levantine origin.",
    "20_balaleet": "Sweet saffron-spiced vermicelli topped with a savoury omelette — a distinctly Emirati breakfast pairing.",
    "21_khameer": "A traditional Emirati sweet leavened bread, often spiced with cardamom or saffron.",
    "22_chebab": "An Emirati pancake flavoured with cardamom and saffron, popular at breakfast.",
    "23_luqaimat": "Sweet fried dough balls drizzled with date syrup, a classic Ramadan and Eid treat across the Gulf.",
    "24_knafeh": "A cheese pastry soaked in sweet syrup, with roots in the Levant, beloved across the Gulf.",
    "25_karak_chai": "Spiced milk tea with South Asian influence, now an everyday favourite across the Gulf.",
}

PORTION_MULTIPLIERS = {"S": 0.7, "M": 1.0, "L": 1.4}
PORTION_LABELS = {"S": "Small", "M": "Medium", "L": "Large"}
CALORIE_RANGE_PCT = 0.15


def display_name(cls: str) -> str:
    return cls.split("_", 1)[1].replace("_", " ").title()


def get_candidate_group(cnn_class: str) -> set:
    for group in CONFUSION_GROUPS.values():
        if cnn_class in group:
            return group
    return {cnn_class}


# ============================================================================
# MODEL LOADING
# ============================================================================


@st.cache_resource
def load_models():
    import tensorflow as tf
    from ultralytics import YOLO

    missing = [
        p
        for p in (
            CNN_MODEL_PATH,
            CLASS_INDICES_PATH,
            YOLO_WEIGHTS_PATH,
            INGREDIENT_CACHE_PATH,
        )
        if not os.path.exists(p)
    ]
    if missing:
        raise FileNotFoundError(
            "Missing model file(s):\n"
            + "\n".join(missing)
            + "\n\nPlease ensure model weights are located in the models/ directory."
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
# PIPELINE FUNCTIONS
# ============================================================================


def run_cnn(pil_image, model, idx_to_class, img_size=(224, 224)):
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    img = pil_image.convert("RGB").resize(img_size)
    arr = np.array(img).astype("float32")
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr, verbose=0)[0]
    top_idx = int(np.argmax(preds))
    confidence = float(preds[top_idx])
    predicted_class = idx_to_class[top_idx]
    return predicted_class, confidence


def run_yolov8(pil_image, yolo_model, conf_threshold=0.25):
    results = yolo_model.predict(
        np.array(pil_image.convert("RGB")), conf=conf_threshold, verbose=False
    )
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
        return None, None, "no_detection"
    valid = [
        (FEATURE_TO_DISH[feat], conf, feat)
        for feat, conf in detections
        if feat in FEATURE_TO_DISH and FEATURE_TO_DISH[feat] in candidates
    ]
    if not valid:
        return None, None, "no_detection"
    valid.sort(key=lambda x: x[1], reverse=True)
    dish, conf, feature = valid[0]
    status = FEATURE_RELIABILITY.get(
        feature, {"status": "insufficient_evidence"}
    )["status"]
    gated = (dish, conf) if status == "reliable" else None
    return (dish, conf), gated, status


def estimate_nutrition(dish_class, portion_size, ingredient_cache):
    multiplier = PORTION_MULTIPLIERS[portion_size]
    totals = {"calories": 0.0, "protein": 0.0, "carbs": 0.0, "fat": 0.0}
    missing = []
    for ingredient_key, base_grams in DISH_RECIPES[dish_class]:
        info = ingredient_cache.get(ingredient_key)
        if info is None or info.get("source") == "NONE":
            missing.append(ingredient_key)
            continue
        grams = base_grams * multiplier
        for macro in totals:
            totals[macro] += info[macro] * (grams / 100)
    cal_low = totals["calories"] * (1 - CALORIE_RANGE_PCT)
    cal_high = totals["calories"] * (1 + CALORIE_RANGE_PCT)
    return {
        "calories_range": (round(cal_low), round(cal_high)),
        "protein_g": round(totals["protein"], 1),
        "carbs_g": round(totals["carbs"], 1),
        "fat_g": round(totals["fat"], 1),
        "missing_ingredients": missing,
    }


# ============================================================================
# MODERN VISUAL THEME & UI COMPONENTS
# ============================================================================


def inject_theme():
    st.markdown(
        """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

    :root {
        --gb-bg: #110e09;
        --gb-surface: #1b160e;
        --gb-surface-card: #231d13;
        --gb-surface-elevated: #2f271a;
        --gb-accent: #E5A93B;
        --gb-accent-dim: #BF8726;
        --gb-accent-soft: rgba(229, 169, 59, 0.12);
        --gb-text: #F8F5EE;
        --gb-muted: #A39682;
        --gb-border: rgba(229, 169, 59, 0.18);
        --gb-border-subtle: rgba(255, 255, 255, 0.08);
    }

    html, body, [class*="css"] { 
        font-family: 'Plus Jakarta Sans', sans-serif; 
    }
    
    .stApp { 
        background-color: var(--gb-bg);
        background-image: radial-gradient(circle at 1px 1px, rgba(229, 169, 59, 0.04) 1px, transparent 0);
        background-size: 28px 28px;
    }

    #MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
    .block-container { max-width: 680px; padding-top: 2rem; padding-bottom: 3rem; }

    /* Header Styling */
    .gb-header-wrap { display: flex; align-items: center; gap: 0.9rem; margin-bottom: 0.2rem; }
    .gb-logo-badge {
        background: linear-gradient(135deg, #E5A93B 0%, #B87B1D 100%);
        width: 44px; height: 44px; border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        box-shadow: 0 4px 14px rgba(229, 169, 59, 0.35);
    }
    .gb-title {
        font-weight: 800; font-size: 1.85rem; letter-spacing: -0.02em;
        color: var(--gb-text); margin: 0; line-height: 1.1;
    }
    .gb-subtitle { color: var(--gb-muted); font-size: 0.92rem; margin: 0.2rem 0 1.2rem 3.6rem; }

    /* Card Containers */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(180deg, var(--gb-surface) 0%, var(--gb-surface-card) 100%) !important;
        border: 1px solid var(--gb-border) !important;
        border-radius: 18px !important;
        padding: 1.2rem !important;
        box-shadow: 0 12px 36px -8px rgba(0, 0, 0, 0.55);
        animation: gbFadeIn 0.3s ease-out;
    }

    @keyframes gbFadeIn {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Images */
    [data-testid="stImage"] {
        position: relative;
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid var(--gb-border);
        box-shadow: 0 8px 24px rgba(0,0,0,0.4);
    }

    /* Buttons */
    div.stButton > button {
        border-radius: 12px;
        border: 1px solid var(--gb-border);
        background: var(--gb-surface-elevated);
        color: var(--gb-text);
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 600;
        padding: 0.65rem 1.2rem;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        width: 100%;
    }
    div.stButton > button:hover {
        border-color: var(--gb-accent);
        background: var(--gb-accent-soft);
        color: var(--gb-accent);
        transform: translateY(-1px);
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #E5A93B 0%, #C98F25 100%);
        color: #120e06 !important;
        font-weight: 700;
        border: none;
        box-shadow: 0 4px 16px rgba(229, 169, 59, 0.35);
    }
    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #F0B84D 0%, #D89C30 100%);
        box-shadow: 0 6px 22px rgba(229, 169, 59, 0.5);
        transform: translateY(-2px);
    }
    div.stButton > button:active { transform: scale(0.98); }

    /* Modern Radio List Options */
    [data-testid="stRadio"] > div { gap: 0.5rem; }
    [data-testid="stRadio"] label {
        background: var(--gb-surface-card);
        border: 1px solid var(--gb-border-subtle);
        border-radius: 12px;
        padding: 0.65rem 0.9rem !important;
        margin: 0 !important;
        transition: all 0.15s ease;
        width: 100%;
    }
    [data-testid="stRadio"] label:hover {
        border-color: var(--gb-accent);
        background: var(--gb-accent-soft);
    }

    /* Expanders */
    [data-testid="stExpander"] {
        background: transparent !important;
        border: none !important;
        border-top: 1px solid var(--gb-border-subtle) !important;
        margin-top: 0.6rem;
        padding-top: 0.2rem;
    }
    [data-testid="stExpander"] summary {
        color: var(--gb-muted) !important;
        font-weight: 500;
        font-size: 0.9rem;
    }
    [data-testid="stExpander"] summary:hover { color: var(--gb-accent) !important; }

    /* Dish Tags */
    .gb-dish-grid { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.6rem; }
    .gb-dish-chip {
        background: var(--gb-surface-card);
        border: 1px solid var(--gb-border-subtle);
        border-radius: 999px;
        padding: 0.35rem 0.8rem;
        font-size: 0.76rem;
        color: var(--gb-text);
        font-weight: 500;
        transition: all 0.15s ease;
    }
    .gb-dish-chip:hover {
        border-color: var(--gb-accent);
        color: var(--gb-accent);
        background: var(--gb-accent-soft);
    }

    .gb-divider { border-top: 1px solid var(--gb-border-subtle); margin: 1.25rem 0; }
    .gb-caption-note { font-size: 0.85rem; color: var(--gb-muted); margin: 0.3rem 0 0.8rem 0; line-height: 1.45; }
    </style>
    """,
        unsafe_allow_html=True,
    )


def render_header():
    st.markdown(
        """
    <div class="gb-header-wrap">
        <div class="gb-logo-badge">
            <svg width="24" height="24" viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
                <circle cx="16" cy="16" r="9" stroke="#120e06" stroke-width="3.5"/>
                <ellipse cx="13.5" cy="17.5" rx="1.5" ry="2.2" transform="rotate(-25 13.5 17.5)" fill="#120e06"/>
                <ellipse cx="17.5" cy="17.0" rx="1.5" ry="2.2" transform="rotate(15 17.5 17.0)" fill="#120e06"/>
                <ellipse cx="16.0" cy="13.2" rx="1.5" ry="2.2" transform="rotate(-5 16.0 13.2)" fill="#120e06"/>
                <line x1="22.5" y1="22.5" x2="32" y2="32" stroke="#120e06" stroke-width="4.5" stroke-linecap="round"/>
            </svg>
        </div>
        <p class="gb-title">GulfBite</p>
    </div>
    <p class="gb-subtitle">Nutrition insights tailored for authentic Gulf cuisine</p>
    """,
        unsafe_allow_html=True,
    )


def render_dish_list():
    dishes = sorted(DISH_RECIPES.keys(), key=display_name)
    chips = "".join(
        f'<div class="gb-dish-chip">{display_name(d)}</div>' for d in dishes
    )
    st.markdown(
        f'<div class="gb-dish-grid">{chips}</div>', unsafe_allow_html=True
    )


def render_stepper(current_stage: str, triggered: bool):
    steps = [("upload", "Upload photo")]
    if triggered:
        steps.append(("confirm_dish", "Confirm dish"))
    steps.append(("select_portion", "Choose portion"))
    steps.append(("result", "Result"))

    keys = [s[0] for s in steps]
    active_idx = keys.index(current_stage) if current_stage in keys else 0

    html = [
        '<div style="display: flex; align-items: center; justify-content: space-between; margin: 0.5rem 0 1.5rem 0;">'
    ]
    for i, (key, label) in enumerate(steps):
        is_done = i < active_idx
        is_active = i == active_idx

        bg = (
            "linear-gradient(135deg, #E5A93B 0%, #BF8726 100%)"
            if is_done or is_active
            else "#1e180f"
        )
        border = "#E5A93B" if (is_done or is_active) else "rgba(229, 169, 59, 0.2)"
        color = "#120e06" if (is_done or is_active) else "#A39682"
        text_color = "#F8F5EE" if (is_done or is_active) else "#6e6454"
        font_weight = "700" if is_active else ("600" if is_done else "400")

        html.append(f"""
        <div style="display: flex; align-items: center; gap: 7px;">
            <div style="width: 24px; height: 24px; border-radius: 50%; background: {bg}; border: 1.5px solid {border}; color: {color}; display: flex; align-items: center; justify-content: center; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; font-weight: 700;">{i + 1}</div>
            <span style="color: {text_color}; font-weight: {font_weight}; font-size: 0.8rem;">{label}</span>
        </div>
        """)
        if i < len(steps) - 1:
            line_color = (
                "#E5A93B" if is_done else "rgba(229, 169, 59, 0.2)"
            )
            html.append(
                f'<div style="flex: 1; height: 2px; background: {line_color}; margin: 0 8px;"></div>'
            )

    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def render_calorie_hero(lo: int, hi: int):
    st.markdown(
        f"""
    <div style="
        background: radial-gradient(circle at top left, rgba(229, 169, 59, 0.15), transparent 70%), #231d13;
        border: 1px solid rgba(229, 169, 59, 0.3);
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        margin: 1rem 0;
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    ">
        <div style="font-size: 0.78rem; color: #E5A93B; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;">Estimated Calorie Range</div>
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 2.7rem; font-weight: 800; color: #F8F5EE; line-height: 1.1; margin: 6px 0;">
            {lo}&ndash;{hi} <span style="font-size: 1.1rem; font-weight: 500; color: #A39682;">kcal</span>
        </div>
        <p style="font-size: 0.78rem; color: #A39682; margin: 0;">Reflects recipe variations and standard plate sizing.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_macro_cards(protein_g, carbs_g, fat_g):
    col1, col2, col3 = st.columns(3)
    metrics = [
        ("Protein", f"{protein_g}g", "#E5A93B", col1),
        ("Carbs", f"{carbs_g}g", "#4EA8DE", col2),
        ("Fat", f"{fat_g}g", "#9D80CB", col3),
    ]

    for label, val, accent, col in metrics:
        with col:
            st.markdown(
                f"""
            <div style="
                background: #231d13;
                border: 1px solid rgba(255, 255, 255, 0.07);
                border-top: 3.5px solid {accent};
                border-radius: 14px;
                padding: 0.9rem 0.5rem;
                text-align: center;
            ">
                <div style="color: #A39682; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600;">{label}</div>
                <div style="font-family: 'JetBrains Mono', monospace; color: #F8F5EE; font-size: 1.35rem; font-weight: 700; margin-top: 3px;">{val}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )


def render_confidence_bar(confidence):
    pct = confidence * 100
    st.markdown(
        f"""
    <div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 0.6rem;">
        <span style="font-size: 0.82rem; color: var(--gb-muted);">AI Confidence</span>
        <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.88rem; font-weight: 700; color: var(--gb-accent);">{pct:.0f}%</span>
    </div>
    <div style="height: 6px; border-radius: 3px; background: rgba(255,255,255,0.06); overflow: hidden; margin: 0.35rem 0 0.8rem 0;">
        <div style="width: {pct:.1f}%; height: 100%; background: linear-gradient(90deg, #E5A93B, #F3C36A); border-radius: 3px;"></div>
    </div>
    """,
        unsafe_allow_html=True,
    )


# ============================================================================
# STREAMLIT UI FLOW
# ============================================================================

st.set_page_config(
    page_title="GulfBite",
    page_icon="🍲",
    layout="centered",
    initial_sidebar_state="collapsed",
)
inject_theme()

if "stage" not in st.session_state:
    st.session_state.stage = "upload"
    st.session_state.triggered = False
    st.session_state.image = None
    st.session_state.cnn_class = None
    st.session_state.cnn_confidence = None
    st.session_state.candidates = None
    st.session_state.yolo_suggestion = None
    st.session_state.yolo_gate_status = None
    st.session_state.tier_used = None
    st.session_state.final_dish = None
    st.session_state.portion_size = "M"


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
        uploaded = st.file_uploader(
            "Upload a photo of your meal",
            type=["jpg", "jpeg", "png", "heic", "heif"],
        )
        if uploaded is not None:
            image = ImageOps.exif_transpose(Image.open(uploaded))
            st.session_state.image = image
            st.image(image, caption="Uploaded photo", use_column_width=True)

            with st.spinner("Analyzing cuisine & ingredients..."):
                cnn_class, cnn_confidence = run_cnn(
                    image, cnn_model, idx_to_class
                )

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
                    run_yolo_here = (cnn_class in YOLO_FEATURE_MAP) or (
                        cnn_class == "03_biryani"
                    )
                    yolo_suggestion, gate_status = None, None
                    if run_yolo_here:
                        detections = run_yolov8(image, yolo_model)
                        _, gated, gate_status = map_detections_to_suggestion(
                            detections, candidates
                        )
                        yolo_suggestion = gated[0] if gated else None

                    st.session_state.candidates = sorted(candidates)
                    st.session_state.yolo_suggestion = yolo_suggestion
                    st.session_state.yolo_gate_status = gate_status
                    st.session_state.tier_used = (
                        "CNN + YOLO + user confirm"
                        if yolo_suggestion
                        else "CNN + user confirm"
                    )
                    st.session_state.stage = "confirm_dish"
                    st.rerun()

# ---------------------------------------------------------------- STAGE: confirm_dish
elif st.session_state.stage == "confirm_dish":
    render_stepper("confirm_dish", True)

    with st.container(border=True):
        st.image(
            st.session_state.image,
            caption="Uploaded photo",
            use_column_width=True,
        )

        cnn_class = st.session_state.cnn_class
        cnn_conf = st.session_state.cnn_confidence
        candidates = st.session_state.candidates
        yolo_suggestion = st.session_state.yolo_suggestion

        st.markdown(
            f'<div style="font-size: 1.25rem; font-weight: 700; color: #F8F5EE; margin-top: 0.5rem;">Initial Match: {display_name(cnn_class)}</div>',
            unsafe_allow_html=True,
        )
        render_confidence_bar(cnn_conf)

        reason = get_group_reason(cnn_class)
        if reason:
            st.markdown(
                f'<p class="gb-caption-note">{reason}</p>',
                unsafe_allow_html=True,
            )

        if yolo_suggestion:
            st.info(
                f"Ingredient visual analysis points towards: **{display_name(yolo_suggestion)}**."
            )

        default_choice = yolo_suggestion if yolo_suggestion else cnn_class
        default_idx = (
            candidates.index(default_choice)
            if default_choice in candidates
            else 0
        )

        choice = st.radio(
            "Select the matching dish:",
            options=candidates,
            format_func=display_name,
            index=default_idx,
        )

        st.write("")
        if st.button("Confirm Dish & Continue", type="primary"):
            st.session_state.final_dish = choice
            st.session_state.stage = "select_portion"
            st.rerun()

# ---------------------------------------------------------------- STAGE: select_portion
elif st.session_state.stage == "select_portion":
    render_stepper("select_portion", st.session_state.triggered)

    with st.container(border=True):
        st.image(
            st.session_state.image,
            caption="Uploaded photo",
            use_column_width=True,
        )

        st.markdown(
            f'<div style="font-size: 1.5rem; font-weight: 800; color: #F8F5EE; margin-top: 0.6rem;">{display_name(st.session_state.final_dish)}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="gb-caption-note">Select your meal portion to calculate macros:</p>',
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)
        if col1.button("🌱 Small", use_container_width=True):
            st.session_state.portion_size = "S"
            st.session_state.stage = "result"
            st.rerun()
        if col2.button("🍽️ Medium", use_container_width=True):
            st.session_state.portion_size = "M"
            st.session_state.stage = "result"
            st.rerun()
        if col3.button("👑 Large", use_container_width=True):
            st.session_state.portion_size = "L"
            st.session_state.stage = "result"
            st.rerun()

# ---------------------------------------------------------------- STAGE: result
elif st.session_state.stage == "result":
    render_stepper("result", st.session_state.triggered)

    with st.container(border=True):
        st.image(
            st.session_state.image,
            caption="Uploaded photo",
            use_column_width=True,
        )

        dish = st.session_state.final_dish
        nutrition = estimate_nutrition(
            dish, st.session_state.portion_size, ingredient_cache
        )
        lo, hi = nutrition["calories_range"]

        st.markdown(
            f'<div style="font-size: 1.6rem; font-weight: 800; color: #F8F5EE; margin-top: 0.6rem;">{display_name(dish)}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="color: #A39682; font-size: 0.88rem; font-weight: 500;">Portion size: <strong style="color:#E5A93B;">{PORTION_LABELS[st.session_state.portion_size]}</strong></div>',
            unsafe_allow_html=True,
        )

        blurb = DISH_BLURBS.get(dish)
        if blurb:
            st.markdown(
                f'<p class="gb-caption-note" style="margin-top: 0.4rem;">{blurb}</p>',
                unsafe_allow_html=True,
            )

        render_calorie_hero(lo, hi)
        render_macro_cards(
            nutrition["protein_g"], nutrition["carbs_g"], nutrition["fat_g"]
        )

        if nutrition["missing_ingredients"]:
            st.warning(
                f"Missing standard data for: {', '.join(nutrition['missing_ingredients'])}."
            )

        st.markdown('<div class="gb-divider"></div>', unsafe_allow_html=True)

        with st.expander("Change dish?"):
            all_dishes = sorted(DISH_RECIPES.keys(), key=display_name)
            corrected = st.selectbox(
                "Select correct dish:",
                options=all_dishes,
                format_func=display_name,
                index=all_dishes.index(dish) if dish in all_dishes else 0,
            )
            if st.button("Update Dish", type="primary"):
                st.session_state.final_dish = corrected
                st.session_state.tier_used = "User correction"
                st.rerun()

        with st.expander("Pipeline breakdown"):
            lines = [
                f"Initial CNN prediction: **{display_name(st.session_state.cnn_class)}** ({st.session_state.cnn_confidence:.0%})"
            ]
            if st.session_state.get("yolo_suggestion"):
                lines.append(
                    f"YOLOv8 visual feature: **{display_name(st.session_state.yolo_suggestion)}**"
                )
            lines.append(f"Confirmed dish: **{display_name(dish)}**")
            lines.append(f"Decision tier: `{st.session_state.tier_used}`")
            st.markdown("<br>".join(lines), unsafe_allow_html=True)

    st.write("")
    st.button("📸 Analyze Another Photo", on_click=reset)
