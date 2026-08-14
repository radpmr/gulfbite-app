"""
GulfBite — Smart Gulf Cuisine Nutrition Assistant
-------------------------------------------------
Identifies authentic Gulf dishes using a multi-tiered pipeline:
1. MobileNetV2 (CNN) classification for initial dish match & confidence scoring.
2. Out-of-distribution / Non-food rejection via margin and entropy checks.
3. YOLOv8 feature detection for visually ambiguous dishes (e.g., loomi in Machboos).
4. Portion-based authentic macro and calorie estimation.

Following files exist in the `models/` directory:
- models/MobileNetV2_best.keras
- models/class_indices.json
- models/yolov8_ingredient_detector-4/weights/best.pt
- models/ingredient_nutrition_cache.json
"""

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps
import streamlit as st

# Force CPU inference for stability and suppress TensorFlow verbose logging
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Register HEIC/HEIF image support for mobile uploads (iPhone camera shots)
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

# ============================================================================
# 1. CONFIGURATION & CONSTANTS
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

# Sets of dishes that frequently confuse the CNN and require verification
TRIGGER_SET = {
    "07_ouzi",
    "01_machboos",
    "09_jisheed",
    "02_kabsa",
    "03_biryani",
    "06_saloona",
}
WRAP_TRIGGER_SET = {"10_shawarma", "11_falafel_wrap"}

# Thresholds for AI confidence and non-food detection
CONFIDENCE_THRESHOLD = 0.70  # Standard high-confidence threshold
MIN_CONFIDENCE = 0.50        # Minimum top-1 probability to qualify as food
MIN_MARGIN = 0.15            # Difference between top 2 classes to prevent random guesses
MAX_ENTROPY = 2.50           # Shannon entropy cap (high entropy = diffuse / non-food)

# Feature-to-dish mappings for YOLO validation
YOLO_FEATURE_MAP = {
    "01_machboos": "loomi",
    "07_ouzi": "whole_shank",
    "08_samak_mashwi": "whole_fish",
    "02_kabsa": "whole_chicken_piece",
    "10_shawarma": "shawarma_meat",
    "11_falafel_wrap": "falafel_ball",
}
FEATURE_TO_DISH = {v: k for k, v in YOLO_FEATURE_MAP.items()}

# Related dish groups for fallback multiple-choice confirmation
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

# User-friendly explanation notes shown during verification
GROUP_REASONS = {
    "rice_cluster": "Rice dishes like Machboos, Kabsa, and Biryani can look very similar, so we like to double-check.",
    "wrap_cluster": "Wrapped dishes can hide their filling, so we like to double-check.",
}


def get_group_reason(cnn_class: str) -> Optional[str]:
    """Retrieve an intuitive explanation for why the app is asking for confirmation."""
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

# Ingredient breakdown recipes (base grams for a medium portion)
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

# Educational dish descriptions
DISH_BLURBS = {
    "01_machboos": "A spiced rice dish with meat or chicken, flavoured with dried lime (loomi) — a Bahraini and Kuwaiti staple.",
    "02_kabsa": "Saudi Arabia's best-known dish: spiced rice with meat, often finished with saffron and tomato.",
    "03_biryani": "A layered spiced rice dish with South Asian roots, now a Gulf favourite thanks to centuries of trade.",
    "04_harees": "A slow-cooked wheat and meat porridge, traditionally eaten during Ramadan and Eid across the Gulf.",
    "05_thareed": "Bread soaked in a rich meat and vegetable stew — an Emirati dish often said to have been a favourite of the Prophet Muhammad.",
    "06_saloona": "An everyday spiced stew of meat and vegetables, found in home kitchens across the Gulf.",
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

# Serving size multipliers
PORTION_MULTIPLIERS = {"S": 0.7, "M": 1.0, "L": 1.4}
PORTION_LABELS = {"S": "Small", "M": "Medium", "L": "Large"}
CALORIE_RANGE_PCT = 0.15


def display_name(cls: str) -> str:
    """Format technical class names into clean titles (e.g., '01_machboos' -> 'Machboos')."""
    return cls.split("_", 1)[1].replace("_", " ").title()


# Categorised groupings for the Explorer tab
DISH_CATEGORIES = {
    "🍚 Rice & Feasts": [
        "01_machboos",
        "02_kabsa",
        "03_biryani",
        "07_ouzi",
        "09_jisheed",
    ],
    "🥘 Stews & Mains": [
        "04_harees",
        "05_thareed",
        "06_saloona",
        "08_samak_mashwi",
    ],
    "🌯 Street Food & Bites": [
        "10_shawarma",
        "11_falafel_wrap",
        "12_falafel",
        "13_samboosa",
        "14_mutabbaq",
    ],
    "🫓 Breads & Breakfast": [
        "18_foul_medames",
        "19_shakshuka",
        "20_balaleet",
        "21_khameer",
        "22_chebab",
    ],
    "🥗 Salads & Dips": ["15_hummus", "16_fattoush", "17_tabbouleh"],
    "🍯 Desserts & Tea": ["23_luqaimat", "24_knafeh", "25_karak_chai"],
}


def get_candidate_group(cnn_class: str) -> set:
    """Return the candidate group of similar dishes if a trigger dish is detected."""
    for group in CONFUSION_GROUPS.values():
        if cnn_class in group:
            return group
    return {cnn_class}


# ============================================================================
# 2. MODEL LOADING & CACHING
# ============================================================================

@st.cache_resource
def load_models():
    """Load neural network models and caches once and keep them in memory."""
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

    # Load CNN classifier
    cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
    with open(CLASS_INDICES_PATH) as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}

    # Load YOLO ingredient detector
    yolo_model = YOLO(YOLO_WEIGHTS_PATH)

    # Load nutrition database
    with open(INGREDIENT_CACHE_PATH) as f:
        ingredient_cache = json.load(f)

    return cnn_model, idx_to_class, yolo_model, ingredient_cache


# ============================================================================
# 3. CORE AI INFERENCE FUNCTIONS
# ============================================================================

def run_cnn(pil_image, model, idx_to_class, img_size=(224, 224)):
    """
    Run MobileNetV2 inference and calculate Shannon Entropy & Top Margin 
    to reliably detect and reject non-food images.
    """
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    img = pil_image.convert("RGB").resize(img_size)
    arr = np.array(img).astype("float32")
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr, verbose=0)[0]

    # Rank predictions
    sorted_indices = np.argsort(preds)[::-1]
    top_idx = int(sorted_indices[0])
    second_idx = int(sorted_indices[1])

    confidence = float(preds[top_idx])
    second_confidence = float(preds[second_idx])
    margin = confidence - second_confidence

    # Shannon Entropy cap check (uniform spread indicates non-food)
    eps = 1e-12
    entropy = -np.sum(preds * np.log(preds + eps))

    predicted_class = idx_to_class[top_idx]
    return predicted_class, confidence, margin, entropy


def run_yolov8(pil_image, yolo_model, conf_threshold=0.25):
    """Run YOLOv8 object detection to locate distinct ingredients/visual features."""
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
    """Correlate ingredient detections to suggested candidate dishes."""
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
    """Calculate scaled macronutrient and calorie ranges based on authentic recipes."""
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
# 4. CUSTOM THEME & UI STYLING
# ============================================================================

def inject_theme():
    """Inject custom dark-gold glassmorphic CSS styling."""
    st.markdown(
        """<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Outfit:wght@500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

:root {
    --gb-bg: #0d0a07;
    --gb-card-bg: rgba(26, 21, 15, 0.75);
    --gb-card-border: rgba(229, 169, 59, 0.22);
    --gb-gold-start: #f3c36a;
    --gb-gold: #E5A93B;
    --gb-gold-dim: #996e21;
    --gb-gold-glow: rgba(229, 169, 59, 0.18);
    --gb-text: #FBF8F1;
    --gb-muted: #A39682;
}

html, body, [class*="css"] { 
    font-family: 'Plus Jakarta Sans', sans-serif; 
}

/* Background Atmosphere */
.stApp { 
    background-color: var(--gb-bg);
    background-image: 
        radial-gradient(ellipse at 50% 0%, rgba(229, 169, 59, 0.12) 0%, transparent 60%),
        radial-gradient(circle at 1px 1px, rgba(229, 169, 59, 0.03) 1px, transparent 0);
    background-size: 100% 100%, 28px 28px;
    color: var(--gb-text);
}

#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { max-width: 640px; padding-top: 2rem; padding-bottom: 3.5rem; }

/* Global Card Glassmorphic Frame */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--gb-card-bg) !important;
    backdrop-filter: blur(16px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(16px) saturate(180%) !important;
    border: 1px solid var(--gb-card-border) !important;
    border-radius: 20px !important;
    padding: 1.4rem !important;
    box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.7), 0 0 15px var(--gb-gold-glow) !important;
    transition: border-color 0.3s ease;
}

/* Sleek Drag & Drop Zone */
[data-testid="stFileUploaderDropzone"] {
    background: radial-gradient(circle at 50% 30%, rgba(229, 169, 59, 0.08) 0%, rgba(20, 16, 11, 0.7) 100%) !important;
    border: 1.5px dashed var(--gb-card-border) !important;
    border-radius: 16px !important;
    padding: 2.2rem 1rem !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--gb-gold) !important;
    background: radial-gradient(circle at 50% 30%, rgba(229, 169, 59, 0.16) 0%, rgba(26, 21, 15, 0.9) 100%) !important;
    box-shadow: 0 0 30px rgba(229, 169, 59, 0.25) !important;
    transform: translateY(-2px);
}

[data-testid="stFileUploaderDropzone"] svg {
    fill: var(--gb-gold) !important;
    color: var(--gb-gold) !important;
    filter: drop-shadow(0 2px 8px rgba(229, 169, 59, 0.4));
}

[data-testid="stFileUploaderDropzoneInstructions"] > div > span {
    font-family: 'Outfit', sans-serif !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    color: var(--gb-text) !important;
    letter-spacing: -0.01em;
}

[data-testid="stFileUploaderDropzoneInstructions"] > div > small {
    color: var(--gb-muted) !important;
    font-size: 0.8rem !important;
}

/* File Upload Button */
[data-testid="stFileUploader"] button {
    border-radius: 10px !important;
    border: 1px solid var(--gb-gold) !important;
    background: rgba(229, 169, 59, 0.1) !important;
    color: var(--gb-gold) !important;
    font-weight: 600 !important;
    padding: 0.45rem 1.2rem !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.3) !important;
    transition: all 0.2s ease !important;
}
[data-testid="stFileUploader"] button:hover {
    background: var(--gb-gold) !important;
    color: #120e06 !important;
    box-shadow: 0 0 16px rgba(229, 169, 59, 0.5) !important;
}

/* Buttons */
div.stButton > button {
    font-family: 'Outfit', sans-serif;
    border-radius: 12px;
    border: 1px solid var(--gb-card-border);
    background: rgba(255,255,255,0.03);
    color: var(--gb-text);
    font-weight: 600;
    font-size: 0.92rem;
    padding: 0.7rem 1.3rem;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    width: 100%;
}
div.stButton > button:hover {
    border-color: var(--gb-gold);
    color: var(--gb-gold);
    background: rgba(229, 169, 59, 0.08);
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(0,0,0,0.4);
}
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #f5c76f 0%, #E5A93B 50%, #b87a17 100%);
    color: #110e08 !important;
    font-weight: 800;
    letter-spacing: 0.02em;
    border: none;
    box-shadow: 0 6px 20px rgba(229, 169, 59, 0.4);
}
div.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #ffd37f 0%, #f0b545 50%, #c9881e 100%);
    box-shadow: 0 8px 26px rgba(229, 169, 59, 0.6);
    transform: translateY(-2px);
}

/* Expanders */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.015) !important;
    border: 1px solid rgba(229, 169, 59, 0.12) !important;
    border-radius: 14px !important;
    margin-bottom: 0.6rem !important;
    transition: border-color 0.2s ease;
}
[data-testid="stExpander"]:hover {
    border-color: rgba(229, 169, 59, 0.28) !important;
}
[data-testid="stExpander"] summary {
    font-family: 'Outfit', sans-serif !important;
    color: var(--gb-text) !important;
    font-weight: 600;
    font-size: 0.95rem;
}
[data-testid="stExpander"] summary:hover { color: var(--gb-gold) !important; }

/* Clear out Streamlit's inner block containers inside columns to avoid double borders */
div[data-testid="column"] [data-testid="stVerticalBlockBorderWrapper"],
div[data-testid="column"] > div[data-testid="stVerticalBlock"],
div[data-testid="column"] > div {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

/* Redesigned interactive portion selection tiles */
div[data-testid="column"] button {
    height: 110px !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    background: linear-gradient(180deg, rgba(35, 29, 19, 0.7) 0%, rgba(20, 16, 11, 0.9) 100%) !important;
    border: 1.5px solid rgba(229, 169, 59, 0.22) !important;
    border-radius: 16px !important;
    padding: 12px !important;
    margin: 0 !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

div[data-testid="column"] button:hover {
    border-color: #E5A93B !important;
    background: linear-gradient(180deg, rgba(229, 169, 59, 0.15) 0%, rgba(35, 29, 19, 0.95)) !important;
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 25px rgba(229, 169, 59, 0.3) !important;
}

div[data-testid="column"] button p {
    margin: 0 !important;
    line-height: 1.3 !important;
    color: #FBF8F1 !important;
    font-weight: 700 !important;
}

/* Green pill badge for verified pipeline path */
.tech-pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.76rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    background: rgba(34, 197, 94, 0.12);
    color: #4ade80;
    border: 1px solid rgba(34, 197, 94, 0.3);
}

/* Verification info callout box */
.verify-callout {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    background: rgba(229, 169, 59, 0.06);
    border: 1px solid rgba(229, 169, 59, 0.2);
    border-left: 3.5px solid #E5A93B;
    border-radius: 12px;
    padding: 10px 14px;
    margin: 10px 0 14px 0;
}

/* Modern segmented top tabs */
div[data-testid="stTabs"] {
    background: transparent !important;
    margin-bottom: 1rem !important;
}

div[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: rgba(22, 18, 12, 0.9) !important;
    border: 1px solid rgba(229, 169, 59, 0.22) !important;
    border-radius: 14px !important;
    padding: 4px !important;
    gap: 6px !important;
}

div[data-testid="stTabs"] [data-baseweb="tab-border"] {
    display: none !important;
}

div[data-testid="stTabs"] [data-baseweb="tab"] {
    border-radius: 10px !important;
    height: 38px !important;
    padding: 0 16px !important;
    color: var(--gb-muted) !important;
    font-weight: 600 !important;
    font-size: 0.86rem !important;
    background: transparent !important;
    border: 1px solid transparent !important;
    transition: all 0.2s ease !important;
}

div[data-testid="stTabs"] [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(229, 169, 59, 0.2) 0%, rgba(229, 169, 59, 0.08) 100%) !important;
    color: #FBF8F1 !important;
    border: 1px solid rgba(229, 169, 59, 0.35) !important;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3) !important;
}

div[data-testid="stTabs"] [data-testid="stVerticalBlockBorderWrapper"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

/* Scoped filter category chips */
.category-pill-container div[data-testid="stRadio"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

.category-pill-container div[data-testid="stRadio"] > div {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: wrap !important;
    gap: 8px !important;
    border: none !important;
    background: transparent !important;
    padding: 0 !important;
}

.category-pill-container div[data-testid="stRadio"] label {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(229, 169, 59, 0.2) !important;
    border-radius: 999px !important;
    padding: 6px 14px !important;
    margin: 0 !important;
    min-width: auto !important;
    height: auto !important;
    width: auto !important;
    display: inline-flex !important;
    align-items: center !important;
    white-space: nowrap !important;
}

.category-pill-container div[data-testid="stRadio"] label:hover {
    border-color: #E5A93B !important;
    background: rgba(229, 169, 59, 0.12) !important;
    transform: translateY(-1px);
}

.category-pill-container div[data-testid="stRadio"] label span p {
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    color: #FBF8F1 !important;
    white-space: nowrap !important;
    margin: 0 !important;
}
</style>""",
        unsafe_allow_html=True,
    )


# ============================================================================
# 5. UI HEADER, STEPPER, & CARD COMPONENTS
# ============================================================================

def render_header():
    """Render top header with glowing badge and logo."""
    st.markdown(
        """<div style="display: flex; align-items: center; gap: 14px; margin-bottom: 0.3rem;">
<div style="
    background: linear-gradient(135deg, #f5c76f 0%, #E5A93B 60%, #a66d12 100%);
    width: 48px; height: 48px; border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 6px 18px rgba(229, 169, 59, 0.4);
">
    <svg width="26" height="26" viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
        <circle cx="16" cy="16" r="9" stroke="#120e06" stroke-width="3.5"/>
        <ellipse cx="13.5" cy="17.5" rx="1.5" ry="2.2" transform="rotate(-25 13.5 17.5)" fill="#120e06"/>
        <ellipse cx="17.5" cy="17.0" rx="1.5" ry="2.2" transform="rotate(15 17.5 17.0)" fill="#120e06"/>
        <ellipse cx="16.0" cy="13.2" rx="1.5" ry="2.2" transform="rotate(-5 16.0 13.2)" fill="#120e06"/>
        <line x1="22.5" y1="22.5" x2="32" y2="32" stroke="#120e06" stroke-width="4.5" stroke-linecap="round"/>
    </svg>
</div>
<div>
    <h1 style="font-family: 'Outfit', sans-serif; font-size: 2.1rem; font-weight: 800; color: #FBF8F1; margin: 0; line-height: 1; letter-spacing: -0.03em;">GulfBite</h1>
    <p style="color: #A39682; font-size: 0.88rem; margin: 4px 0 0 0;">AI Nutrition Insights for Authentic Gulf Cuisine</p>
</div>
</div>
<div style="height: 1px; background: linear-gradient(90deg, rgba(229,169,59,0.3) 0%, transparent 100%); margin: 1rem 0 1.4rem 0;"></div>""",
        unsafe_allow_html=True,
    )


def render_interactive_dish_explorer():
    """Render interactive category filter and dish information viewer."""
    st.markdown('<div class="category-pill-container">', unsafe_allow_html=True)
    selected_category = st.radio(
        "Filter by category:",
        options=list(DISH_CATEGORIES.keys()),
        index=0,
        horizontal=True,
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if not selected_category:
        selected_category = "🍚 Rice & Feasts"

    category_dishes = DISH_CATEGORIES[selected_category]

    st.markdown(
        '<div style="margin-top: 14px; margin-bottom: 6px; font-size: 0.8rem; font-weight: 700; color: #A39682; text-transform: uppercase; letter-spacing: 0.05em;">Select Dish</div>',
        unsafe_allow_html=True,
    )

    selected_dish = st.selectbox(
        "Choose a dish to explore:",
        options=category_dishes,
        format_func=display_name,
        index=0,
        label_visibility="collapsed",
    )

    if selected_dish:
        blurb = DISH_BLURBS.get(selected_dish, "")
        st.markdown(
            f"""<div style="
                background: linear-gradient(135deg, rgba(229, 169, 59, 0.08) 0%, rgba(20, 16, 11, 0.6) 100%);
                border: 1px solid rgba(229, 169, 59, 0.25);
                border-left: 4px solid #E5A93B;
                border-radius: 14px;
                padding: 14px 16px;
                margin-top: 10px;
                box-shadow: 0 4px 14px rgba(0,0,0,0.3);
            ">
                <div style="font-weight: 800; color: #FBF8F1; font-size: 1.05rem; margin-bottom: 4px;">{display_name(selected_dish)}</div>
                <div style="color: #A39682; font-size: 0.84rem; line-height: 1.45;">{blurb}</div>
            </div>""",
            unsafe_allow_html=True,
        )


def render_stepper(current_stage: str, triggered: bool):
    """Render numbered pipeline steps indicator."""
    steps = [("upload", "Upload photo")]
    if triggered:
        steps.append(("confirm_dish", "Confirm dish"))
    steps.append(("select_portion", "Choose portion"))
    steps.append(("result", "Result"))

    keys = [s[0] for s in steps]
    active_idx = keys.index(current_stage) if current_stage in keys else 0

    html = [
        '<div style="display: flex; align-items: center; justify-content: space-between; margin: 0.2rem 0 1.4rem 0; padding: 0 4px;">'
    ]
    for i, (key, label) in enumerate(steps):
        is_done = i < active_idx
        is_active = i == active_idx

        bg = (
            "linear-gradient(135deg, #f5c76f 0%, #E5A93B 100%)"
            if is_done or is_active
            else "rgba(255,255,255,0.04)"
        )
        border = (
            "#E5A93B"
            if (is_done or is_active)
            else "rgba(255, 255, 255, 0.08)"
        )
        color = "#110e08" if (is_done or is_active) else "#A39682"
        text_color = "#FBF8F1" if (is_done or is_active) else "#6e6454"
        font_weight = "700" if is_active else ("600" if is_done else "400")
        glow = (
            "box-shadow: 0 0 12px rgba(229, 169, 59, 0.45);"
            if is_active
            else ""
        )

        html.append(f"""<div style="display: flex; align-items: center; gap: 8px;">
<div style="width: 26px; height: 26px; border-radius: 50%; background: {bg}; border: 1.5px solid {border}; color: {color}; display: flex; align-items: center; justify-content: center; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; font-weight: 700; {glow}">{i + 1}</div>
<span style="font-family: 'Outfit', sans-serif; color: {text_color}; font-weight: {font_weight}; font-size: 0.85rem;">{label}</span>
</div>""")
        if i < len(steps) - 1:
            line_color = (
                "linear-gradient(90deg, #E5A93B, rgba(229,169,59,0.3))"
                if is_done
                else "rgba(255, 255, 255, 0.08)"
            )
            html.append(
                f'<div style="flex: 1; height: 2px; background: {line_color}; margin: 0 10px; border-radius: 1px;"></div>'
            )

    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def render_calorie_hero(lo: int, hi: int):
    """Render high-contrast calorie range highlight card."""
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
    """Render three distinct nutrient breakdown cards."""
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
    """Render styled percentage progress bar for AI prediction confidence."""
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
# 6. APP INITIALIZATION & STATE SETUP
# ============================================================================

st.set_page_config(
    page_title="GulfBite",
    page_icon="🍲",
    layout="centered",
    initial_sidebar_state="collapsed",
)
inject_theme()

# Initialize session state variables
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
    """Reset session state to restart the analysis pipeline."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]


# ============================================================================
# 7. TOP HEADER & INTERACTIVE TABS
# ============================================================================

render_header()

guide_tab, dishes_tab = st.tabs(["✨ How It Works", "🍲 Supported Dishes (25)"])

with guide_tab:
    st.markdown(
        """<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin: 10px 0 6px 0;">
<div style="background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(229, 169, 59, 0.2); border-top: 3px solid #E5A93B; border-radius: 14px; padding: 14px 10px; text-align: center;">
    <div style="font-size: 1.6rem; margin-bottom: 6px;">📸</div>
    <div style="color: #F8F5EE; font-weight: 700; font-size: 0.85rem; margin-bottom: 4px;">1. Snap Meal</div>
    <div style="color: #A39682; font-size: 0.76rem; line-height: 1.35;">Upload a photo of your traditional Gulf plate.</div>
</div>
<div style="background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(229, 169, 59, 0.2); border-top: 3px solid #E5A93B; border-radius: 14px; padding: 14px 10px; text-align: center;">
    <div style="font-size: 1.6rem; margin-bottom: 6px;">🔍</div>
    <div style="color: #F8F5EE; font-weight: 700; font-size: 0.85rem; margin-bottom: 4px;">2. AI Check</div>
    <div style="color: #A39682; font-size: 0.76rem; line-height: 1.35;">If dishes look similar, we double-check with you.</div>
</div>
<div style="background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(229, 169, 59, 0.2); border-top: 3px solid #E5A93B; border-radius: 14px; padding: 14px 10px; text-align: center;">
    <div style="font-size: 1.6rem; margin-bottom: 6px;">⚖️</div>
    <div style="color: #F8F5EE; font-weight: 700; font-size: 0.85rem; margin-bottom: 4px;">3. Honest Range</div>
    <div style="color: #A39682; font-size: 0.76rem; line-height: 1.35;">Real nutrition ranges from authentic ingredients.</div>
</div>
</div>""",
        unsafe_allow_html=True,
    )

with dishes_tab:
    render_interactive_dish_explorer()

st.markdown("<div style='margin-bottom: 1.4rem;'></div>", unsafe_allow_html=True)


# ============================================================================
# 8. LOAD MODELS INTO MEMORY
# ============================================================================

try:
    cnn_model, idx_to_class, yolo_model, ingredient_cache = load_models()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()


# ============================================================================
# 9. STAGE 1: UPLOAD & INFERENCE PIPELINE
# ============================================================================

if st.session_state.stage == "upload":
    render_stepper("upload", st.session_state.triggered)

    with st.container(border=True):
        st.markdown(
            """<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px; flex-wrap: wrap;">
<span style="background: rgba(229, 169, 59, 0.08); border: 1px solid rgba(229, 169, 59, 0.2); border-radius: 8px; padding: 4px 10px; font-size: 0.76rem; color: #E5A93B; font-weight: 500;">📸 Top-down photo</span>
<span style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 4px 10px; font-size: 0.76rem; color: #A39682;">💡 Good lighting</span>
<span style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 4px 10px; font-size: 0.76rem; color: #A39682;">🍲 Single dish</span>
</div>""",
            unsafe_allow_html=True,
        )

        image_to_process = None

        uploaded = st.file_uploader(
            "Upload a photo of your meal",
            type=["jpg", "jpeg", "png", "heic", "heif"],
            label_visibility="collapsed",
        )

        if uploaded is not None:
            image_to_process = ImageOps.exif_transpose(Image.open(uploaded))

        if image_to_process is not None:
            st.session_state.image = image_to_process
            st.image(
                image_to_process,
                caption="Uploaded photo",
                use_column_width=True,
            )

            with st.spinner("Analyzing photo & recipe ingredients..."):
                # Run CNN image classification and calculate entropy
                cnn_class, cnn_confidence, margin, entropy = run_cnn(
                    image_to_process, cnn_model, idx_to_class
                )

                # Non-food guardrail check
                is_non_food = (
                    cnn_confidence < MIN_CONFIDENCE
                    or margin < MIN_MARGIN
                    or entropy > MAX_ENTROPY
                )

                if is_non_food:
                    st.markdown(
                        f"""<div style="background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.35); border-radius: 14px; padding: 16px; margin-top: 14px; text-align: center;">
<div style="font-size: 1.8rem; margin-bottom: 4px;">🍽️❓</div>
<div style="color: #F87171; font-weight: 700; font-size: 1rem; margin-bottom: 4px;">No Supported Gulf Dish Found</div>
<div style="color: #A39682; font-size: 0.82rem; line-height: 1.45;">
Please upload a clear, focused photo of a traditional Gulf dish.
</div>
</div>""",
                        unsafe_allow_html=True,
                    )
                    st.write("")
                    st.button(
                        "🔄 Upload Another Photo",
                        on_click=reset,
                        use_container_width=True,
                    )
                    st.stop()

                # Multi-tier decision gating logic
                triggered = (
                    cnn_confidence < CONFIDENCE_THRESHOLD
                    or cnn_class in TRIGGER_SET
                    or cnn_class in WRAP_TRIGGER_SET
                )

                st.session_state.cnn_class = cnn_class
                st.session_state.cnn_confidence = cnn_confidence
                st.session_state.triggered = triggered

                if not triggered:
                    # High confidence and non-confusable dish: advance directly to portion sizing
                    st.session_state.final_dish = cnn_class
                    st.session_state.tier_used = "CNN only"
                    st.session_state.stage = "select_portion"
                    st.rerun()
                else:
                    # Ambiguous or confusable dish: trigger YOLO detection and candidate verification
                    candidates = get_candidate_group(cnn_class)
                    run_yolo_here = (cnn_class in YOLO_FEATURE_MAP) or (
                        cnn_class == "03_biryani"
                    )
                    yolo_suggestion, gate_status = None, None
                    if run_yolo_here:
                        detections = run_yolov8(image_to_process, yolo_model)
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


# ============================================================================
# 10. STAGE 2: DISH CONFIRMATION & SELECTION
# ============================================================================

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
            f"""
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 0.8rem;">
                <div style="font-size: 1.45rem; font-weight: 800; color: #FBF8F1; letter-spacing: -0.02em;">
                    Initial Match: <span style="color: #E5A93B;">{display_name(cnn_class)}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        render_confidence_bar(cnn_conf)

        reason = get_group_reason(cnn_class)
        if reason:
            st.markdown(
                f"""
                <div class="verify-callout">
                    <span style="font-size: 1.1rem; line-height: 1;">🔍</span>
                    <span style="color: #A39682; font-size: 0.84rem; line-height: 1.4;">{reason}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if yolo_suggestion:
            st.info(
                f"✨ Visual ingredient inspection detected: **{display_name(yolo_suggestion)}**."
            )

        # Pre-select YOLO suggestion if available; otherwise fallback to top CNN prediction
        default_choice = yolo_suggestion if yolo_suggestion else cnn_class
        default_idx = (
            candidates.index(default_choice)
            if default_choice in candidates
            else 0
        )

        st.markdown(
            '<p style="font-size: 0.85rem; font-weight: 700; color: #FBF8F1; margin: 12px 0 4px 0;">Select your matching dish:</p>',
            unsafe_allow_html=True,
        )

        choice = st.radio(
            "Select the matching dish:",
            options=candidates,
            format_func=lambda x: f"🍲 {display_name(x)}",
            index=default_idx,
            label_visibility="collapsed",
        )

        st.write("")
        if st.button("Confirm Dish & Continue →", type="primary"):
            st.session_state.final_dish = choice
            st.session_state.stage = "select_portion"
            st.rerun()


# ============================================================================
# 11. STAGE 3: PORTION SIZE SELECTION
# ============================================================================

elif st.session_state.stage == "select_portion":
    render_stepper("select_portion", st.session_state.triggered)

    with st.container(border=True):
        st.image(
            st.session_state.image,
            caption="Uploaded photo",
            use_column_width=True,
        )

        st.markdown(
            f"""
            <div style="font-size: 1.6rem; font-weight: 800; color: #FBF8F1; margin: 0.8rem 0 0.2rem 0;">
                {display_name(st.session_state.final_dish)}
            </div>
            <p style="color: var(--gb-muted); font-size: 0.86rem; margin-bottom: 1.2rem;">
                Choose your serving size to calculate authentic nutrition values:
            </p>
            """,
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🌱\n\nSmall\n\n(~250g)", key="btn_s", use_container_width=True):
                st.session_state.portion_size = "S"
                st.session_state.stage = "result"
                st.rerun()

        with col2:
            if st.button("🍽️\n\nMedium\n\n(~400g)", key="btn_m", use_container_width=True):
                st.session_state.portion_size = "M"
                st.session_state.stage = "result"
                st.rerun()

        with col3:
            if st.button("👑\n\nLarge\n\n(~550g)", key="btn_l", use_container_width=True):
                st.session_state.portion_size = "L"
                st.session_state.stage = "result"
                st.rerun()


# ============================================================================
# 12. STAGE 4: RESULTS & NUTRITION BREAKDOWN
# ============================================================================

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

        # Calorie highlight & macronutrient pills
        render_calorie_hero(lo, hi)
        render_macro_cards(
            nutrition["protein_g"], nutrition["carbs_g"], nutrition["fat_g"]
        )

        if nutrition["missing_ingredients"]:
            st.warning(
                f"Missing standard data for: {', '.join(nutrition['missing_ingredients'])}."
            )

        st.markdown('<div class="gb-divider"></div>', unsafe_allow_html=True)

        # Allow user correction without restarting the session
        # Detailed breakdown of the multi-tier AI inference path
        st.markdown('<div class="gb-divider"></div>', unsafe_allow_html=True)

tab_correct, tab_tech = st.tabs(["✏️ Change Dish", "⚙️ Pipeline Breakdown"])

with tab_correct:
    all_dishes = sorted(DISH_RECIPES.keys(), key=display_name)
    corrected = st.selectbox(
        "Select correct dish:",
        options=all_dishes,
        format_func=display_name,
        index=all_dishes.index(dish) if dish in all_dishes else 0,
        label_visibility="collapsed"
    )
    if st.button("Update Dish", type="primary", use_container_width=True):
        st.session_state.final_dish = corrected
        st.session_state.tier_used = "User correction"
        st.rerun()

with tab_tech:
    yolo_row = (
        f'<div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.06); padding-bottom: 8px;"><span style="color: var(--gb-muted); font-size: 0.84rem;">YOLOv8 Feature</span><span style="color: #FBF8F1; font-weight: 600; font-size: 0.88rem;">{display_name(st.session_state.yolo_suggestion)}</span></div>'
        if st.session_state.get("yolo_suggestion")
        else ""
    )

    st.markdown(
        f"""<div style="display: flex; flex-direction: column; gap: 10px; padding: 6px 0;">
<div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.06); padding-bottom: 8px;">
<span style="color: var(--gb-muted); font-size: 0.84rem;">CNN Classifier</span>
<span style="color: #FBF8F1; font-weight: 600; font-size: 0.88rem;">{display_name(st.session_state.cnn_class)} <span style="color: #E5A93B; font-family: 'JetBrains Mono', monospace;">({st.session_state.cnn_confidence:.0%})</span></span>
</div>
{yolo_row}
<div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.06); padding-bottom: 8px;">
<span style="color: var(--gb-muted); font-size: 0.84rem;">Confirmed Match</span>
<span style="color: #FBF8F1; font-weight: 700; font-size: 0.88rem;">{display_name(dish)}</span>
</div>
<div style="display: flex; justify-content: space-between; align-items: center; padding-top: 2px;">
<span style="color: var(--gb-muted); font-size: 0.84rem;">Execution Path</span>
<span class="tech-pill">{st.session_state.tier_used}</span>
</div>
</div>""",
        unsafe_allow_html=True,
    )

    st.write("")
    st.button("📸 Analyze Another Photo", on_click=reset)
