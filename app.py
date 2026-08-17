"""
GulfBite — Smart Gulf Cuisine Nutrition Assistant (Mobile Light-Gold Edition)
-----------------------------------------------------------------------------
Identifies authentic Gulf dishes using a multi-tiered pipeline:
1. MobileNetV2 (CNN) classification for initial dish match & confidence scoring[cite: 1].
2. Out-of-distribution / Non-food rejection via margin and entropy checks[cite: 1].
3. YOLOv8 feature detection for visually ambiguous dishes (e.g., loomi in Machboos)[cite: 1].
4. Portion-based authentic macro and calorie estimation[cite: 1].

Required files in the `models/` directory:
- models/MobileNetV2_best.keras[cite: 1]
- models/class_indices.json[cite: 1]
- models/yolov8_ingredient_detector-4/weights/best.pt[cite: 1]
- models/ingredient_nutrition_cache.json[cite: 1]
"""

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps
import streamlit as st

# Force CPU inference for stability and suppress TensorFlow verbose logging[cite: 1]
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"[cite: 1]
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"[cite: 1]

# Register HEIC/HEIF image support for mobile uploads[cite: 1]
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

# ============================================================================
# 1. CONFIGURATION & CONSTANTS
# ============================================================================

MODELS_DIR = "models"[cite: 1]
CNN_MODEL_PATH = os.path.join(MODELS_DIR, "MobileNetV2_best.keras")[cite: 1]
CLASS_INDICES_PATH = os.path.join(MODELS_DIR, "class_indices.json")[cite: 1]
YOLO_WEIGHTS_PATH = os.path.join(
    MODELS_DIR, "yolov8_ingredient_detector-4", "weights", "best.pt"
)[cite: 1]
INGREDIENT_CACHE_PATH = os.path.join(
    MODELS_DIR, "ingredient_nutrition_cache.json"
)[cite: 1]

# Sets of dishes that frequently confuse the CNN and require verification[cite: 1]
TRIGGER_SET = {
    "07_ouzi",
    "01_machboos",
    "09_jisheed",
    "02_kabsa",
    "03_biryani",
    "06_saloona",
}[cite: 1]
WRAP_TRIGGER_SET = {"10_shawarma", "11_falafel_wrap"}[cite: 1]

# Thresholds for AI confidence and non-food detection[cite: 1]
CONFIDENCE_THRESHOLD = 0.70[cite: 1]
MIN_CONFIDENCE = 0.50[cite: 1]
MIN_MARGIN = 0.15[cite: 1]
MAX_ENTROPY = 2.50[cite: 1]

# Feature-to-dish mappings for YOLO validation[cite: 1]
YOLO_FEATURE_MAP = {
    "01_machboos": "loomi",
    "07_ouzi": "whole_shank",
    "08_samak_mashwi": "whole_fish",
    "02_kabsa": "whole_chicken_piece",
    "10_shawarma": "shawarma_meat",
    "11_falafel_wrap": "falafel_ball",
}[cite: 1]
FEATURE_TO_DISH = {v: k for k, v in YOLO_FEATURE_MAP.items()}[cite: 1]

# Related dish groups for fallback confirmation[cite: 1]
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
}[cite: 1]

GROUP_REASONS = {
    "rice_cluster": "Rice dishes like Machboos, Kabsa, and Biryani share aromatic spice bases, so we double-check.",
    "wrap_cluster": "Wrapped dishes hide their core filling, so we double-check with you.",
}[cite: 1]


def get_group_reason(cnn_class: str) -> Optional[str]:
    """Retrieve an intuitive explanation for why the app is asking for confirmation."""
    for group_name, group_set in CONFUSION_GROUPS.items():[cite: 1]
        if cnn_class in group_set:[cite: 1]
            return GROUP_REASONS.get(group_name)[cite: 1]
    return None[cite: 1]


FEATURE_RELIABILITY = {
    "loomi": {"status": "reliable"},
    "whole_chicken_piece": {"status": "unreliable"},
    "whole_shank": {"status": "insufficient_evidence"},
    "whole_fish": {"status": "insufficient_evidence"},
    "shawarma_meat": {"status": "reliable"},
    "falafel_ball": {"status": "reliable"},
}[cite: 1]

# Base recipes for Medium portion (grams)[cite: 1]
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
}[cite: 1]

DISH_BLURBS = {
    "01_machboos": "A fragrant spiced rice plate with meat or chicken, infused with black dried lime (loomi).",
    "02_kabsa": "Saudi Arabia's signature spiced rice dish finished with saffron, tomatoes, and tender meat.",
    "03_biryani": "Richly layered basmati rice spiced with cloves, cardamoms, and marinated chicken.",
    "04_harees": "Slow-cooked wheat and shredded meat porridge seasoned with aromatic ghee.",
    "05_thareed": "Crisp thin flatbread layered with hearty lamb and slow-simmered vegetable broth.",
    "06_saloona": "A traditional comforting Gulf stew simmered with seasonal vegetables and spices.",
    "07_ouzi": "Spiced spiced rice loaded with slow-roasted tender lamb and toasted golden nuts.",
    "08_samak_mashwi": "Locally caught fish marinated in regional spices and flame-grilled over open coals.",
    "09_jisheed": "Flaked Gulf fish seasoned with dried lime, turmeric, and served over steamed rice.",
    "10_shawarma": "Thinly shaved marinated chicken wrapped in warm pita with garlic toum sauce.",
    "11_falafel_wrap": "Crisp golden chickpea falafels with fresh salad and silky tahini sauce in a warm wrap.",
    "12_falafel": "Deep-fried seasoned chickpea fritters with garlic, parsley, and roasted coriander.",
    "13_samboosa": "Crispy golden fried pastry triangles filled with spiced minced meat or vegetables.",
    "14_mutabbaq": "Folded pan-fried thin pastry stuffed with spiced meat, eggs, and scallions.",
    "15_hummus": "Silky blended chickpeas with tahini, lemon juice, and extra virgin olive oil.",
    "16_fattoush": "Crunchy garden salad tossed with toasted pita crisps, pomegranate molasses, and sumac.",
    "17_tabbouleh": "Finely chopped fresh parsley salad with bulgur, tomatoes, mint, and lemon olive dressing.",
    "18_foul_medames": "Slow-cooked creamy fava beans dressed with cumin, garlic, and cold-pressed olive oil.",
    "19_shakshuka": "Gently poached farm eggs in a skillet of spiced tomato, bell pepper, and cumin sauce.",
    "20_balaleet": "Sweet cardamom-saffron vermicelli noodles crowned with a savoury spiced omelette.",
    "21_khameer": "Fluffy yeast leavened bread dusted with sesame seeds and dates.",
    "22_chebab": "Golden Emirati pancakes scented with cardamom, saffron, and drizzled with honey.",
    "23_luqaimat": "Crispy golden fried dough puffs drizzled generously with local date molasses.",
    "24_knafeh": "Warm melted akkawi cheese wrapped in shredded crisp filo pastry soaked in orange blossom syrup.",
    "25_karak_chai": "Rich black tea slow-simmered with evaporated milk and crushed cardamom pods.",
}[cite: 1]

PORTION_MULTIPLIERS = {"S": 0.7, "M": 1.0, "L": 1.4}[cite: 1]
PORTION_LABELS = {"S": "Small", "M": "Medium", "L": "Large"}[cite: 1]
CALORIE_RANGE_PCT = 0.15[cite: 1]


def display_name(cls: str) -> str:
    """Clean technical class names into title format (e.g. '01_machboos' -> 'Machboos')."""
    return cls.split("_", 1)[1].replace("_", " ").title()[cite: 1]


DISH_CATEGORIES = {
    "🍚 Rice Dishes": [
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
    "🌯 Wraps & Bites": [
        "10_shawarma",
        "11_falafel_wrap",
        "12_falafel",
        "13_samboosa",
        "14_mutabbaq",
    ],
    "🫓 Breakfast & Breads": [
        "18_foul_medames",
        "19_shakshuka",
        "20_balaleet",
        "21_khameer",
        "22_chebab",
    ],
    "🥗 Fresh Salads & Dips": ["15_hummus", "16_fattoush", "17_tabbouleh"],
    "🍯 Sweets & Tea": ["23_luqaimat", "24_knafeh", "25_karak_chai"],
}[cite: 1]


def get_candidate_group(cnn_class: str) -> set:
    for group in CONFUSION_GROUPS.values():[cite: 1]
        if cnn_class in group:[cite: 1]
            return group[cite: 1]
    return {cnn_class}[cite: 1]


# ============================================================================
# 2. MODEL LOADING & INFERENCE
# ============================================================================

@st.cache_resource
def load_models():
    import tensorflow as tf[cite: 1]
    from ultralytics import YOLO[cite: 1]

    missing = [
        p
        for p in (
            CNN_MODEL_PATH,
            CLASS_INDICES_PATH,
            YOLO_WEIGHTS_PATH,
            INGREDIENT_CACHE_PATH,
        )
        if not os.path.exists(p)
    ][cite: 1]
    if missing:[cite: 1]
        raise FileNotFoundError(
            "Missing model file(s):\n"
            + "\n".join(missing)
            + "\n\nPlease ensure model weights are located in the models/ directory."
        )[cite: 1]

    cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)[cite: 1]
    with open(CLASS_INDICES_PATH) as f:[cite: 1]
        class_indices = json.load(f)[cite: 1]
    idx_to_class = {v: k for k, v in class_indices.items()}[cite: 1]

    yolo_model = YOLO(YOLO_WEIGHTS_PATH)[cite: 1]

    with open(INGREDIENT_CACHE_PATH) as f:[cite: 1]
        ingredient_cache = json.load(f)[cite: 1]

    return cnn_model, idx_to_class, yolo_model, ingredient_cache[cite: 1]


def run_cnn(pil_image, model, idx_to_class, img_size=(224, 224)):
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input[cite: 1]

    img = pil_image.convert("RGB").resize(img_size)[cite: 1]
    arr = np.array(img).astype("float32")[cite: 1]
    arr = preprocess_input(arr)[cite: 1]
    arr = np.expand_dims(arr, axis=0)[cite: 1]

    preds = model.predict(arr, verbose=0)[0][cite: 1]

    sorted_indices = np.argsort(preds)[::-1][cite: 1]
    top_idx = int(sorted_indices[0])[cite: 1]
    second_idx = int(sorted_indices[1])[cite: 1]

    confidence = float(preds[top_idx])[cite: 1]
    second_confidence = float(preds[second_idx])[cite: 1]
    margin = confidence - second_confidence[cite: 1]

    eps = 1e-12[cite: 1]
    entropy = -np.sum(preds * np.log(preds + eps))[cite: 1]

    predicted_class = idx_to_class[top_idx][cite: 1]
    return predicted_class, confidence, margin, entropy[cite: 1]


def run_yolov8(pil_image, yolo_model, conf_threshold=0.25):
    results = yolo_model.predict(
        np.array(pil_image.convert("RGB")), conf=conf_threshold, verbose=False
    )[cite: 1]
    detections = [][cite: 1]
    r = results[0][cite: 1]
    for box in r.boxes:[cite: 1]
        cls_id = int(box.cls[0])[cite: 1]
        cls_name = yolo_model.names[cls_id][cite: 1]
        box_conf = float(box.conf[0])[cite: 1]
        detections.append((cls_name, box_conf))[cite: 1]
    return detections[cite: 1]


def map_detections_to_suggestion(detections, candidates):
    if not detections:[cite: 1]
        return None, None, "no_detection"[cite: 1]
    valid = [
        (FEATURE_TO_DISH[feat], conf, feat)
        for feat, conf in detections
        if feat in FEATURE_TO_DISH and FEATURE_TO_DISH[feat] in candidates
    ][cite: 1]
    if not valid:[cite: 1]
        return None, None, "no_detection"[cite: 1]
    valid.sort(key=lambda x: x[1], reverse=True)[cite: 1]
    dish, conf, feature = valid[0][cite: 1]
    status = FEATURE_RELIABILITY.get(
        feature, {"status": "insufficient_evidence"}
    )["status"][cite: 1]
    gated = (dish, conf) if status == "reliable" else None[cite: 1]
    return (dish, conf), gated, status[cite: 1]


def estimate_nutrition(dish_class, portion_size, ingredient_cache):
    multiplier = PORTION_MULTIPLIERS[portion_size][cite: 1]
    totals = {"calories": 0.0, "protein": 0.0, "carbs": 0.0, "fat": 0.0}[cite: 1]
    missing = [][cite: 1]
    for ingredient_key, base_grams in DISH_RECIPES[dish_class]:[cite: 1]
        info = ingredient_cache.get(ingredient_key)[cite: 1]
        if info is None or info.get("source") == "NONE":[cite: 1]
            missing.append(ingredient_key)[cite: 1]
            continue[cite: 1]
        grams = base_grams * multiplier[cite: 1]
        for macro in totals:[cite: 1]
            totals[macro] += info[macro] * (grams / 100)[cite: 1]
    cal_low = totals["calories"] * (1 - CALORIE_RANGE_PCT)[cite: 1]
    cal_high = totals["calories"] * (1 + CALORIE_RANGE_PCT)[cite: 1]
    return {
        "calories_range": (round(cal_low), round(cal_high)),
        "protein_g": round(totals["protein"], 1),
        "carbs_g": round(totals["carbs"], 1),
        "fat_g": round(totals["fat"], 1),
        "missing_ingredients": missing,
    }[cite: 1]


# ============================================================================
# 3. MODERN LIGHT MOBILE THEME (WARM GOLD PALETTE)
# ============================================================================

def inject_theme():
    st.markdown(
        """<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Outfit:wght@500;600;700;800;900&family=JetBrains+Mono:wght@500;700&display=swap');

:root {
    --app-bg: #F9F7F2;
    --card-bg: #FFFFFF;
    --card-border: rgba(229, 169, 59, 0.18);
    --gold-primary: #E5A93B;
    --gold-dark: #C28416;
    --gold-soft: #FDF6E9;
    --gold-dim: #996E21;
    --text-dark: #1E1B16;
    --text-muted: #7A7468;
    --chip-bg: #F4EFE6;
}

html, body, [class*="css"] { 
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: var(--text-dark);
}

/* Backdrop */
.stApp { 
    background-color: var(--app-bg);
    background-image: 
        radial-gradient(circle at 50% -10%, #FBF1DC 0%, transparent 65%),
        radial-gradient(circle at 100% 100%, #FAF3E3 0%, transparent 50%);
    color: var(--text-dark);
}

#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { 
    max-width: 440px !important; 
    padding-top: 1.5rem !important; 
    padding-bottom: 3.5rem !important; 
}

/* Mobile Frame Card Container */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--card-bg) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 32px !important;
    padding: 1.4rem !important;
    box-shadow: 0 20px 45px -12px rgba(229, 169, 59, 0.16), 0 2px 10px rgba(0,0,0,0.02) !important;
}

/* --- Fixed Tab Bar (Pill Switcher) --- */
div[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: #EBE4D5 !important;
    border-radius: 999px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: none !important;
    display: flex !important;
    width: 100% !important;
}

div[data-testid="stTabs"] [data-baseweb="tab-border"],
div[data-testid="stTabs"] [data-baseweb="tab-highlight"] { 
    display: none !important; 
}

div[data-testid="stTabs"] [data-baseweb="tab"] {
    flex: 1 !important;
    border-radius: 999px !important;
    height: 38px !important;
    color: var(--text-muted) !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
    background: transparent !important;
    border: none !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    transition: all 0.2s ease !important;
}

div[data-testid="stTabs"] [aria-selected="true"] {
    background: #FFFFFF !important;
    color: var(--text-dark) !important;
    box-shadow: 0 3px 12px rgba(0,0,0,0.08) !important;
}

/* --- Category Capsule Filter Pills --- */
div[data-testid="stRadio"] > div[role="radiogroup"] {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: wrap !important;
    gap: 8px !important;
    background: transparent !important;
    border: none !important;
    padding: 4px 0 !important;
}

div[data-testid="stRadio"] label[data-baseweb="radio"] {
    background: #F4EDE0 !important;
    border: 1px solid #E4D8C1 !important;
    border-radius: 999px !important;
    padding: 7px 14px !important;
    margin: 0 !important;
    cursor: pointer !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    min-width: auto !important;
    width: auto !important;
}

div[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child {
    display: none !important;
}

div[data-testid="stRadio"] label[data-baseweb="radio"]:hover {
    background: #EAE0CD !important;
    border-color: var(--gold-primary) !important;
    transform: translateY(-1px);
}

div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
    background: linear-gradient(135deg, #F3C36A 0%, #E5A93B 100%) !important;
    border-color: #D19428 !important;
    box-shadow: 0 4px 12px rgba(229, 169, 59, 0.3) !important;
}

div[data-testid="stRadio"] label[data-baseweb="radio"] p,
div[data-testid="stRadio"] label[data-baseweb="radio"] span {
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    color: var(--text-dark) !important;
    white-space: nowrap !important;
    margin: 0 !important;
    line-height: 1.2 !important;
}

div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) p,
div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) span {
    color: #1A1305 !important;
}

/* --- Clean Modern Dropzone & Button --- */
[data-testid="stFileUploaderDropzone"] {
    background: #FAF8F3 !important;
    border: 2px dashed #E2D5B8 !important;
    border-radius: 24px !important;
    padding: 1.8rem 1.2rem !important;
    transition: all 0.25s ease !important;
}

[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--gold-primary) !important;
    background: #FDF9EE !important;
    box-shadow: 0 8px 24px rgba(229, 169, 59, 0.14) !important;
}

[data-testid="stFileUploaderDropzone"] svg {
    fill: var(--gold-primary) !important;
    color: var(--gold-primary) !important;
}

[data-testid="stFileUploaderDropzoneInstructions"] > div > span {
    font-family: 'Outfit', sans-serif !important;
    font-size: 1.05rem !important;
    font-weight: 800 !important;
    color: var(--text-dark) !important;
}

[data-testid="stFileUploaderDropzoneInstructions"] > div > small {
    color: #8C8476 !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
}

[data-testid="stFileUploader"] button {
    background: linear-gradient(135deg, #F3C36A 0%, #E5A93B 100%) !important;
    border: none !important;
    color: #1A1305 !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 800 !important;
    font-size: 0.88rem !important;
    border-radius: 999px !important;
    padding: 0.65rem 1.4rem !important;
    height: auto !important;
    min-height: 42px !important;
    width: auto !important;
    min-width: 120px !important;
    white-space: nowrap !important;
    box-shadow: 0 4px 14px rgba(229, 169, 59, 0.32) !important;
    transition: all 0.2s ease !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
}

[data-testid="stFileUploader"] button p {
    white-space: nowrap !important;
    margin: 0 !important;
    line-height: 1 !important;
}

[data-testid="stFileUploader"] button:hover {
    background: linear-gradient(135deg, #FEDD97 0%, #F0B547 100%) !important;
    box-shadow: 0 6px 18px rgba(229, 169, 59, 0.45) !important;
    transform: translateY(-2px);
}

/* Global Button Styles */
div.stButton > button {
    font-family: 'Outfit', sans-serif;
    border-radius: 999px;
    border: none;
    background: var(--chip-bg);
    color: var(--text-dark);
    font-weight: 700;
    font-size: 0.95rem;
    padding: 0.85rem 1.5rem;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    width: 100%;
}

div.stButton > button:hover {
    background: #EAE3D5;
    transform: translateY(-1px);
}

div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #F3C36A 0%, #E5A93B 55%, #CF8E1D 100%) !important;
    color: #1A1305 !important;
    font-weight: 800;
    box-shadow: 0 10px 24px rgba(229, 169, 59, 0.35) !important;
}

div.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #FDD68A 0%, #F0B547 55%, #DE9A26 100%) !important;
    box-shadow: 0 12px 28px rgba(229, 169, 59, 0.45) !important;
    transform: translateY(-2px);
}

/* --- Full-Width Portion Selection Cards --- */
.portion-card-group div[data-testid="stRadio"] > div[role="radiogroup"] {
    display: flex !important;
    flex-direction: column !important;
    flex-wrap: nowrap !important;
    gap: 10px !important;
    width: 100% !important;
}

.portion-card-group div[data-testid="stRadio"] label[data-baseweb="radio"] {
    width: 100% !important;
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
    background: #FAF8F3 !important;
    border: 1.5px solid #EBE2CF !important;
    border-radius: 20px !important;
    padding: 14px 18px !important;
    margin: 0 !important;
    box-sizing: border-box !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

.portion-card-group div[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child {
    display: none !important;
}

.portion-card-group div[data-testid="stRadio"] label[data-baseweb="radio"]:hover {
    background: #FDF9EE !important;
    border-color: var(--gold-primary) !important;
    transform: translateY(-1px);
}

.portion-card-group div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
    background: linear-gradient(135deg, #FDF7EC 0%, #FAF0D8 100%) !important;
    border-color: var(--gold-primary) !important;
    box-shadow: 0 4px 14px rgba(229, 169, 59, 0.25) !important;
}

.portion-card-group div[data-testid="stRadio"] label[data-baseweb="radio"] span p {
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 800 !important;
    color: var(--text-dark) !important;
    margin: 0 !important;
    white-space: nowrap !important;
}

.portion-card-group div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) span p {
    color: #1A1305 !important;
}

/* Verification Alert Callout */
.verify-callout {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    background: #FDF9EE;
    border: 1px solid #F5E5BE;
    border-left: 4px solid var(--gold-primary);
    border-radius: 16px;
    padding: 12px 14px;
    margin: 12px 0 16px 0;
}

.ingredient-badge {
    display: flex;
    align-items: center;
    gap: 8px;
    background: linear-gradient(135deg, #FDF9EE 0%, #FAF2DC 100%);
    border: 1px solid #EFE0BD;
    border-radius: 16px;
    padding: 10px 14px;
    margin: 8px 0 12px 0;
    color: #1E1B16;
    font-size: 0.86rem;
    font-weight: 600;
}

.tech-pill {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 0.76rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    background: #ECFDF5;
    color: #059669;
    border: 1px solid #A7F3D0;
}
</style>""",
        unsafe_allow_html=True,
    )


# ============================================================================
# 4. APP NAVIGATION & MOBILE HEADER
# ============================================================================

def render_header():
    """Render sleek mobile navigation header with crisp vector icons."""
    st.markdown(
        """<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.4rem; padding: 2px 0;">
<div style="width: 44px; height: 44px; border-radius: 50%; background: #FFFFFF; box-shadow: 0 4px 14px rgba(0, 0, 0, 0.04), 0 1px 3px rgba(0, 0, 0, 0.02); display: flex; align-items: center; justify-content: center; border: 1px solid #ECE6DB; cursor: pointer;">
<svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M4 7H20M4 12H20M4 17H14" stroke="#1E1B16" stroke-width="2.2" stroke-linecap="round"/>
</svg>
</div>
<div style="display: flex; gap: 10px; align-items: center;">
<div style="width: 44px; height: 44px; border-radius: 50%; background: #FFFFFF; box-shadow: 0 4px 14px rgba(0, 0, 0, 0.04), 0 1px 3px rgba(0, 0, 0, 0.02); display: flex; align-items: center; justify-content: center; border: 1px solid #ECE6DB; cursor: pointer; position: relative;">
<svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9" stroke="#E5A93B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
<path d="M13.73 21a2 2 0 0 1-3.46 0" stroke="#E5A93B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
<span style="position: absolute; top: 11px; right: 11px; width: 7px; height: 7px; background: #FF5A1F; border-radius: 50%; border: 1.5px solid #FFFFFF;"></span>
</div>
<div style="width: 44px; height: 44px; border-radius: 50%; background: linear-gradient(135deg, #F3C36A 0%, #E5A93B 100%); display: flex; align-items: center; justify-content: center; color: #1A1305; font-family: 'Outfit', sans-serif; font-weight: 900; font-size: 0.95rem; letter-spacing: 0.02em; box-shadow: 0 6px 16px rgba(229, 169, 59, 0.35);">
GB
</div>
</div>
</div>
<div style="margin-bottom: 1.4rem;">
<h1 style="font-family: 'Outfit', sans-serif; font-size: 2.25rem; font-weight: 900; line-height: 1.15; color: #1E1B16; margin: 0; letter-spacing: -0.03em;">
<span style="color: #E5A93B;">GulfBite</span><br>AI Nutrition
</h1>
<p style="color: #8F887C; font-size: 0.88rem; font-weight: 500; margin-top: 6px;">Authentic Gulf Cuisine Recognition & Macro Insights</p>
</div>""",
        unsafe_allow_html=True,
    )


def render_stepper(current_stage: str, triggered: bool):
    """Render mobile progress chips in warm gold."""
    steps = [("upload", "Upload")]
    if triggered:
        steps.append(("confirm_dish", "Confirm"))
    steps.append(("select_portion", "Portion"))
    steps.append(("result", "Macros"))

    keys = [s[0] for s in steps]
    active_idx = keys.index(current_stage) if current_stage in keys else 0

    html = [
        '<div style="display: flex; align-items: center; justify-content: space-between; margin: 0.4rem 0 1.2rem 0; padding: 0 2px;">'
    ]
    for i, (key, label) in enumerate(steps):
        is_done = i < active_idx
        is_active = i == active_idx

        bg = "linear-gradient(135deg, #F3C36A, #E5A93B)" if (is_done or is_active) else "#EFEAE0"
        color = "#1B1304" if (is_done or is_active) else "#A39E96"
        text_color = "#1E1B16" if (is_done or is_active) else "#A39E96"
        font_weight = "800" if is_active else "600"
        glow = "box-shadow: 0 4px 12px rgba(229, 169, 59, 0.35);" if is_active else ""

        html.append(f"""<div style="display: flex; align-items: center; gap: 6px;">
<div style="width: 24px; height: 24px; border-radius: 50%; background: {bg}; color: {color}; display: flex; align-items: center; justify-content: center; font-family: 'Outfit', sans-serif; font-size: 0.75rem; font-weight: 800; {glow}">{i + 1}</div>
<span style="font-family: 'Outfit', sans-serif; color: {text_color}; font-weight: {font_weight}; font-size: 0.84rem;">{label}</span>
</div>""")
        if i < len(steps) - 1:
            line_color = "#E5A93B" if is_done else "#EFEAE0"
            html.append(
                f'<div style="flex: 1; min-width: 10px; height: 2.5px; background: {line_color}; margin: 0 6px; border-radius: 2px;"></div>'
            )

    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def render_calorie_hero(lo: int, hi: int):
    """Render gold-accented calorie highlight banner."""
    st.markdown(
        f"""<div style="
            background: linear-gradient(135deg, #FDF9EE 0%, #FAF3DE 100%);
            border: 1.5px solid #F3E0B5;
            border-radius: 24px;
            padding: 1.2rem 1.4rem;
            margin: 1rem 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        ">
            <div>
                <div style="font-size: 0.75rem; font-weight: 800; color: #D19428; text-transform: uppercase; letter-spacing: 0.05em;">Estimated Energy</div>
                <div style="font-family: 'Outfit', sans-serif; font-size: 2.3rem; font-weight: 900; color: #1E1B16; line-height: 1.1; margin-top: 2px;">
                    {lo}&ndash;{hi} <span style="font-size: 1.05rem; font-weight: 600; color: #8F887C;">kcal</span>
                </div>
            </div>
            <div style="background: #FFFFFF; padding: 8px 16px; border-radius: 999px; border: 1.5px solid #E5A93B; color: #B37B1B; font-weight: 800; font-size: 0.92rem; box-shadow: 0 4px 10px rgba(229,169,59,0.15);">
                ✨ Validated
            </div>
        </div>""",
        unsafe_allow_html=True,
    )


def render_macro_cards(protein_g, carbs_g, fat_g):
    """Render 4 responsive macro micro-cards without column squishing."""
    fiber_estimate = round(carbs_g * 0.12, 1)
    
    st.markdown(
        f"""<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; margin: 0.8rem 0 1.2rem 0;">
    <div style="background: #FAF8F3; border: 1px solid #EBE2CF; border-radius: 18px; padding: 10px 4px; text-align: center;">
        <div style="font-size: 0.72rem; font-weight: 700; color: #8F887C; margin-bottom: 2px;">🤍 Protein</div>
        <div style="font-family: 'Outfit', sans-serif; color: #1E1B16; font-size: 1.15rem; font-weight: 800;">{protein_g}g</div>
    </div>
    <div style="background: #FAF8F3; border: 1px solid #EBE2CF; border-radius: 18px; padding: 10px 4px; text-align: center;">
        <div style="font-size: 0.72rem; font-weight: 700; color: #8F887C; margin-bottom: 2px;">🌾 Carbs</div>
        <div style="font-family: 'Outfit', sans-serif; color: #1E1B16; font-size: 1.15rem; font-weight: 800;">{carbs_g}g</div>
    </div>
    <div style="background: #FAF8F3; border: 1px solid #EBE2CF; border-radius: 18px; padding: 10px 4px; text-align: center;">
        <div style="font-size: 0.72rem; font-weight: 700; color: #8F887C; margin-bottom: 2px;">🧈 Fat</div>
        <div style="font-family: 'Outfit', sans-serif; color: #1E1B16; font-size: 1.15rem; font-weight: 800;">{fat_g}g</div>
    </div>
    <div style="background: #FAF8F3; border: 1px solid #EBE2CF; border-radius: 18px; padding: 10px 4px; text-align: center;">
        <div style="font-size: 0.72rem; font-weight: 700; color: #8F887C; margin-bottom: 2px;">🍃 Fiber</div>
        <div style="font-family: 'Outfit', sans-serif; color: #1E1B16; font-size: 1.15rem; font-weight: 800;">{fiber_estimate}g</div>
    </div>
</div>""",
        unsafe_allow_html=True,
    )


def render_confidence_bar(confidence):
    pct = confidence * 100
    st.markdown(
        f"""<div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 0.6rem;">
            <span style="font-size: 0.82rem; color: #8F887C; font-weight: 600;">Match Accuracy</span>
            <span style="font-family: 'Outfit', sans-serif; font-size: 0.92rem; font-weight: 800; color: #E5A93B;">{pct:.0f}%</span>
        </div>
        <div style="height: 7px; border-radius: 999px; background: #EFEAE0; overflow: hidden; margin: 0.35rem 0 0.8rem 0;">
            <div style="width: {pct:.1f}%; height: 100%; background: linear-gradient(90deg, #F3C36A, #E5A93B); border-radius: 999px;"></div>
        </div>""",
        unsafe_allow_html=True,
    )


def render_interactive_dish_explorer():
    selected_category = st.radio(
        "Filter by category:",
        options=list(DISH_CATEGORIES.keys()),
        index=0,
        horizontal=True,
        label_visibility="collapsed",
    )

    if not selected_category:
        selected_category = "🍚 Rice Dishes"

    category_dishes = DISH_CATEGORIES[selected_category]

    st.markdown(
        '<div style="margin-top: 14px; margin-bottom: 6px; font-size: 0.78rem; font-weight: 800; color: #8F887C; text-transform: uppercase; letter-spacing: 0.05em;">Choose Recipe</div>',
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
                background: #FAF8F3;
                border: 1px solid #EBE2CF;
                border-left: 4px solid #E5A93B;
                border-radius: 20px;
                padding: 14px 16px;
                margin-top: 10px;
            ">
                <div style="font-family: 'Outfit', sans-serif; font-weight: 800; color: #1E1B16; font-size: 1.05rem; margin-bottom: 4px;">{display_name(selected_dish)}</div>
                <div style="color: #736C61; font-size: 0.84rem; line-height: 1.45;">{blurb}</div>
            </div>""",
            unsafe_allow_html=True,
        )


# ============================================================================
# 5. STREAMLIT APP STATE SETUP
# ============================================================================

st.set_page_config(
    page_title="GulfBite",
    page_icon="🍲",
    layout="centered",
    initial_sidebar_state="collapsed",
)
inject_theme()

if "stage" not in st.session_state:
    st.session_state.stage = "upload"[cite: 1]
    st.session_state.triggered = False[cite: 1]
    st.session_state.image = None[cite: 1]
    st.session_state.cnn_class = None[cite: 1]
    st.session_state.cnn_confidence = None[cite: 1]
    st.session_state.candidates = None[cite: 1]
    st.session_state.yolo_suggestion = None[cite: 1]
    st.session_state.yolo_gate_status = None[cite: 1]
    st.session_state.tier_used = None[cite: 1]
    st.session_state.final_dish = None[cite: 1]
    st.session_state.portion_size = "M"[cite: 1]


def reset():
    for key in list(st.session_state.keys()):[cite: 1]
        del st.session_state[key][cite: 1]


# ============================================================================
# 6. APP MAIN HEADER & EXPLORER TABS
# ============================================================================

render_header()

guide_tab, dishes_tab = st.tabs(["⚡ Quick Guide", "🍽️ Supported Dishes"])

with guide_tab:
    st.markdown(
        """<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin: 10px 0 6px 0;">
<div style="background: #FFFFFF; border: 1px solid #EBE2CF; border-radius: 20px; padding: 14px 8px; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.02);">
    <div style="font-size: 1.5rem; margin-bottom: 4px;">📸</div>
    <div style="font-family: 'Outfit', sans-serif; color: #1E1B16; font-weight: 800; font-size: 0.8rem;">1. Snap Meal</div>
    <div style="color: #8F887C; font-size: 0.72rem; line-height: 1.3; margin-top: 2px;">Top-down plate photo</div>
</div>
<div style="background: #FFFFFF; border: 1px solid #EBE2CF; border-radius: 20px; padding: 14px 8px; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.02);">
    <div style="font-size: 1.5rem; margin-bottom: 4px;">🔍</div>
    <div style="font-family: 'Outfit', sans-serif; color: #1E1B16; font-weight: 800; font-size: 0.8rem;">2. AI Verify</div>
    <div style="color: #8F887C; font-size: 0.72rem; line-height: 1.3; margin-top: 2px;">Smart recipe check</div>
</div>
<div style="background: #FFFFFF; border: 1px solid #EBE2CF; border-radius: 20px; padding: 14px 8px; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.02);">
    <div style="font-size: 1.5rem; margin-bottom: 4px;">⚖️</div>
    <div style="font-family: 'Outfit', sans-serif; color: #1E1B16; font-weight: 800; font-size: 0.8rem;">3. Get Macros</div>
    <div style="color: #8F887C; font-size: 0.72rem; line-height: 1.3; margin-top: 2px;">Authentic range data</div>
</div>
</div>""",
        unsafe_allow_html=True,
    )

with dishes_tab:
    render_interactive_dish_explorer()

st.markdown("<div style='margin-bottom: 1.4rem;'></div>", unsafe_allow_html=True)


# ============================================================================
# 7. LOAD MODELS INTO CACHE
# ============================================================================

try:
    cnn_model, idx_to_class, yolo_model, ingredient_cache = load_models()[cite: 1]
except FileNotFoundError as e:[cite: 1]
    st.error(str(e))[cite: 1]
    st.stop()[cite: 1]


# ============================================================================
# 8. STAGE 1: UPLOAD
# ============================================================================

if st.session_state.stage == "upload":[cite: 1]
    render_stepper("upload", st.session_state.triggered)[cite: 1]

    with st.container(border=True):[cite: 1]
        st.markdown(
            """<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
                <div style="font-family: 'Outfit', sans-serif; font-size: 1.1rem; font-weight: 800; color: #1E1B16;">Scan Your Plate</div>
                <span style="background: #FDF6E9; color: #C28416; font-size: 0.76rem; font-weight: 800; padding: 4px 10px; border-radius: 999px; border: 1px solid #F5E3BE;">Top-down</span>
            </div>""",
            unsafe_allow_html=True,
        )

        image_to_process = None[cite: 1]

        uploaded = st.file_uploader(
            "Upload meal photo",
            type=["jpg", "jpeg", "png", "heic", "heif"],
            label_visibility="collapsed",
        )[cite: 1]

        if uploaded is not None:[cite: 1]
            image_to_process = ImageOps.exif_transpose(Image.open(uploaded))[cite: 1]

        if image_to_process is not None:[cite: 1]
            st.session_state.image = image_to_process[cite: 1]
            st.image(
                image_to_process,
                caption="Scanned Plate",
                use_column_width=True,
            )[cite: 1]

            with st.spinner("Analyzing ingredients & calculating nutritional profile..."):
                cnn_class, cnn_confidence, margin, entropy = run_cnn(
                    image_to_process, cnn_model, idx_to_class
                )[cite: 1]

                # Non-food guardrail[cite: 1]
                is_non_food = (
                    cnn_confidence < MIN_CONFIDENCE
                    or margin < MIN_MARGIN
                    or entropy > MAX_ENTROPY
                )[cite: 1]

                if is_non_food:[cite: 1]
                    st.markdown(
                        f"""<div style="background: #FFF5F5; border: 1px solid #FED7D7; border-radius: 22px; padding: 18px; margin-top: 14px; text-align: center;">
<div style="font-size: 2rem; margin-bottom: 4px;">🍽️❓</div>
<div style="color: #E53E3E; font-family: 'Outfit', sans-serif; font-weight: 800; font-size: 1.05rem;">No Supported Gulf Dish Found</div>
<div style="color: #718096; font-size: 0.82rem; line-height: 1.45; margin-top: 4px;">
Please upload a clear, top-down photo of a traditional Gulf dish.
</div>
</div>""",
                        unsafe_allow_html=True,
                    )[cite: 1]
                    st.write("")[cite: 1]
                    st.button(
                        "🔄 Try Another Photo",
                        on_click=reset,
                        use_container_width=True,
                    )[cite: 1]
                    st.stop()[cite: 1]

                triggered = (
                    cnn_confidence < CONFIDENCE_THRESHOLD
                    or cnn_class in TRIGGER_SET
                    or cnn_class in WRAP_TRIGGER_SET
                )[cite: 1]

                st.session_state.cnn_class = cnn_class[cite: 1]
                st.session_state.cnn_confidence = cnn_confidence[cite: 1]
                st.session_state.triggered = triggered[cite: 1]

                if not triggered:[cite: 1]
                    st.session_state.final_dish = cnn_class[cite: 1]
                    st.session_state.tier_used = "CNN direct match"
                    st.session_state.stage = "select_portion"[cite: 1]
                    st.rerun()[cite: 1]
                else:
                    candidates = get_candidate_group(cnn_class)[cite: 1]
                    run_yolo_here = (cnn_class in YOLO_FEATURE_MAP) or (
                        cnn_class == "03_biryani"
                    )[cite: 1]
                    yolo_suggestion, gate_status = None, None[cite: 1]
                    if run_yolo_here:[cite: 1]
                        detections = run_yolov8(image_to_process, yolo_model)[cite: 1]
                        _, gated, gate_status = map_detections_to_suggestion(
                            detections, candidates
                        )[cite: 1]
                        yolo_suggestion = gated[0] if gated else None[cite: 1]

                    st.session_state.candidates = sorted(candidates)[cite: 1]
                    st.session_state.yolo_suggestion = yolo_suggestion[cite: 1]
                    st.session_state.yolo_gate_status = gate_status[cite: 1]
                    st.session_state.tier_used = (
                        "CNN + YOLO + user confirm"
                        if yolo_suggestion
                        else "CNN + user confirm"
                    )[cite: 1]
                    st.session_state.stage = "confirm_dish"[cite: 1]
                    st.rerun()[cite: 1]


# ============================================================================
# 9. STAGE 2: CONFIRM DISH
# ============================================================================

elif st.session_state.stage == "confirm_dish":[cite: 1]
    render_stepper("confirm_dish", True)[cite: 1]

    with st.container(border=True):[cite: 1]
        st.image(
            st.session_state.image,
            caption="Scanned Plate",
            use_column_width=True,
        )[cite: 1]

        cnn_class = st.session_state.cnn_class[cite: 1]
        cnn_conf = st.session_state.cnn_confidence[cite: 1]
        candidates = st.session_state.candidates[cite: 1]
        yolo_suggestion = st.session_state.yolo_suggestion[cite: 1]

        st.markdown(
            f"""<div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 0.6rem;">
                <div style="font-family: 'Outfit', sans-serif; font-size: 1.45rem; font-weight: 900; color: #1E1B16;">
                    Initial Match: <span style="color: #E5A93B;">{display_name(cnn_class)}</span>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )[cite: 1]

        render_confidence_bar(cnn_conf)[cite: 1]

        reason = get_group_reason(cnn_class)[cite: 1]
        if reason:[cite: 1]
            st.markdown(
                f"""<div class="verify-callout">
                    <span style="font-size: 1.1rem; line-height: 1;">🔍</span>
                    <span style="color: #736C61; font-size: 0.84rem; line-height: 1.4;">{reason}</span>
                </div>""",
                unsafe_allow_html=True,
            )[cite: 1]

        if yolo_suggestion:[cite: 1]
            st.markdown(
                f"""<div class="ingredient-badge">
                    <span style="font-size: 1.1rem;">✨</span>
                    <span>Visual ingredient inspection detected: <strong style="color: #C28416;">{display_name(yolo_suggestion)}</strong></span>
                </div>""",
                unsafe_allow_html=True,
            )

        default_choice = yolo_suggestion if yolo_suggestion else cnn_class[cite: 1]
        default_idx = (
            candidates.index(default_choice)
            if default_choice in candidates
            else 0
        )[cite: 1]

        st.markdown(
            '<p style="font-family: \'Outfit\', sans-serif; font-size: 0.88rem; font-weight: 800; color: #1E1B16; margin: 12px 0 6px 0;">Select your dish:</p>',
            unsafe_allow_html=True,
        )[cite: 1]

        choice = st.radio(
            "Select matching dish:",
            options=candidates,
            format_func=lambda x: f"🍲 {display_name(x)}",
            index=default_idx,
            label_visibility="collapsed",
        )[cite: 1]

        st.write("")[cite: 1]
        if st.button("Confirm Dish & Continue →", type="primary"):[cite: 1]
            st.session_state.final_dish = choice[cite: 1]
            st.session_state.stage = "select_portion"[cite: 1]
            st.rerun()[cite: 1]


# ============================================================================
# 10. STAGE 3: SELECT PORTION (Segmented Pill Layout)
# ============================================================================

elif st.session_state.stage == "select_portion":[cite: 1]
    render_stepper("select_portion", st.session_state.triggered)[cite: 1]

    with st.container(border=True):[cite: 1]
        st.image(
            st.session_state.image,
            caption="Scanned Plate",
            use_column_width=True,
        )[cite: 1]

        st.markdown(
            f"""<div style="font-family: 'Outfit', sans-serif; font-size: 1.65rem; font-weight: 900; color: #1E1B16; margin: 0.6rem 0 0.2rem 0;">
                {display_name(st.session_state.final_dish)}
            </div>
            <p style="color: #8F887C; font-size: 0.86rem; font-weight: 500; margin-bottom: 1.2rem;">
                Select your portion size to calculate authentic nutrition values:
            </p>""",
            unsafe_allow_html=True,
        )[cite: 1]

        portion_map = {
            "S": "🌱   Small   (~250g)",
            "M": "🍽️   Medium   (~400g)",
            "L": "👑   Large   (~550g)",
        }

        st.markdown('<div class="portion-card-group">', unsafe_allow_html=True)
        selected_p = st.radio(
            "Choose portion:",
            options=["S", "M", "L"],
            format_func=lambda x: portion_map[x],
            index=1,
            horizontal=False,
            label_visibility="collapsed",
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.write("")[cite: 1]
        if st.button("Calculate Nutrition →", type="primary", use_container_width=True):
            st.session_state.portion_size = selected_p
            st.session_state.stage = "result"[cite: 1]
            st.rerun()[cite: 1]


# ============================================================================
# 11. STAGE 4: NUTRITIONAL BREAKDOWN RESULT
# ============================================================================

elif st.session_state.stage == "result":[cite: 1]
    render_stepper("result", st.session_state.triggered)[cite: 1]

    with st.container(border=True):[cite: 1]
        st.image(
            st.session_state.image,
            caption="Scanned Plate",
            use_column_width=True,
        )[cite: 1]

        dish = st.session_state.get("final_dish")[cite: 1]
        if not dish:[cite: 1]
            dish = st.session_state.get("cnn_class", "01_machboos")[cite: 1]

        nutrition = estimate_nutrition(
            dish, st.session_state.portion_size, ingredient_cache
        )[cite: 1]
        lo, hi = nutrition["calories_range"][cite: 1]

        st.markdown(
            f"""<div style="display: flex; justify-content: space-between; align-items: center; margin-top: 0.6rem;">
                <div>
                    <div style="font-family: 'Outfit', sans-serif; font-size: 1.65rem; font-weight: 900; color: #1E1B16;">{display_name(dish)}</div>
                    <div style="color: #8F887C; font-size: 0.84rem; font-weight: 600;">Portion size: <strong style="color:#C28416;">{PORTION_LABELS[st.session_state.portion_size]}</strong></div>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

        blurb = DISH_BLURBS.get(dish)[cite: 1]
        if blurb:[cite: 1]
            st.markdown(
                f'<p style="color: #736C61; font-size: 0.84rem; line-height: 1.45; margin: 0.6rem 0 0 0;">{blurb}</p>',
                unsafe_allow_html=True,
            )

        render_calorie_hero(lo, hi)[cite: 1]
        render_macro_cards(
            nutrition["protein_g"], nutrition["carbs_g"], nutrition["fat_g"]
        )[cite: 1]

        if nutrition["missing_ingredients"]:[cite: 1]
            st.warning(
                f"Missing standard data for: {', '.join(nutrition['missing_ingredients'])}."
            )[cite: 1]

        st.markdown("<div style='margin-top: 1.2rem;'></div>", unsafe_allow_html=True)

        tab_correct, tab_tech = st.tabs(["✏️ Edit Dish", "⚙️ Pipeline Info"])

        with tab_correct:
            all_dishes = sorted(DISH_RECIPES.keys(), key=display_name)[cite: 1]
            current_idx = all_dishes.index(dish) if dish in all_dishes else 0[cite: 1]

            corrected = st.selectbox(
                "Select correct dish:",
                options=all_dishes,
                format_func=display_name,
                index=current_idx,
                label_visibility="collapsed",
            )[cite: 1]
            if st.button("Update Dish", type="primary", use_container_width=True):[cite: 1]
                st.session_state.final_dish = corrected[cite: 1]
                st.session_state.tier_used = "User correction"[cite: 1]
                st.rerun()[cite: 1]

        with tab_tech:
            yolo_row = (
                f'<div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #EBE2CF; padding-bottom: 8px;"><span style="color: #8F887C; font-size: 0.84rem;">YOLOv8 Feature</span><span style="color: #1E1B16; font-weight: 700; font-size: 0.88rem;">{display_name(st.session_state.yolo_suggestion)}</span></div>'
                if st.session_state.get("yolo_suggestion")
                else ""
            )

            st.markdown(
                f"""<div style="display: flex; flex-direction: column; gap: 10px; padding: 6px 0;">
<div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #EBE2CF; padding-bottom: 8px;">
<span style="color: #8F887C; font-size: 0.84rem;">CNN Classifier</span>
<span style="color: #1E1B16; font-weight: 700; font-size: 0.88rem;">{display_name(st.session_state.cnn_class)} <span style="color: #E5A93B; font-family: 'JetBrains Mono', monospace;">({st.session_state.cnn_confidence:.0%})</span></span>
</div>
{yolo_row}
<div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #EBE2CF; padding-bottom: 8px;">
<span style="color: #8F887C; font-size: 0.84rem;">Confirmed Dish</span>
<span style="color: #1E1B16; font-weight: 800; font-size: 0.88rem;">{display_name(dish)}</span>
</div>
<div style="display: flex; justify-content: space-between; align-items: center; padding-top: 2px;">
<span style="color: #8F887C; font-size: 0.84rem;">Pipeline Path</span>
<span class="tech-pill">{st.session_state.tier_used}</span>
</div>
</div>""",
                unsafe_allow_html=True,
            )

    st.write("")[cite: 1]
    st.button("📸 Scan Another Plate", on_click=reset)
