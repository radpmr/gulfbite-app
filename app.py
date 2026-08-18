"""
GulfBite — Smart Gulf Cuisine Nutrition Assistant (Mobile Light-Gold Edition)
-----------------------------------------------------------------------------
Identifies authentic Gulf dishes using a multi-tiered pipeline:
1. MobileNetV2 (CNN) classification for initial dish match & confidence scoring.
2. Out-of-distribution / Non-food rejection via margin and entropy checks.
3. YOLOv8 feature detection with visual bounding overlays & calorie pointers.
4. Portion-based authentic macro and calorie estimation with SVG Macro Rings.
"""

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import streamlit as st

# Force CPU inference for stability and suppress TensorFlow verbose logging
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Register HEIC/HEIF image support for mobile uploads
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

TRIGGER_SET = {
    "07_ouzi",
    "01_machboos",
    "09_jisheed",
    "02_kabsa",
    "03_biryani",
    "06_saloona",
}
WRAP_TRIGGER_SET = {"10_shawarma", "11_falafel_wrap"}

CONFIDENCE_THRESHOLD = 0.70
MIN_CONFIDENCE = 0.50
MIN_MARGIN = 0.15
MAX_ENTROPY = 2.50

YOLO_FEATURE_MAP = {
    "01_machboos": "loomi",
    "07_ouzi": "whole_shank",
    "08_samak_mashwi": "whole_fish",
    "02_kabsa": "whole_chicken_piece",
    "10_shawarma": "shawarma_meat",
    "11_falafel_wrap": "falafel_ball",
}
FEATURE_TO_DISH = {v: k for k, v in YOLO_FEATURE_MAP.items()}

FEATURE_CALORIE_ESTIMATES = {
    "loomi": "15 kcal",
    "whole_chicken_piece": "240 kcal",
    "whole_shank": "320 kcal",
    "whole_fish": "210 kcal",
    "shawarma_meat": "190 kcal",
    "falafel_ball": "60 kcal",
}

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
    "rice_cluster": "Rice dishes like Machboos, Kabsa, and Biryani share aromatic spice bases, so we double-check.",
    "wrap_cluster": "Wrapped dishes hide their core filling, so we double-check with you.",
}


def get_group_reason(cnn_class: str) -> Optional[str]:
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
    "01_machboos": [("rice", 150), ("chicken", 130), ("olive_oil", 15), ("onion", 20), ("tomato", 15)],
    "02_kabsa": [("rice", 150), ("chicken", 130), ("olive_oil", 15), ("tomato", 20), ("onion", 15)],
    "03_biryani": [("rice", 160), ("chicken", 140), ("olive_oil", 15), ("yogurt", 20), ("onion", 20)],
    "04_harees": [("bulgur", 100), ("lamb", 100), ("ghee", 15)],
    "05_thareed": [("pita_bread", 80), ("lamb", 120), ("mixed_vegetables", 60)],
    "06_saloona": [("lamb", 120), ("mixed_vegetables", 100), ("tomato", 40), ("olive_oil", 15)],
    "07_ouzi": [("rice", 150), ("lamb", 180), ("mixed_nuts", 15), ("olive_oil", 15)],
    "08_samak_mashwi": [("fish", 200), ("olive_oil", 10)],
    "09_jisheed": [("rice", 150), ("fish", 100), ("olive_oil", 10)],
    "10_shawarma": [("pita_bread", 80), ("chicken", 100), ("garlic_sauce", 20), ("pickles", 10)],
    "11_falafel_wrap": [("pita_bread", 80), ("falafel", 90), ("tahini", 15), ("mixed_vegetables", 30)],
    "12_falafel": [("falafel", 120), ("olive_oil", 10)],
    "13_samboosa": [("pastry_dough", 60), ("ground_meat", 60), ("olive_oil", 10)],
    "14_mutabbaq": [("pastry_dough", 100), ("ground_meat", 80), ("olive_oil", 15)],
    "15_hummus": [("chickpeas", 80), ("tahini", 15), ("olive_oil", 10)],
    "16_fattoush": [("mixed_vegetables", 150), ("pita_bread", 20), ("olive_oil", 10)],
    "17_tabbouleh": [("parsley", 80), ("bulgur", 20), ("tomato", 30), ("olive_oil", 15)],
    "18_foul_medames": [("fava_beans", 150), ("olive_oil", 15)],
    "19_shakshuka": [("eggs", 100), ("tomato_sauce", 150), ("olive_oil", 10)],
    "20_balaleet": [("vermicelli", 80), ("sugar", 15), ("eggs", 50)],
    "21_khameer": [("bread_wheat", 80)],
    "22_chebab": [("pancake_batter", 100)],
    "23_luqaimat": [("fried_dough", 100), ("date_syrup", 30)],
    "24_knafeh": [("kunafa_dough", 80), ("soft_cheese", 60), ("sugar_syrup", 40), ("ghee", 15)],
    "25_karak_chai": [("milk", 100), ("black_tea", 100), ("sugar", 10)],
}

DISH_METADATA = {
    "01_machboos": {"spice": "Aromatic 🌶️🌶️", "prep": "Slow-Simmered ⏳", "density": "High Protein 🥩", "time": "60 min"},
    "02_kabsa": {"spice": "Aromatic 🌶️🌶️", "prep": "Infused Broth 🍲", "density": "Balanced Macros ⚖️", "time": "50 min"},
    "03_biryani": {"spice": "Richly Spiced 🌶️🌶️🌶️", "prep": "Dum Layered ♨️", "density": "Carb & Protein 🌾", "time": "55 min"},
    "04_harees": {"spice": "Mild 🌶️", "prep": "Slow-Beaten ⏳", "density": "Complex Carbs 🌾", "time": "90 min"},
    "05_thareed": {"spice": "Aromatic 🌶️🌶️", "prep": "Broth Layered 🍲", "density": "High Protein 🥩", "time": "45 min"},
    "06_saloona": {"spice": "Medium 🌶️🌶️", "prep": "Clay Pot Simmer 🥘", "density": "Micronutrient Rich 🥗", "time": "40 min"},
    "07_ouzi": {"spice": "Mild & Nutty 🌰", "prep": "Pit Roasted 🔥", "density": "High Protein 🥩", "time": "75 min"},
    "08_samak_mashwi": {"spice": "Citrus Herb 🍋", "prep": "Charcoal Grilled 🔥", "density": "Lean Protein 🐟", "time": "30 min"},
    "09_jisheed": {"spice": "Loomi & Turmeric 🍋", "prep": "Pan-Flaked 🍳", "density": "Lean Protein 🐟", "time": "35 min"},
    "10_shawarma": {"spice": "Garlic Spiced 🧄", "prep": "Vertical Spit 🔥", "density": "High Protein 🥩", "time": "15 min"},
    "11_falafel_wrap": {"spice": "Herbal Cumin 🌿", "prep": "Crisp Fried 🫓", "density": "Plant Fiber 🌱", "time": "15 min"},
    "12_falafel": {"spice": "Herbaceous 🌿", "prep": "Golden Fried 🧆", "density": "Plant Protein 🌱", "time": "20 min"},
    "13_samboosa": {"spice": "Spiced Minced 🌶️", "prep": "Pastry Crisp 🥟", "density": "High Energy ⚡", "time": "20 min"},
    "14_mutabbaq": {"spice": "Scallion Pepper 🧅", "prep": "Griddle Pan 🍳", "density": "Protein Pastry 🥩", "time": "25 min"},
    "15_hummus": {"spice": "Tahini Citrus 🍋", "prep": "Cold Blended 🥣", "density": "Heart-Healthy Fats 🥑", "time": "10 min"},
    "16_fattoush": {"spice": "Sumac Zesty 🍋", "prep": "Fresh Crisp Toss 🥗", "density": "High Fiber 🍃", "time": "15 min"},
    "17_tabbouleh": {"spice": "Mint Lemon 🌿", "prep": "Fine Chipped 🥗", "density": "Antioxidant Rich 🍃", "time": "20 min"},
    "18_foul_medames": {"spice": "Cumin Olive Oil 🫒", "prep": "Slow Stewed 🫘", "density": "High Fiber & Protein 🌱", "time": "30 min"},
    "19_shakshuka": {"spice": "Tomato Cumin 🍅", "prep": "Skillet Poached 🍳", "density": "Lean Protein 🥚", "time": "20 min"},
    "20_balaleet": {"spice": "Cardamom Saffron 🍯", "prep": "Sweet Savoury Omelette 🍳", "density": "Energy Carbs 🌾", "time": "25 min"},
    "21_khameer": {"spice": "Date Scented 🌴", "prep": "Tannur Baked 🫓", "density": "Artisan Carbs 🌾", "time": "30 min"},
    "22_chebab": {"spice": "Cardamom Honey 🍯", "prep": "Golden Griddle 🥞", "density": "Carb Fuel 🌾", "time": "20 min"},
    "23_luqaimat": {"spice": "Date Molasses 🍯", "prep": "Crisp Puffs 🥟", "density": "Sweet Treat 🍯", "time": "25 min"},
    "24_knafeh": {"spice": "Orange Blossom 🌸", "prep": "Golden Filo Bake 🧀", "density": "Energy Rich 🧀", "time": "35 min"},
    "25_karak_chai": {"spice": "Crushed Cardamom ☕", "prep": "Slow Simmered 🫖", "density": "Comfort Beverage 🫖", "time": "15 min"},
}

DISH_BLURBS = {
    "01_machboos": "A fragrant spiced rice plate with meat or chicken, infused with black dried lime (loomi).",
    "02_kabsa": "Saudi Arabia's signature spiced rice dish finished with saffron, tomatoes, and tender meat.",
    "03_biryani": "Richly layered basmati rice spiced with cloves, cardamoms, and marinated chicken.",
    "04_harees": "Slow-cooked wheat and shredded meat porridge seasoned with aromatic ghee.",
    "05_thareed": "Crisp thin flatbread layered with hearty lamb and slow-simmered vegetable broth.",
    "06_saloona": "A traditional comforting Gulf stew simmered with seasonal vegetables and spices.",
    "07_ouzi": "Spiced rice loaded with slow-roasted tender lamb and toasted golden nuts.",
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
}

PORTION_MULTIPLIERS = {"S": 0.7, "M": 1.0, "L": 1.4}
PORTION_LABELS = {"S": "Small", "M": "Medium", "L": "Large"}
CALORIE_RANGE_PCT = 0.15


def display_name(cls: str) -> str:
    return cls.split("_", 1)[1].replace("_", " ").title()


DISH_CATEGORIES_DATA = {
    "🍚 Rice Mains": ["01_machboos", "02_kabsa", "03_biryani", "07_ouzi", "09_jisheed"],
    "🥘 Stews & Mains": ["04_harees", "05_thareed", "06_saloona", "08_samak_mashwi"],
    "🌯 Wraps & Bites": ["10_shawarma", "11_falafel_wrap", "12_falafel", "13_samboosa", "14_mutabbaq"],
    "🫓 Breads & Morning": ["18_foul_medames", "19_shakshuka", "20_balaleet", "21_khameer", "22_chebab"],
    "🥗 Fresh Salads": ["15_hummus", "16_fattoush", "17_tabbouleh"],
    "🍯 Sweets & Karak": ["23_luqaimat", "24_knafeh", "25_karak_chai"],
}


def get_candidate_group(cnn_class: str) -> set:
    for group in CONFUSION_GROUPS.values():
        if cnn_class in group:
            return group
    return {cnn_class}


# ============================================================================
# 2. MODEL LOADING & INFERENCE
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


def run_cnn(pil_image, model, idx_to_class, img_size=(224, 224)):
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    img = pil_image.convert("RGB").resize(img_size)
    arr = np.array(img).astype("float32")
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr, verbose=0)[0]

    sorted_indices = np.argsort(preds)[::-1]
    top_idx = int(sorted_indices[0])
    second_idx = int(sorted_indices[1])

    confidence = float(preds[top_idx])
    second_confidence = float(preds[second_idx])
    margin = confidence - second_confidence

    eps = 1e-12
    entropy = -np.sum(preds * np.log(preds + eps))

    predicted_class = idx_to_class[top_idx]
    return predicted_class, confidence, margin, entropy


def run_yolov8_with_boxes(pil_image, yolo_model, conf_threshold=0.25):
    results = yolo_model.predict(
        np.array(pil_image.convert("RGB")), conf=conf_threshold, verbose=False
    )
    detections = []
    r = results[0]
    for box in r.boxes:
        cls_id = int(box.cls[0])
        cls_name = yolo_model.names[cls_id]
        box_conf = float(box.conf[0])
        coords = [float(x) for x in box.xyxy[0].tolist()]
        detections.append((cls_name, box_conf, coords))
    return detections


def create_ai_decoded_overlay(pil_image, detections):
    img_draw = pil_image.convert("RGB").copy()
    draw = ImageDraw.Draw(img_draw, "RGBA")
    w, h = img_draw.size

    for feat, conf, (x1, y1, x2, y2) in detections:
        draw.rectangle([x1, y1, x2, y2], outline="#E5A93B", width=max(3, int(w * 0.006)))
        corner_len = max(12, int(w * 0.03))
        draw.line([x1, y1, x1 + corner_len, y1], fill="#FFFFFF", width=4)
        draw.line([x1, y1, x1, y1 + corner_len], fill="#FFFFFF", width=4)
        draw.line([x2, y1, x2 - corner_len, y1], fill="#FFFFFF", width=4)
        draw.line([x2, y1, x2, y1 + corner_len], fill="#FFFFFF", width=4)
        draw.line([x1, y2, x1 + corner_len, y2], fill="#FFFFFF", width=4)
        draw.line([x1, y2, x1, y2 - corner_len], fill="#FFFFFF", width=4)
        draw.line([x2, y2, x2 - corner_len, y2], fill="#FFFFFF", width=4)
        draw.line([x2, y2, x2, y2 - corner_len], fill="#FFFFFF", width=4)

        badge_text = f"{feat.replace('_', ' ').title()} • ~{FEATURE_CALORIE_ESTIMATES.get(feat, '120 kcal')}"
        bx = max(10, min(w - 200, int(x1)))
        by = max(10, int(y1 - 32))
        
        draw.rounded_rectangle([bx, by, bx + 190, by + 26], radius=13, fill=(255, 255, 255, 235), outline="#E5A93B", width=2)
        draw.ellipse([bx + 8, by + 9, bx + 16, by + 17], fill="#E5A93B")
        draw.text((bx + 22, by + 5), badge_text[:24], fill="#1E1B16")

    return img_draw


def map_detections_to_suggestion(detections, candidates):
    if not detections:
        return None, None, "no_detection"
    valid = [
        (FEATURE_TO_DISH[feat], conf, feat)
        for feat, conf, _ in detections
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
# 3. MODERN LIGHT MOBILE THEME & CSS
# ============================================================================

def inject_theme():
    st.markdown(
        """<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Outfit:wght@500;600;700;800;900&family=JetBrains+Mono:wght@500;700&display=swap');

:root {
    --app-bg: #F7F4ED;
    --surface: #FFFFFF;
    --surface-soft: #FCFAF6;
    --gold: #E5A93B;
    --gold-2: #F3C36A;
    --gold-dark: #B9780E;
    --gold-soft: #FFF7E7;
    --ink: #1E1B16;
    --muted: #7D766B;
    --line: #EAE2D4;
    --danger: #C2413A;
    --success: #18795B;
}

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: var(--ink);
}

.stApp {
    background:
        radial-gradient(circle at 50% -8%, rgba(243,195,106,.23) 0%, rgba(243,195,106,0) 42%),
        linear-gradient(180deg, #FBF8F0 0%, var(--app-bg) 100%);
    color: var(--ink);
}

#MainMenu, footer, header[data-testid="stHeader"] {
    visibility: hidden;
    height: 0;
}

.block-container {
    max-width: 460px !important;
    padding: 1rem 1rem 2rem 1rem !important;
}

/* Reduce Streamlit's default vertical gaps so the app feels like a mobile product. */
[data-testid="stVerticalBlock"] { gap: .72rem !important; }
[data-testid="stElementContainer"] { margin-bottom: 0 !important; }

/* General cards */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255,255,255,.96) !important;
    border: 1px solid rgba(229,169,59,.20) !important;
    border-radius: 24px !important;
    padding: 1rem !important;
    box-shadow: 0 14px 34px -20px rgba(68,45,9,.28), 0 1px 3px rgba(0,0,0,.025) !important;
}

/* Buttons */
div.stButton > button {
    min-height: 48px;
    width: 100%;
    border: 1px solid transparent;
    border-radius: 16px;
    background: #1D1A16;
    color: #FFFFFF;
    font-family: 'Outfit', sans-serif;
    font-size: .94rem;
    font-weight: 800;
    letter-spacing: .005em;
    box-shadow: none;
    transition: transform .18s ease, box-shadow .18s ease, background .18s ease;
}

div.stButton > button:hover {
    background: #2A251F;
    color: #FFFFFF;
    border-color: transparent;
    transform: translateY(-1px);
}

div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--gold-2) 0%, var(--gold) 62%, #D79620 100%) !important;
    color: #171007 !important;
    box-shadow: 0 9px 20px rgba(229,169,59,.26) !important;
}

div.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #F8CF80 0%, #EAB34C 62%, #D99A29 100%) !important;
}

/* Selects */
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    min-height: 48px !important;
    border-radius: 15px !important;
    border-color: var(--line) !important;
    background: var(--surface-soft) !important;
    box-shadow: none !important;
}

/* Native segmented controls — used for app navigation, image source and portion. */
[data-testid="stSegmentedControl"] {
    width: 100% !important;
}
[data-testid="stSegmentedControl"] > div {
    width: 100% !important;
}
[data-testid="stSegmentedControl"] [role="radiogroup"] {
    width: 100% !important;
    gap: 5px !important;
}
[data-testid="stSegmentedControl"] button {
    flex: 1 1 0 !important;
    min-height: 42px !important;
    border-radius: 13px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 800 !important;
}

/* Fallback radio used only when segmented_control is unavailable. */
.compact-radio div[data-testid="stRadio"] > div[role="radiogroup"] {
    display: flex !important;
    flex-direction: row !important;
    gap: 6px !important;
    width: 100% !important;
}
.compact-radio div[data-testid="stRadio"] label {
    flex: 1 1 0 !important;
}

/* Upload zone */
[data-testid="stFileUploaderDropzone"] {
    min-height: 150px !important;
    padding: 1.3rem 1rem !important;
    border: 1.5px dashed #DBC99E !important;
    border-radius: 18px !important;
    background: #FBF8F1 !important;
    transition: .2s ease !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--gold) !important;
    background: #FFF9EC !important;
    box-shadow: 0 8px 22px rgba(229,169,59,.10) !important;
}
[data-testid="stFileUploaderDropzone"] svg {
    color: var(--gold) !important;
    fill: var(--gold) !important;
}
[data-testid="stFileUploader"] button {
    border-radius: 12px !important;
    border: 1px solid #E5D8BC !important;
    background: #FFFFFF !important;
    color: var(--ink) !important;
    font-weight: 800 !important;
}

/* Camera input */
[data-testid="stCameraInput"] {
    border-radius: 18px;
    overflow: hidden;
}

/* Secondary result tabs only (main navigation no longer uses tabs). */
div[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 6px !important;
    border-bottom: 1px solid var(--line) !important;
}
div[data-testid="stTabs"] [data-baseweb="tab"] {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 800 !important;
    color: #8A8275 !important;
    padding-left: 8px !important;
    padding-right: 8px !important;
}
div[data-testid="stTabs"] [aria-selected="true"] {
    color: var(--ink) !important;
}

/* Verification / model feedback */
.verify-callout {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    background: #FFF9EC;
    border: 1px solid #F2DEAF;
    border-left: 4px solid var(--gold);
    border-radius: 14px;
    padding: 11px 12px;
    margin: 10px 0 13px 0;
}

.ingredient-badge {
    display: flex;
    align-items: center;
    gap: 8px;
    background: linear-gradient(135deg, #FFF8E9 0%, #FBF1D9 100%);
    border: 1px solid #EEDBB0;
    border-radius: 14px;
    padding: 10px 12px;
    margin: 8px 0 12px 0;
    color: var(--ink);
    font-size: .84rem;
    font-weight: 600;
}

.tech-pill {
    display: inline-block;
    max-width: 190px;
    padding: 4px 9px;
    border-radius: 999px;
    font-size: .70rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    background: #ECFDF5;
    color: #08785D;
    border: 1px solid #B8EAD9;
    text-align: right;
}

/* Onboarding collage */
.gulf-grid-collage {
    position: relative;
    height: 292px;
    display: grid;
    grid-template-columns: 1.08fr .92fr;
    grid-template-rows: 1fr 1fr;
    gap: 5px;
    overflow: hidden;
    border-radius: 24px;
    background: #E9E0D0;
    box-shadow: 0 20px 38px -22px rgba(0,0,0,.34);
    margin-bottom: 1rem;
}
.grid-cell {
    position: relative;
    background-size: cover;
    background-position: center;
}
.grid-cell:first-child { grid-row: 1 / span 2; }
.grid-cell-overlay {
    position: absolute;
    inset: 0;
    background: linear-gradient(180deg, rgba(0,0,0,.02), rgba(0,0,0,.35));
}
.micro-pill {
    position: absolute;
    z-index: 2;
    display: inline-flex;
    align-items: center;
    gap: 5px;
    max-width: calc(100% - 16px);
    padding: 5px 9px;
    border-radius: 999px;
    background: rgba(255,255,255,.93);
    border: 1px solid rgba(255,255,255,.9);
    box-shadow: 0 4px 14px rgba(0,0,0,.14);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    color: var(--ink);
    font-family: 'Outfit', sans-serif;
    font-size: .68rem;
    font-weight: 800;
    white-space: nowrap;
}
.pill-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--gold);
    flex: 0 0 auto;
}

/* Small screens */
@media (max-width: 480px) {
    .block-container { padding: .72rem .72rem 1.5rem .72rem !important; }
    [data-testid="stVerticalBlockBorderWrapper"] { border-radius: 20px !important; padding: .9rem !important; }
    .gulf-grid-collage { height: 270px; border-radius: 20px; }
}
</style>""",
        unsafe_allow_html=True,
    )


# ============================================================================
# 4. APP NAVIGATION, STEPPERS & VISUALIZERS
# ============================================================================

def segmented_choice(label, options, default=None, key=None, label_visibility="collapsed"):
    """Use Streamlit segmented_control when available, with a radio fallback."""
    if hasattr(st, "segmented_control"):
        kwargs = {
            "label": label,
            "options": options,
            "key": key,
            "selection_mode": "single",
            "label_visibility": label_visibility,
        }
        # Avoid Streamlit's "default + existing session_state" warning.
        if not (key and key in st.session_state):
            kwargs["default"] = default
        return st.segmented_control(**kwargs)

    if key and key in st.session_state and st.session_state[key] in options:
        default_idx = options.index(st.session_state[key])
    else:
        default_idx = options.index(default) if default in options else 0
    return st.radio(
        label,
        options=options,
        index=default_idx,
        horizontal=True,
        key=key,
        label_visibility=label_visibility,
    )


def render_header(compact: bool = True):
    title_size = "1.55rem" if compact else "1.85rem"
    subtitle = "Gulf cuisine recognition • calories • macros"
    st.markdown(
        f"""<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:.8rem;">
<div style="display:flex;align-items:center;gap:10px;min-width:0;">
    <div style="width:42px;height:42px;border-radius:14px;background:linear-gradient(135deg,#F4C66F,#E5A93B);display:flex;align-items:center;justify-content:center;box-shadow:0 7px 16px rgba(229,169,59,.22);font-family:'Outfit',sans-serif;font-weight:900;color:#1A1305;">GB</div>
    <div style="min-width:0;">
        <div style="font-family:'Outfit',sans-serif;font-size:{title_size};font-weight:900;line-height:1;letter-spacing:-.025em;color:#1E1B16;white-space:nowrap;"><span style="color:#D99926;">GulfBite</span> AI</div>
        <div style="font-size:.72rem;color:#91897D;font-weight:600;margin-top:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:260px;">{subtitle}</div>
    </div>
</div>
<div style="display:flex;align-items:center;gap:8px;">
    <div style="width:38px;height:38px;border-radius:13px;background:#FFFFFF;border:1px solid #EBE4D8;display:flex;align-items:center;justify-content:center;position:relative;box-shadow:0 4px 12px rgba(0,0,0,.035);">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9" stroke="#D99926" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M13.73 21a2 2 0 0 1-3.46 0" stroke="#D99926" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
        <span style="position:absolute;top:8px;right:8px;width:6px;height:6px;background:#E5553F;border-radius:50%;border:1.5px solid white;"></span>
    </div>
</div>
</div>""",
        unsafe_allow_html=True,
    )


def render_main_navigation():
    nav_options = ["🏠 Home", "📖 Menu", "📷 Scan"]

    # Buttons elsewhere in the UI set a pending destination. Apply it before
    # the navigation widget is instantiated on this rerun.
    pending = st.session_state.pop("pending_main_section", None)
    if pending in nav_options:
        st.session_state.main_section = pending
        st.session_state.main_nav_segment = pending

    current = st.session_state.get("main_section", "🏠 Home")
    selected = segmented_choice(
        "Main navigation",
        nav_options,
        default=current if current in nav_options else nav_options[0],
        key="main_nav_segment",
    )
    if selected:
        st.session_state.main_section = selected
    return st.session_state.get("main_section", nav_options[0])


def render_segmented_stepper(current_stage: str, triggered: bool):
    raw_steps = [("upload", "Scan")]
    if triggered:
        raw_steps.append(("confirm_dish", "Verify"))
    raw_steps.append(("select_portion", "Portion"))
    raw_steps.append(("result", "Macros"))

    keys = [key for key, _ in raw_steps]
    active_idx = keys.index(current_stage) if current_stage in keys else 0

    segments = []
    labels = []
    for i, (_, label) in enumerate(raw_steps):
        active = i <= active_idx
        current = i == active_idx
        bg = "linear-gradient(90deg,#F3C36A,#E5A93B)" if active else "#EAE4D8"
        shadow = "box-shadow:0 2px 8px rgba(229,169,59,.22);" if active else ""
        segments.append(f'<div style="flex:1;height:5px;border-radius:999px;background:{bg};{shadow}"></div>')
        color = "#1E1B16" if current else "#9A9286"
        weight = "900" if current else "700"
        labels.append(f'<span style="font-family:\'Outfit\',sans-serif;font-size:.70rem;font-weight:{weight};color:{color};">{i+1}. {label}</span>')

    st.markdown(
        '<div style="margin:.15rem 0 .9rem 0;">'
        '<div style="display:flex;gap:6px;margin-bottom:7px;">' + ''.join(segments) + '</div>'
        '<div style="display:flex;justify-content:space-between;padding:0 1px;">' + ''.join(labels) + '</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def render_quick_guide():
    st.markdown(
        """<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:7px;margin:4px 0 12px 0;">
<div style="background:#FFFFFF;border:1px solid #EAE2D4;border-radius:17px;padding:12px 6px;text-align:center;">
    <div style="font-size:1.3rem;margin-bottom:3px;">📸</div>
    <div style="font-family:'Outfit',sans-serif;font-weight:900;font-size:.76rem;color:#1E1B16;">Snap</div>
    <div style="font-size:.66rem;color:#90887C;line-height:1.25;margin-top:2px;">Top-down plate</div>
</div>
<div style="background:#FFFFFF;border:1px solid #EAE2D4;border-radius:17px;padding:12px 6px;text-align:center;">
    <div style="font-size:1.3rem;margin-bottom:3px;">✨</div>
    <div style="font-family:'Outfit',sans-serif;font-weight:900;font-size:.76rem;color:#1E1B16;">Recognize</div>
    <div style="font-size:.66rem;color:#90887C;line-height:1.25;margin-top:2px;">AI dish check</div>
</div>
<div style="background:#FFFFFF;border:1px solid #EAE2D4;border-radius:17px;padding:12px 6px;text-align:center;">
    <div style="font-size:1.3rem;margin-bottom:3px;">🥗</div>
    <div style="font-family:'Outfit',sans-serif;font-weight:900;font-size:.76rem;color:#1E1B16;">Track</div>
    <div style="font-size:.66rem;color:#90887C;line-height:1.25;margin-top:2px;">Calories + macros</div>
</div>
</div>""",
        unsafe_allow_html=True,
    )


def render_scan_input():
    """Render camera/upload source selector and return a PIL image or None."""
    sources = ["📷 Camera", "⬆️ Upload"] if hasattr(st, "camera_input") else ["⬆️ Upload"]
    default_source = st.session_state.get("scan_source", sources[0])
    if default_source not in sources:
        default_source = sources[0]

    source = segmented_choice(
        "Photo source",
        sources,
        default=default_source,
        key="scan_source_segment",
    )
    st.session_state.scan_source = source or default_source

    image_file = None
    if st.session_state.scan_source == "📷 Camera" and hasattr(st, "camera_input"):
        image_file = st.camera_input("Take a clear top-down photo", label_visibility="collapsed")
    else:
        image_file = st.file_uploader(
            "Upload meal photo",
            type=["jpg", "jpeg", "png", "heic", "heif"],
            label_visibility="collapsed",
            key="meal_upload",
        )

    if image_file is None:
        return None

    return ImageOps.exif_transpose(Image.open(image_file))


def render_category_squircle_cards():
    categories = list(DISH_CATEGORIES_DATA.keys())
    
    if "selected_category" not in st.session_state:
        st.session_state.selected_category = categories[0]

    selected_cat = st.selectbox(
        "Filter by Category:",
        options=categories,
        index=categories.index(st.session_state.selected_category) if st.session_state.selected_category in categories else 0,
        key="cat_select_box",
    )
    st.session_state.selected_category = selected_cat

    dishes = DISH_CATEGORIES_DATA[selected_cat]

    st.markdown(
        '<div style="margin-top: 14px; margin-bottom: 6px; font-size: 0.78rem; font-weight: 800; color: #8F887C; text-transform: uppercase; letter-spacing: 0.05em;">Choose Recipe</div>',
        unsafe_allow_html=True,
    )

    selected_dish = st.selectbox(
        f"Select a {selected_cat} dish to explore:",
        options=dishes,
        format_func=display_name,
        index=0,
        label_visibility="collapsed",
    )

    if selected_dish:
        meta = DISH_METADATA.get(selected_dish, {"spice": "Aromatic 🌶️", "prep": "Traditional", "density": "Nutritious", "time": "30 min"})
        blurb = DISH_BLURBS.get(selected_dish, "")
        st.markdown(
            f"""<div style="background: #FAF8F3; border: 1.5px solid #EBE2CF; border-radius: 22px; padding: 14px 16px; margin-top: 10px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="font-family: 'Outfit', sans-serif; font-weight: 900; color: #1E1B16; font-size: 1.1rem;">{display_name(selected_dish)}</div>
                    <span style="background: #FDF6E9; color: #C28416; font-size: 0.74rem; font-weight: 800; padding: 4px 10px; border-radius: 999px; border: 1px solid #F5E3BE;">⏱️ {meta['time']}</span>
                </div>
                <p style="color: #736C61; font-size: 0.84rem; line-height: 1.45; margin: 8px 0 10px 0;">{blurb}</p>
                <div style="display: flex; gap: 6px; flex-wrap: wrap;">
                    <span style="font-size: 0.72rem; font-weight: 700; background: #FFFFFF; border: 1px solid #E2D7C3; padding: 3px 8px; border-radius: 8px;">{meta['spice']}</span>
                    <span style="font-size: 0.72rem; font-weight: 700; background: #FFFFFF; border: 1px solid #E2D7C3; padding: 3px 8px; border-radius: 8px;">{meta['prep']}</span>
                    <span style="font-size: 0.72rem; font-weight: 700; background: #FFFFFF; border: 1px solid #E2D7C3; padding: 3px 8px; border-radius: 8px;">{meta['density']}</span>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )


def render_culinary_badges(dish_class: str):
    meta = DISH_METADATA.get(dish_class, {"spice": "Aromatic 🌶️", "prep": "Slow-Simmered ⏳", "density": "Nutrient Rich 🥗", "time": "45 min"})
    st.markdown(
        f"""<div style="display: flex; justify-content: space-between; gap: 6px; margin: 0.8rem 0 1rem 0; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 90px; background: #FAF8F3; border: 1px solid #EBE2CF; border-radius: 14px; padding: 8px 6px; text-align: center;">
                <div style="font-size: 0.68rem; color: #8F887C; font-weight: 700;">FLAVOR</div>
                <div style="font-size: 0.78rem; font-weight: 800; color: #1E1B16; margin-top: 2px;">{meta['spice']}</div>
            </div>
            <div style="flex: 1; min-width: 90px; background: #FAF8F3; border: 1px solid #EBE2CF; border-radius: 14px; padding: 8px 6px; text-align: center;">
                <div style="font-size: 0.68rem; color: #8F887C; font-weight: 700;">COOK STYLE</div>
                <div style="font-size: 0.78rem; font-weight: 800; color: #1E1B16; margin-top: 2px;">{meta['prep']}</div>
            </div>
            <div style="flex: 1; min-width: 90px; background: #FAF8F3; border: 1px solid #EBE2CF; border-radius: 14px; padding: 8px 6px; text-align: center;">
                <div style="font-size: 0.68rem; color: #8F887C; font-weight: 700;">PROFILE</div>
                <div style="font-size: 0.78rem; font-weight: 800; color: #1E1B16; margin-top: 2px;">{meta['density']}</div>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )


def render_macro_donut_and_cards(protein_g: float, carbs_g: float, fat_g: float, lo: int, hi: int):
    cal_prot = protein_g * 4
    cal_carb = carbs_g * 4
    cal_fat = fat_g * 9
    total_cal = max(1.0, cal_prot + cal_carb + cal_fat)

    pct_p = cal_prot / total_cal
    pct_c = cal_carb / total_cal
    pct_f = cal_fat / total_cal

    circumference = 2 * 3.14159 * 42
    len_p = pct_p * circumference
    len_c = pct_c * circumference
    len_f = pct_f * circumference

    off_p = 0
    off_c = -len_p
    off_f = -(len_p + len_c)

    svg_donut = f"""<svg width="116" height="116" viewBox="0 0 100 100" style="transform: rotate(-90deg);">
        <circle cx="50" cy="50" r="42" fill="transparent" stroke="#EFEAE0" stroke-width="12"/>
        <circle cx="50" cy="50" r="42" fill="transparent" stroke="#E5A93B" stroke-width="12" stroke-dasharray="{len_p:.2f} {circumference:.2f}" stroke-dashoffset="{off_p:.2f}" stroke-linecap="round"/>
        <circle cx="50" cy="50" r="42" fill="transparent" stroke="#059669" stroke-width="12" stroke-dasharray="{len_c:.2f} {circumference:.2f}" stroke-dashoffset="{off_c:.2f}" stroke-linecap="round"/>
        <circle cx="50" cy="50" r="42" fill="transparent" stroke="#FF5A1F" stroke-width="12" stroke-dasharray="{len_f:.2f} {circumference:.2f}" stroke-dashoffset="{off_f:.2f}" stroke-linecap="round"/>
    </svg>"""

    avg_cal = round((lo + hi) / 2)

    st.markdown(
        f"""<div style="background: linear-gradient(135deg, #FDF9EE 0%, #FAF3DE 100%); border: 1.5px solid #F3E0B5; border-radius: 26px; padding: 1.2rem 1.3rem; margin: 1rem 0; display: flex; align-items: center; justify-content: space-between;">
            <div>
                <div style="font-size: 0.74rem; font-weight: 800; color: #D19428; text-transform: uppercase; letter-spacing: 0.05em;">Estimated Energy</div>
                <div style="font-family: 'Outfit', sans-serif; font-size: 2.1rem; font-weight: 900; color: #1E1B16; line-height: 1.1; margin: 2px 0 6px 0;">
                    {lo}&ndash;{hi} <span style="font-size: 0.95rem; font-weight: 600; color: #8F887C;">kcal</span>
                </div>
                <div style="display: flex; gap: 8px; font-size: 0.72rem; font-weight: 800;">
                    <span style="color: #E5A93B;">● Prot {pct_p*100:.0f}%</span>
                    <span style="color: #059669;">● Carb {pct_c*100:.0f}%</span>
                    <span style="color: #FF5A1F;">● Fat {pct_f*100:.0f}%</span>
                </div>
            </div>
            <div style="position: relative; width: 116px; height: 116px; display: flex; align-items: center; justify-content: center;">
                {svg_donut}
                <div style="position: absolute; text-align: center; transform: rotate(0deg);">
                    <div style="font-family: 'Outfit', sans-serif; font-weight: 900; font-size: 1.1rem; color: #1E1B16; line-height: 1;">{avg_cal}</div>
                    <div style="font-size: 0.62rem; font-weight: 700; color: #8F887C;">avg kcal</div>
                </div>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin: 0.6rem 0 1.2rem 0;">
    <div style="background: #FAF8F3; border: 1.5px solid #EBE2CF; border-radius: 18px; padding: 12px 6px; text-align: center;">
        <div style="font-size: 0.74rem; font-weight: 700; color: #8F887C; margin-bottom: 2px;">🤍 Protein</div>
        <div style="font-family: 'Outfit', sans-serif; color: #1E1B16; font-size: 1.25rem; font-weight: 900;">{protein_g}g</div>
    </div>
    <div style="background: #FAF8F3; border: 1.5px solid #EBE2CF; border-radius: 18px; padding: 12px 6px; text-align: center;">
        <div style="font-size: 0.74rem; font-weight: 700; color: #8F887C; margin-bottom: 2px;">🌾 Carbs</div>
        <div style="font-family: 'Outfit', sans-serif; color: #1E1B16; font-size: 1.25rem; font-weight: 900;">{carbs_g}g</div>
    </div>
    <div style="background: #FAF8F3; border: 1.5px solid #EBE2CF; border-radius: 18px; padding: 12px 6px; text-align: center;">
        <div style="font-size: 0.74rem; font-weight: 700; color: #8F887C; margin-bottom: 2px;">🧈 Fat</div>
        <div style="font-family: 'Outfit', sans-serif; color: #1E1B16; font-size: 1.25rem; font-weight: 900;">{fat_g}g</div>
    </div>
</div>""",
        unsafe_allow_html=True,
    )


def render_confidence_bar(confidence):
    pct = confidence * 100
    st.markdown(
        f"""<div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 0.6rem;">
            <span style="font-size: 0.82rem; color: #8F887C; font-weight: 600;">Recognition confidence</span>
            <span style="font-family: 'Outfit', sans-serif; font-size: 0.92rem; font-weight: 800; color: #E5A93B;">{pct:.0f}%</span>
        </div>
        <div style="height: 7px; border-radius: 999px; background: #EFEAE0; overflow: hidden; margin: 0.35rem 0 0.8rem 0;">
            <div style="width: {pct:.1f}%; height: 100%; background: linear-gradient(90deg, #F3C36A, #E5A93B); border-radius: 999px;"></div>
        </div>""",
        unsafe_allow_html=True,
    )


# ============================================================================
# 5. STREAMLIT APP STATE & ROUTING
# ============================================================================

st.set_page_config(
    page_title="GulfBite",
    page_icon="🍲",
    layout="centered",
    initial_sidebar_state="collapsed",
)
inject_theme()

if "stage" not in st.session_state:
    st.session_state.stage = "onboarding"
if "triggered" not in st.session_state:
    st.session_state.triggered = False
if "image" not in st.session_state:
    st.session_state.image = None
if "annotated_image" not in st.session_state:
    st.session_state.annotated_image = None
if "cnn_class" not in st.session_state:
    st.session_state.cnn_class = None
if "cnn_confidence" not in st.session_state:
    st.session_state.cnn_confidence = None
if "candidates" not in st.session_state:
    st.session_state.candidates = None
if "yolo_suggestion" not in st.session_state:
    st.session_state.yolo_suggestion = None
if "yolo_gate_status" not in st.session_state:
    st.session_state.yolo_gate_status = None
if "tier_used" not in st.session_state:
    st.session_state.tier_used = None
if "final_dish" not in st.session_state:
    st.session_state.final_dish = None
if "portion_size" not in st.session_state:
    st.session_state.portion_size = "M"
if "main_section" not in st.session_state:
    st.session_state.main_section = "🏠 Home"
if "scan_source" not in st.session_state:
    st.session_state.scan_source = "📷 Camera"


def reset(open_scan=True):
    st.session_state.stage = "main"
    st.session_state.triggered = False
    st.session_state.image = None
    st.session_state.annotated_image = None
    st.session_state.cnn_class = None
    st.session_state.cnn_confidence = None
    st.session_state.candidates = None
    st.session_state.yolo_suggestion = None
    st.session_state.yolo_gate_status = None
    st.session_state.tier_used = None
    st.session_state.final_dish = None
    st.session_state.portion_size = "M"
    st.session_state.main_section = "📷 Scan" if open_scan else "🏠 Home"
    st.session_state.pending_main_section = st.session_state.main_section


# ============================================================================
# 6. LOAD MODELS INTO CACHE
# ============================================================================

try:
    cnn_model, idx_to_class, yolo_model, ingredient_cache = load_models()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()


# ============================================================================
# 7. SCREEN 0: "GET STARTED" ONBOARDING HERO
# ============================================================================

if st.session_state.stage == "onboarding":
    render_header(compact=True)

    # Visual sample of the supported Gulf-food domain. Labels intentionally avoid
    # presenting fixed calories before a real plate has been analyzed.
    st.markdown(
        """<div class="gulf-grid-collage">
    <div class="grid-cell" style="background-image:url('https://images.unsplash.com/photo-1563379091339-03b21ab4a4f8?auto=format&fit=crop&w=700&q=82');">
        <div class="grid-cell-overlay"></div>
        <div class="micro-pill" style="bottom:10px;left:10px;"><span class="pill-dot"></span><span>Machboos</span></div>
    </div>
    <div class="grid-cell" style="background-image:url('https://images.unsplash.com/photo-1529006557810-274b9b2fc783?auto=format&fit=crop&w=500&q=82');">
        <div class="grid-cell-overlay"></div>
        <div class="micro-pill" style="top:10px;left:10px;"><span class="pill-dot"></span><span>Shawarma</span></div>
    </div>
    <div class="grid-cell" style="background-image:url('https://images.unsplash.com/photo-1576092768241-dec231879fc3?auto=format&fit=crop&w=500&q=82');">
        <div class="grid-cell-overlay"></div>
        <div class="micro-pill" style="bottom:10px;right:10px;"><span class="pill-dot"></span><span>Karak</span></div>
    </div>
</div>
<div style="padding:.1rem .2rem .7rem .2rem;">
    <div style="display:inline-flex;align-items:center;gap:6px;padding:5px 9px;border-radius:999px;background:#FFF7E7;border:1px solid #F0D9A8;color:#B9780E;font-size:.70rem;font-weight:800;margin-bottom:9px;">25 Gulf dishes • AI-assisted recognition</div>
    <h1 style="font-family:'Outfit',sans-serif;font-size:2.15rem;font-weight:900;line-height:1.06;color:#1E1B16;margin:0;letter-spacing:-.035em;">Know your Gulf plate.<br><span style="color:#D99A28;">Track it smarter.</span></h1>
    <p style="color:#7C756A;font-size:.88rem;font-weight:500;margin:10px 0 7px 0;line-height:1.5;">Photograph a traditional Gulf dish, verify the AI match when needed, choose your portion, and view estimated calories and macros.</p>
</div>""",
        unsafe_allow_html=True,
    )

    if st.button("Start scanning →", key="btn_get_started", type="primary", use_container_width=True):
        st.session_state.stage = "main"
        st.session_state.main_section = "🏠 Home"
        st.rerun()


# ============================================================================
# 8. SCREEN 1: MAIN NAVIGATION (HOME / MENU / SCAN)
# ============================================================================

elif st.session_state.stage in ["main", "upload"]:
    render_header(compact=True)
    active_section = render_main_navigation()

    if active_section == "🏠 Home":
        st.markdown(
            '<div style="font-family:\'Outfit\',sans-serif;font-size:1.05rem;font-weight:900;color:#1E1B16;margin:5px 0 7px 0;">How GulfBite works</div>',
            unsafe_allow_html=True,
        )
        render_quick_guide()

        with st.container(border=True):
            st.markdown(
                """<div style="padding:2px 2px 5px 2px;">
<div style="display:flex;align-items:center;justify-content:space-between;gap:12px;">
    <div>
        <div style="font-family:'Outfit',sans-serif;font-size:1.18rem;font-weight:900;color:#1E1B16;">Scan your next meal</div>
        <div style="font-size:.79rem;color:#8B8377;line-height:1.45;margin-top:4px;">Best results come from one clear plate photographed from above.</div>
    </div>
    <div style="width:54px;height:54px;border-radius:18px;background:#FFF7E7;border:1px solid #F0D8A5;display:flex;align-items:center;justify-content:center;font-size:1.65rem;flex:0 0 auto;">📷</div>
</div>
</div>""",
                unsafe_allow_html=True,
            )
            if st.button("Scan a plate →", type="primary", use_container_width=True, key="home_scan_cta"):
                st.session_state.pending_main_section = "📷 Scan"
                st.rerun()

        st.markdown(
            """<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:2px;">
<div style="background:#FFFFFF;border:1px solid #EAE2D4;border-radius:17px;padding:12px;">
    <div style="font-size:.68rem;color:#948B7F;font-weight:800;text-transform:uppercase;letter-spacing:.04em;">Recognition</div>
    <div style="font-family:'Outfit',sans-serif;font-size:1rem;font-weight:900;margin-top:3px;">25 Gulf dishes</div>
</div>
<div style="background:#FFFFFF;border:1px solid #EAE2D4;border-radius:17px;padding:12px;">
    <div style="font-size:.68rem;color:#948B7F;font-weight:800;text-transform:uppercase;letter-spacing:.04em;">Output</div>
    <div style="font-family:'Outfit',sans-serif;font-size:1rem;font-weight:900;margin-top:3px;">Calories + macros</div>
</div>
</div>""",
            unsafe_allow_html=True,
        )

    elif active_section == "📖 Menu":
        st.markdown(
            '<div style="font-family:\'Outfit\',sans-serif;font-size:1.05rem;font-weight:900;color:#1E1B16;margin:5px 0 7px 0;">Explore supported dishes</div>',
            unsafe_allow_html=True,
        )
        st.caption("Browse the 25 dishes currently recognized by the model.")
        render_category_squircle_cards()

    else:  # 📷 Scan
        render_segmented_stepper("upload", st.session_state.get("triggered", False))

        with st.container(border=True):
            st.markdown(
                """<div style="display:flex;justify-content:space-between;align-items:flex-start;gap:10px;margin-bottom:9px;">
<div>
    <div style="font-family:'Outfit',sans-serif;font-size:1.12rem;font-weight:900;color:#1E1B16;">Scan your plate</div>
    <div style="font-size:.76rem;color:#91897D;margin-top:2px;">Camera or photo upload</div>
</div>
<span style="background:#FFF7E7;color:#B9780E;font-size:.70rem;font-weight:800;padding:5px 9px;border-radius:999px;border:1px solid #EED8A8;">Top-down works best</span>
</div>""",
                unsafe_allow_html=True,
            )

            image_to_process = render_scan_input()

            st.markdown(
                '<div style="font-size:.72rem;color:#91897D;line-height:1.4;margin-top:2px;">Tip: keep the full plate visible, use good lighting, and avoid heavy filters.</div>',
                unsafe_allow_html=True,
            )

            if image_to_process is not None:
                st.session_state.image = image_to_process
                st.image(image_to_process, caption="Meal preview", use_column_width=True)

                with st.spinner("Recognizing dish and checking visual markers..."):
                    cnn_class, cnn_confidence, margin, entropy = run_cnn(
                        image_to_process, cnn_model, idx_to_class
                    )

                    is_non_food = (
                        cnn_confidence < MIN_CONFIDENCE
                        or margin < MIN_MARGIN
                        or entropy > MAX_ENTROPY
                    )

                    if is_non_food:
                        st.markdown(
                            """<div style="background:#FFF5F3;border:1px solid #F3D0CB;border-radius:16px;padding:15px;margin-top:10px;">
<div style="font-family:'Outfit',sans-serif;font-weight:900;color:#B33C34;">No supported Gulf dish detected</div>
<div style="color:#7D756B;font-size:.80rem;line-height:1.45;margin-top:4px;">Try a clearer top-down image with one traditional Gulf dish filling most of the frame.</div>
</div>""",
                            unsafe_allow_html=True,
                        )
                        st.button("Try another photo", on_click=reset, use_container_width=True)
                        st.stop()

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
                        st.session_state.tier_used = "CNN direct match"
                        st.session_state.stage = "select_portion"
                        st.rerun()
                    else:
                        candidates = get_candidate_group(cnn_class)
                        run_yolo_here = (cnn_class in YOLO_FEATURE_MAP) or (cnn_class == "03_biryani")
                        yolo_suggestion, gate_status = None, None
                        annotated_img = None

                        if run_yolo_here:
                            detections = run_yolov8_with_boxes(image_to_process, yolo_model)
                            if detections:
                                annotated_img = create_ai_decoded_overlay(image_to_process, detections)
                            _, gated, gate_status = map_detections_to_suggestion(detections, candidates)
                            yolo_suggestion = gated[0] if gated else None

                        st.session_state.annotated_image = annotated_img
                        st.session_state.candidates = sorted(candidates)
                        st.session_state.yolo_suggestion = yolo_suggestion
                        st.session_state.yolo_gate_status = gate_status
                        st.session_state.tier_used = "CNN + YOLO + user confirm" if yolo_suggestion else "CNN + user confirm"
                        st.session_state.stage = "confirm_dish"
                        st.rerun()


# ============================================================================
# 9. SCREEN 2: CONFIRM DISH (WITH AI DECODED VISUAL OVERLAY)
# ============================================================================

elif st.session_state.stage == "confirm_dish":
    render_header()
    render_segmented_stepper("confirm_dish", True)

    with st.container(border=True):
        display_img = st.session_state.annotated_image if st.session_state.annotated_image else st.session_state.image
        st.image(
            display_img,
            caption="AI Decoded Ingredients & Markers",
            use_column_width=True,
        )

        cnn_class = st.session_state.cnn_class
        cnn_conf = st.session_state.cnn_confidence
        candidates = st.session_state.candidates
        yolo_suggestion = st.session_state.yolo_suggestion

        st.markdown(
            f"""<div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 0.6rem;">
                <div style="font-family: 'Outfit', sans-serif; font-size: 1.45rem; font-weight: 900; color: #1E1B16;">
                    Initial Match: <span style="color: #E5A93B;">{display_name(cnn_class)}</span>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

        render_confidence_bar(cnn_conf)

        reason = get_group_reason(cnn_class)
        if reason:
            st.markdown(
                f"""<div class="verify-callout">
                    <span style="font-size: 1.1rem; line-height: 1;">🔍</span>
                    <span style="color: #736C61; font-size: 0.84rem; line-height: 1.4;">{reason}</span>
                </div>""",
                unsafe_allow_html=True,
            )

        if yolo_suggestion:
            st.markdown(
                f"""<div class="ingredient-badge">
                    <span style="font-size: 1.1rem;">✨</span>
                    <span>Visual inspection detected marker: <strong style="color: #C28416;">{display_name(yolo_suggestion)}</strong></span>
                </div>""",
                unsafe_allow_html=True,
            )

        default_choice = yolo_suggestion if yolo_suggestion else cnn_class
        default_idx = (
            candidates.index(default_choice)
            if default_choice in candidates
            else 0
        )

        st.markdown(
            '<p style="font-family: \'Outfit\', sans-serif; font-size: 0.88rem; font-weight: 800; color: #1E1B16; margin: 12px 0 6px 0;">Select your dish:</p>',
            unsafe_allow_html=True,
        )

        choice = st.selectbox(
            "Select matching dish:",
            options=candidates,
            format_func=lambda x: f"🍲 {display_name(x)}",
            index=default_idx,
            label_visibility="collapsed",
            key="dish_confirmation_select",
        )

        if st.button("Confirm dish →", type="primary", use_container_width=True):
            st.session_state.final_dish = choice
            st.session_state.stage = "select_portion"
            st.rerun()


# ============================================================================
# 10. SCREEN 3: SELECT PORTION
# ============================================================================

elif st.session_state.stage == "select_portion":
    render_header()
    render_segmented_stepper("select_portion", st.session_state.get("triggered", False))

    with st.container(border=True):
        st.image(
            st.session_state.image,
            caption="Scanned Plate",
            use_column_width=True,
        )

        st.markdown(
            f"""<div style="font-family: 'Outfit', sans-serif; font-size: 1.65rem; font-weight: 900; color: #1E1B16; margin: 0.6rem 0 0.2rem 0;">
                {display_name(st.session_state.final_dish)}
            </div>
            <p style="color: #8F887C; font-size: 0.86rem; font-weight: 500; margin-bottom: 1.2rem;">
                Select your portion size to calculate authentic nutrition values:
            </p>""",
            unsafe_allow_html=True,
        )

        portion_labels = ["Small · ~250g", "Medium · ~400g", "Large · ~550g"]
        portion_lookup = {
            "Small · ~250g": "S",
            "Medium · ~400g": "M",
            "Large · ~550g": "L",
        }
        default_label = {"S": portion_labels[0], "M": portion_labels[1], "L": portion_labels[2]}.get(
            st.session_state.get("portion_size", "M"), portion_labels[1]
        )
        selected_label = segmented_choice(
            "Choose portion",
            portion_labels,
            default=default_label,
            key="portion_segment",
        )
        selected_p = portion_lookup.get(selected_label, "M")

        st.markdown(
            '<div style="display:flex;justify-content:space-between;font-size:.70rem;color:#938B7F;margin-top:4px;"><span>Light meal</span><span>Typical plate</span><span>Large serving</span></div>',
            unsafe_allow_html=True,
        )

        if st.button("Calculate nutrition →", type="primary", use_container_width=True):
            st.session_state.portion_size = selected_p
            st.session_state.stage = "result"
            st.rerun()


# ============================================================================
# 11. SCREEN 4: NUTRITIONAL BREAKDOWN RESULT & MACRO RING
# ============================================================================

elif st.session_state.stage == "result":
    render_header()
    render_segmented_stepper("result", st.session_state.get("triggered", False))

    with st.container(border=True):
        st.image(
            st.session_state.image,
            caption="Scanned Plate",
            use_column_width=True,
        )

        dish = st.session_state.get("final_dish")
        if not dish:
            dish = st.session_state.get("cnn_class", "01_machboos")

        nutrition = estimate_nutrition(
            dish, st.session_state.portion_size, ingredient_cache
        )
        lo, hi = nutrition["calories_range"]

        st.markdown(
            f"""<div style="display: flex; justify-content: space-between; align-items: center; margin-top: 0.6rem;">
                <div>
                    <div style="font-family: 'Outfit', sans-serif; font-size: 1.65rem; font-weight: 900; color: #1E1B16;">{display_name(dish)}</div>
                    <div style="color: #8F887C; font-size: 0.84rem; font-weight: 600;">Portion size: <strong style="color:#C28416;">{PORTION_LABELS[st.session_state.portion_size]}</strong></div>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

        blurb = DISH_BLURBS.get(dish)
        if blurb:
            st.markdown(
                f'<p style="color: #736C61; font-size: 0.84rem; line-height: 1.45; margin: 0.6rem 0 0 0;">{blurb}</p>',
                unsafe_allow_html=True,
            )

        render_culinary_badges(dish)
        render_macro_donut_and_cards(
            nutrition["protein_g"], nutrition["carbs_g"], nutrition["fat_g"], lo, hi
        )

        if nutrition["missing_ingredients"]:
            st.warning(
                f"Missing standard data for: {', '.join(nutrition['missing_ingredients'])}."
            )

        st.markdown("<div style='margin-top: 1.2rem;'></div>", unsafe_allow_html=True)

        tab_correct, tab_tech = st.tabs(["✏️ Edit Dish", "⚙️ Pipeline Info"])

        with tab_correct:
            all_dishes = sorted(DISH_RECIPES.keys(), key=display_name)
            current_idx = all_dishes.index(dish) if dish in all_dishes else 0

            corrected = st.selectbox(
                "Select correct dish:",
                options=all_dishes,
                format_func=display_name,
                index=current_idx,
                label_visibility="collapsed",
            )
            if st.button("Update Dish", type="primary", use_container_width=True):
                st.session_state.final_dish = corrected
                st.session_state.tier_used = "User correction"
                st.rerun()

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

    st.write("")
    st.button("📸 Scan another plate", on_click=reset, use_container_width=True)
