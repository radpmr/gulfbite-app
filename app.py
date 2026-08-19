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

MACHBOOS_ONBOARDING_URI = "data:image/jpeg;base64,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2HpjWp8c+EPF+h2FpqLfwlfaO31kSSUVVWV0LJO7/AOVCpzt/TSNx98PPHXwv1i18NeeN7zw/VwXzT2C/1lVRz0dVUxxgsk0ZLYY5H4f76Z6fU6fU21W7brNq3bdtxj98/rE2p0up09ttmvtsttps3bfOM8f+M82eY77bA56e+vMnJznOMHXXBxnGN++Ndfa+/trQzOcjnJ8ue4123b5A/trrg4P5j2wddZw2Rj6nroMznGc7ffXP541zzYxnrjp31z7gn9ddExI4/QfXruBznGMdNc4zkE+v110Mev0xrpkTnPPtvrkk7+u+uu/vn12668zzjOMbfbXSYzIznd9tdjAHTb667zgg++uMjJO2+NenOZwB06/TXmcdMY3xjrrljjGMHfrrjjY7bfrroJzzI55h1PXrnXWc+o211nPfG+euutj7fbXXUZkcf0z7a93OB023zr0HG38tcEev066mcxI5xgnfI3O/vrmx8u25+muuecYznfXXrDbPr1xqJMgzk56df66935cffbrrjOCfv7Z1x2Pfnr6aTMScZznnvrpTjAxtvvrmvOcfrnpvrrknG/pjb1+uqic4zM68e/P114MYzjONtd7gnONj7a5yM59emmuuMTJ85xz741x83669ztg7j9NdEdM43ON+muuYkc65zvnrjnXTbYHOPrrlxnOeu+2ukzmRzkf16jrhsn1I21znPTf1xrhxj379NdEznP/2Q=="

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
    padding: 1rem 1rem 1.2rem 1rem !important;
}

/* Reduce Streamlit's default vertical gaps so the app feels like a mobile product. */
[data-testid="stVerticalBlock"] { gap: .55rem !important; }
[data-testid="stElementContainer"] { margin-bottom: 0 !important; }

/* General cards */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255,255,255,.96) !important;
    border: 1px solid rgba(229,169,59,.20) !important;
    border-radius: 22px !important;
    padding: 0.85rem !important;
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

/* Selects — clearer text and stronger control contrast. */
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    min-height: 48px !important;
    border-radius: 15px !important;
    border-color: #E5D8BF !important;
    background: #FFFDF9 !important;
    box-shadow: none !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] span,
[data-testid="stSelectbox"] [data-baseweb="select"] div {
    color: var(--ink) !important;
    opacity: 1 !important;
}
[data-testid="stSelectbox"] label,
[data-testid="stSelectbox"] label p {
    color: #8A8174 !important;
    font-weight: 700 !important;
}
[data-testid="stSelectbox"] svg {
    color: #B88423 !important;
    fill: #B88423 !important;
}
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
    color: #81786C !important;
    opacity: 1 !important;
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
    gap: 6px !important;
}
[data-testid="stSegmentedControl"] button {
    flex: 1 1 0 !important;
    min-height: 42px !important;
    border-radius: 13px !important;
    border: 1px solid #E9E0D1 !important;
    background: #FFFDF9 !important;
    color: #7D7468 !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 800 !important;
    box-shadow: none !important;
}
[data-testid="stSegmentedControl"] button[aria-pressed="true"],
[data-testid="stSegmentedControl"] button[data-selected="true"] {
    background: linear-gradient(135deg,#F5C56D 0%,#E5A93B 100%) !important;
    border-color: #E5A93B !important;
    color: #171007 !important;
    box-shadow: 0 5px 14px rgba(229,169,59,.22) !important;
}
[data-testid="stSegmentedControl"] input,
[data-testid="stSegmentedControl"] [data-baseweb="radio"] > div:first-child {
    display: none !important;
}

/* Older Streamlit versions fallback radios */
div[data-testid="stRadio"] > div[role="radiogroup"] {
    display: flex !important;
    flex-direction: row !important;
    gap: 6px !important;
    width: 100% !important;
}
div[data-testid="stRadio"] label[data-baseweb="radio"] {
    flex: 1 1 0 !important;
    min-width: 0 !important;
    min-height: 42px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 9px 7px !important;
    margin: 0 !important;
    border: 1px solid #E9E0D1 !important;
    border-radius: 13px !important;
    background: #FFFDF9 !important;
    cursor: pointer !important;
}
div[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child {
    display: none !important;
}
div[data-testid="stRadio"] label[data-baseweb="radio"] span,
div[data-testid="stRadio"] label[data-baseweb="radio"] p {
    color: #7D7468 !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: .82rem !important;
    font-weight: 800 !important;
    white-space: nowrap !important;
}
div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
    background: linear-gradient(135deg,#F5C56D 0%,#E5A93B 100%) !important;
    border-color: #E5A93B !important;
    box-shadow: 0 5px 14px rgba(229,169,59,.22) !important;
}
div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) span,
div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) p {
    color: #171007 !important;
}

/* =========================================================================
   GULFBITE FIXED BOTTOM TAB BAR — pure HTML, independent of Streamlit DOM.
   ========================================================================= */
.gulf-bottom-nav {
    position: fixed !important;
    left: 50% !important;
    bottom: max(6px, env(safe-area-inset-bottom)) !important;
    transform: translateX(-50%) !important;
    width: min(430px, calc(100vw - 16px)) !important;
    z-index: 2147483000 !important;
    display: grid !important;
    grid-template-columns: repeat(3, minmax(0, 1fr)) !important;
    gap: 6px !important;
    padding: 6px !important;
    box-sizing: border-box !important;
    border: 1px solid rgba(226,216,198,.96) !important;
    border-radius: 20px !important;
    background: rgba(255,253,249,.97) !important;
    box-shadow: 0 12px 34px rgba(58,40,12,.18), 0 2px 6px rgba(0,0,0,.05) !important;
    -webkit-backdrop-filter: blur(18px) saturate(1.15) !important;
    backdrop-filter: blur(18px) saturate(1.15) !important;
}
.gulf-bottom-tab {
    min-width: 0 !important;
    min-height: 52px !important;
    border-radius: 14px !important;
    border: 1px solid #E7DED0 !important;
    background: #FFFDF9 !important;
    color: #81796E !important;
    text-decoration: none !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 3px !important;
    font-family: 'Outfit', sans-serif !important;
    box-sizing: border-box !important;
    -webkit-tap-highlight-color: transparent !important;
}
.gulf-bottom-tab:hover,
.gulf-bottom-tab:focus,
.gulf-bottom-tab:visited {
    color: #81796E !important;
    text-decoration: none !important;
}
.gulf-bottom-tab.active {
    background: linear-gradient(135deg,#F5C56D 0%,#E5A93B 100%) !important;
    border-color: #E5A93B !important;
    color: #171007 !important;
    box-shadow: 0 5px 12px rgba(229,169,59,.22) !important;
}
.gulf-bottom-tab.active:hover,
.gulf-bottom-tab.active:focus,
.gulf-bottom-tab.active:visited {
    color: #171007 !important;
}
.gulf-bottom-icon {
    width: 18px !important;
    height: 18px !important;
    display: block !important;
}
.gulf-bottom-icon svg {
    width: 18px !important;
    height: 18px !important;
    display: block !important;
    fill: none !important;
    stroke: currentColor !important;
    stroke-width: 2 !important;
    stroke-linecap: round !important;
    stroke-linejoin: round !important;
}
.gulf-bottom-label {
    font-size: .70rem !important;
    font-weight: 800 !important;
    line-height: 1 !important;
    white-space: nowrap !important;
}

/* Ensure container padding matches the compact bottom bar */
.block-container {
    padding-bottom: calc(5.8rem + env(safe-area-inset-bottom)) !important;
}

@media (max-width: 480px) {
    .gulf-bottom-nav {
        width: calc(100vw - 12px) !important;
        bottom: max(4px, env(safe-area-inset-bottom)) !important;
        padding: 5px !important;
        gap: 4px !important;
    }
    .gulf-bottom-tab {
        min-height: 48px !important;
    }
}

/* Upload zone */
[data-testid="stFileUploaderDropzone"] {
    min-height: 130px !important;
    padding: 1rem 1rem !important;
    border: 1.5px dashed #DBC99E !important;
    border-radius: 16px !important;
    background: #FBF8F1 !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--gold) !important;
    background: #FFF9EC !important;
}
[data-testid="stFileUploaderDropzone"] svg {
    color: var(--gold) !important;
    fill: var(--gold) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] span {
    color: var(--ink) !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 800 !important;
}

/* Camera input */
[data-testid="stCameraInput"] {
    border-radius: 16px;
    overflow: hidden;
}

/* Secondary result tabs */
div[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 6px !important;
    border-bottom: 1px solid var(--line) !important;
}
div[data-testid="stTabs"] [data-baseweb="tab"] {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 800 !important;
    color: #8A8275 !important;
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
    padding: 10px 12px;
    margin: 8px 0 10px 0;
}

.ingredient-badge {
    display: flex;
    align-items: center;
    gap: 8px;
    background: linear-gradient(135deg, #FFF8E9 0%, #FBF1D9 100%);
    border: 1px solid #EEDBB0;
    border-radius: 14px;
    padding: 9px 12px;
    margin: 6px 0 10px 0;
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
    height: 275px;
    display: grid;
    grid-template-columns: 1.08fr .92fr;
    grid-template-rows: 1fr 1fr;
    gap: 5px;
    overflow: hidden;
    border-radius: 22px;
    background: #E9E0D0;
    box-shadow: 0 18px 34px -20px rgba(0,0,0,.32);
    margin-bottom: 0.8rem;
}
.grid-cell {
    position: relative;
    overflow: hidden;
    background: #E9E0D0;
}
.grid-food-photo {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    object-fit: cover;
    object-position: center;
    display: block;
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
    padding: 3px 7px;
    border-radius: 999px;
    background: rgba(255,255,255,.93);
    border: 1px solid rgba(255,255,255,.9);
    box-shadow: 0 4px 14px rgba(0,0,0,.14);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    color: var(--ink);
    font-family: 'Outfit', sans-serif;
    font-size: .62rem;
    font-weight: 800;
    white-space: nowrap;
}
.pill-dot {
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: var(--gold);
    flex: 0 0 auto;
}

/* Small screens */
@media (max-width: 480px) {
    .block-container { padding: .65rem .65rem 1rem .65rem !important; }
    [data-testid="stVerticalBlockBorderWrapper"] { border-radius: 18px !important; padding: .8rem !important; }
    .gulf-grid-collage { height: 250px; border-radius: 18px; }
}
</style>""",
        unsafe_allow_html=True,
    )


# ============================================================================
# 4. APP NAVIGATION, STEPPERS & VISUALIZERS
# ============================================================================

def segmented_choice(
    label,
    options,
    default=None,
    key=None,
    label_visibility="collapsed",
    format_func=None,
):
    """Use Streamlit segmented_control when available, with a clean radio fallback."""
    if hasattr(st, "segmented_control"):
        kwargs = {
            "label": label,
            "options": options,
            "key": key,
            "selection_mode": "single",
            "label_visibility": label_visibility,
            "width": "stretch",
        }
        if format_func is not None:
            kwargs["format_func"] = format_func
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
        format_func=(format_func if format_func is not None else str),
    )


def line_icon(name: str, size: int = 24, color: str = "#D99926") -> str:
    """Small inline SVG icon set used by GulfBite cards and callouts."""
    icons = {
        "camera": (
            '<path d="M4 7.5h3l1.4-2h7.2l1.4 2h3a2 2 0 0 1 2 2v8.5a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V9.5a2 2 0 0 1 2-2Z"/>'
            '<circle cx="12" cy="13.5" r="4"/>'
        ),
        "sparkles": (
            '<path d="m12 3 1.15 3.35L16.5 7.5l-3.35 1.15L12 12l-1.15-3.35L7.5 7.5l3.35-1.15L12 3Z"/>'
            '<path d="m18.5 12.5.7 2.05 2.05.7-2.05.7-.7 2.05-.7-2.05-2.05-.7 2.05-.7.7-2.05Z"/>'
            '<path d="m5.5 13 .8 2.3 2.2.8-2.2.8-.8 2.3-.8-2.3-2.2-.8 2.2-.8.8-2.3Z"/>'
        ),
        "nutrition": (
            '<circle cx="12" cy="12" r="8"/>'
            '<path d="M12 4v8l5.7 5.7"/>'
            '<path d="M12 12 6.3 17.7"/>'
        ),
        "search": (
            '<circle cx="10.5" cy="10.5" r="5.5"/>'
            '<path d="m15 15 5 5"/>'
        ),
    }
    paths = icons.get(name, icons["camera"])
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'xmlns="http://www.w3.org/2000/svg" aria-hidden="true" style="display:block;">'
        f'<g stroke="{color}" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">{paths}</g></svg>'
    )


def onboarding_calories(dish_key: str, ingredient_cache, fallback: int) -> int:
    """Use GulfBite's own medium-portion nutrition estimator for onboarding sample kcal."""
    try:
        nutrition = estimate_nutrition(dish_key, "M", ingredient_cache)
        low, high = nutrition["calories_range"]
        if low > 0 and high > 0:
            return round((low + high) / 2)
    except Exception:
        pass
    return fallback


def render_header(compact: bool = True):
    title_size = "1.45rem" if compact else "1.75rem"
    subtitle = "Gulf cuisine recognition • calories • macros"
    st.markdown(
        f"""<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:.35rem;">
<div style="display:flex;align-items:center;gap:9px;min-width:0;">
    <div style="width:38px;height:38px;border-radius:12px;background:linear-gradient(135deg,#F4C66F,#E5A93B);display:flex;align-items:center;justify-content:center;box-shadow:0 6px 14px rgba(229,169,59,.22);font-family:'Outfit',sans-serif;font-weight:900;color:#1A1305;font-size:.9rem;">GB</div>
    <div style="min-width:0;">
        <div style="font-family:'Outfit',sans-serif;font-size:{title_size};font-weight:900;line-height:1;letter-spacing:-.025em;color:#1E1B16;white-space:nowrap;"><span style="color:#D99926;">GulfBite</span><span style="display:inline-block;margin-left:5px;">AI</span></div>
        <div style="font-size:.68rem;color:#91897D;font-weight:600;margin-top:3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:250px;">{subtitle}</div>
    </div>
</div>
<div style="display:flex;align-items:center;gap:6px;">
    <div style="width:34px;height:34px;border-radius:11px;background:#FFFFFF;border:1px solid #EBE4D8;display:flex;align-items:center;justify-content:center;position:relative;box-shadow:0 3px 10px rgba(0,0,0,.03);">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9" stroke="#D99926" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M13.73 21a2 2 0 0 1-3.46 0" stroke="#D99926" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
        <span style="position:absolute;top:7px;right:7px;width:5px;height:5px;background:#E5553F;border-radius:50%;border:1px solid white;"></span>
    </div>
</div>
</div>""",
        unsafe_allow_html=True,
    )


def _get_nav_query_value():
    """Return ?nav=Home/Menu/Scan across old and new Streamlit versions."""
    try:
        value = st.query_params.get("nav")
        if isinstance(value, (list, tuple)):
            value = value[-1] if value else None
        return value
    except Exception:
        try:
            params = st.experimental_get_query_params()
            value = params.get("nav")
            if isinstance(value, (list, tuple)):
                value = value[-1] if value else None
            return value
        except Exception:
            return None


def render_main_navigation():
    """Render a true fixed HTML bottom tab bar that does not depend on Streamlit widget DOM selectors."""
    nav_options = ["Home", "Menu", "Scan"]

    # Programmatic navigation inside the app takes priority.
    pending = st.session_state.pop("pending_main_section", None)
    if pending in nav_options:
        st.session_state.main_section = pending

    requested = _get_nav_query_value()
    last_requested = st.session_state.get("_last_nav_query")
    if requested in nav_options and requested != last_requested:
        st.session_state.main_section = requested
        st.session_state.stage = "main"
        st.session_state._last_nav_query = requested

    current = st.session_state.get("main_section", "Home")
    if current not in nav_options:
        current = "Home"
        st.session_state.main_section = current

    icons = {
        "Home": """<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3.5 10.5 12 3l8.5 7.5v9A1.5 1.5 0 0 1 19 21h-5v-6h-4v6H5a1.5 1.5 0 0 1-1.5-1.5z"/></svg>""",
        "Menu": """<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 5.5A2.5 2.5 0 0 1 6.5 3H11v16H6.5A2.5 2.5 0 0 0 4 21.5zM20 5.5A2.5 2.5 0 0 0 17.5 3H13v16h4.5a2.5 2.5 0 0 1 2.5 2.5z"/></svg>""",
        "Scan": """<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8.2 6 9.6 4h4.8l1.4 2H19a2.5 2.5 0 0 1 2.5 2.5v9A2.5 2.5 0 0 1 19 20H5a2.5 2.5 0 0 1-2.5-2.5v-9A2.5 2.5 0 0 1 5 6z"/><circle cx="12" cy="13" r="4"/></svg>""",
    }

    items = []
    for label in nav_options:
        active = " active" if label == current else ""
        items.append(
            f'<a class="gulf-bottom-tab{active}" href="?nav={label}" target="_self" aria-label="{label}">'
            f'<span class="gulf-bottom-icon">{icons[label]}</span>'
            f'<span class="gulf-bottom-label">{label}</span>'
            '</a>'
        )

    st.markdown(
        '<nav class="gulf-bottom-nav" aria-label="Main navigation">'
        + ''.join(items)
        + '</nav>',
        unsafe_allow_html=True,
    )

    return current


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
        if i < active_idx:
            bg = "#F2D79D"
            shadow = ""
            label_text = f"✓ {label}"
            color = "#B8862E"
            weight = "700"
        elif i == active_idx:
            bg = "linear-gradient(90deg,#F3C36A,#E5A93B)"
            shadow = "box-shadow:0 2px 8px rgba(229,169,59,.24);"
            label_text = f"{i+1}. {label}"
            color = "#1E1B16"
            weight = "900"
        else:
            bg = "#EAE4D8"
            shadow = ""
            label_text = f"{i+1}. {label}"
            color = "#9A9286"
            weight = "650"

        segments.append(
            f'<div style="flex:1;height:4px;border-radius:999px;background:{bg};{shadow}"></div>'
        )
        labels.append(
            f"<span style=\"font-family:'Outfit',sans-serif;font-size:.66rem;font-weight:{weight};color:{color};\">{label_text}</span>"
        )

    st.markdown(
        '<div style="margin:.1rem 0 .7rem 0;">'
        '<div style="display:flex;gap:5px;margin-bottom:5px;">' + ''.join(segments) + '</div>'
        '<div style="display:flex;justify-content:space-between;padding:0 1px;">' + ''.join(labels) + '</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def render_quick_guide():
    camera_icon = line_icon("camera", 22)
    sparkle_icon = line_icon("sparkles", 22)
    nutrition_icon = line_icon("nutrition", 22)
    st.markdown(
        f"""<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin:2px 0 10px 0;">
<div style="background:#FFFFFF;border:1px solid #EAE2D4;border-radius:15px;padding:10px 5px;text-align:center;">
    <div style="width:26px;height:26px;margin:0 auto 3px auto;display:flex;align-items:center;justify-content:center;">{camera_icon}</div>
    <div style="font-family:'Outfit',sans-serif;font-weight:900;font-size:.73rem;color:#1E1B16;">Snap</div>
    <div style="font-size:.62rem;color:#90887C;line-height:1.2;margin-top:1px;">Top-down plate</div>
</div>
<div style="background:#FFFFFF;border:1px solid #EAE2D4;border-radius:15px;padding:10px 5px;text-align:center;">
    <div style="width:26px;height:26px;margin:0 auto 3px auto;display:flex;align-items:center;justify-content:center;">{sparkle_icon}</div>
    <div style="font-family:'Outfit',sans-serif;font-weight:900;font-size:.73rem;color:#1E1B16;">Recognize</div>
    <div style="font-size:.62rem;color:#90887C;line-height:1.2;margin-top:1px;">AI dish check</div>
</div>
<div style="background:#FFFFFF;border:1px solid #EAE2D4;border-radius:15px;padding:10px 5px;text-align:center;">
    <div style="width:26px;height:26px;margin:0 auto 3px auto;display:flex;align-items:center;justify-content:center;">{nutrition_icon}</div>
    <div style="font-family:'Outfit',sans-serif;font-weight:900;font-size:.73rem;color:#1E1B16;">Track</div>
    <div style="font-size:.62rem;color:#90887C;line-height:1.2;margin-top:1px;">Calories + macros</div>
</div>
</div>""",
        unsafe_allow_html=True,
    )


def render_scan_input():
    """Render reliable Camera / Upload buttons and return a PIL image or None."""
    sources = ["Camera", "Upload"] if hasattr(st, "camera_input") else ["Upload"]

    current = st.session_state.get("scan_source", sources[0])
    if current not in sources:
        current = sources[0]
        st.session_state.scan_source = current

    if len(sources) == 2:
        c1, c2 = st.columns(2, gap="small")
        with c1:
            if st.button(
                "Camera",
                key="source_camera",
                type="primary" if current == "Camera" else "secondary",
                use_container_width=True,
            ):
                if current != "Camera":
                    st.session_state.scan_source = "Camera"
                    st.rerun()
        with c2:
            if st.button(
                "Upload",
                key="source_upload",
                type="primary" if current == "Upload" else "secondary",
                use_container_width=True,
            ):
                if current != "Upload":
                    st.session_state.scan_source = "Upload"
                    st.rerun()
    else:
        st.session_state.scan_source = "Upload"

    image_file = None
    if st.session_state.scan_source == "Camera" and hasattr(st, "camera_input"):
        image_file = st.camera_input(
            "Take a clear top-down photo",
            label_visibility="collapsed",
            key="meal_camera",
        )
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
        '<div style="margin-top: 10px; margin-bottom: 4px; font-size: 0.74rem; font-weight: 800; color: #8F887C; text-transform: uppercase; letter-spacing: 0.05em;">Choose Recipe</div>',
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
            f"""<div style="background: #FAF8F3; border: 1.5px solid #EBE2CF; border-radius: 20px; padding: 12px 14px; margin-top: 8px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="font-family: 'Outfit', sans-serif; font-weight: 900; color: #1E1B16; font-size: 1.05rem;">{display_name(selected_dish)}</div>
                    <span style="background: #FDF6E9; color: #C28416; font-size: 0.72rem; font-weight: 800; padding: 3px 9px; border-radius: 999px; border: 1px solid #F5E3BE;">⏱️ {meta['time']}</span>
                </div>
                <p style="color: #736C61; font-size: 0.82rem; line-height: 1.4; margin: 6px 0 8px 0;">{blurb}</p>
                <div style="display: flex; gap: 5px; flex-wrap: wrap;">
                    <span style="font-size: 0.70rem; font-weight: 700; background: #FFFFFF; border: 1px solid #E2D7C3; padding: 2px 7px; border-radius: 7px;">{meta['spice']}</span>
                    <span style="font-size: 0.70rem; font-weight: 700; background: #FFFFFF; border: 1px solid #E2D7C3; padding: 2px 7px; border-radius: 7px;">{meta['prep']}</span>
                    <span style="font-size: 0.70rem; font-weight: 700; background: #FFFFFF; border: 1px solid #E2D7C3; padding: 2px 7px; border-radius: 7px;">{meta['density']}</span>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )


def render_culinary_badges(dish_class: str):
    meta = DISH_METADATA.get(dish_class, {"spice": "Aromatic 🌶️", "prep": "Slow-Simmered ⏳", "density": "Nutrient Rich 🥗", "time": "45 min"})
    st.markdown(
        f"""<div style="display: flex; justify-content: space-between; gap: 5px; margin: 0.6rem 0 0.8rem 0; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 85px; background: #FAF8F3; border: 1px solid #EBE2CF; border-radius: 13px; padding: 7px 5px; text-align: center;">
                <div style="font-size: 0.65rem; color: #8F887C; font-weight: 700;">FLAVOR</div>
                <div style="font-size: 0.75rem; font-weight: 800; color: #1E1B16; margin-top: 1px;">{meta['spice']}</div>
            </div>
            <div style="flex: 1; min-width: 85px; background: #FAF8F3; border: 1px solid #EBE2CF; border-radius: 13px; padding: 7px 5px; text-align: center;">
                <div style="font-size: 0.65rem; color: #8F887C; font-weight: 700;">COOK STYLE</div>
                <div style="font-size: 0.75rem; font-weight: 800; color: #1E1B16; margin-top: 1px;">{meta['prep']}</div>
            </div>
            <div style="flex: 1; min-width: 85px; background: #FAF8F3; border: 1px solid #EBE2CF; border-radius: 13px; padding: 7px 5px; text-align: center;">
                <div style="font-size: 0.65rem; color: #8F887C; font-weight: 700;">PROFILE</div>
                <div style="font-size: 0.75rem; font-weight: 800; color: #1E1B16; margin-top: 1px;">{meta['density']}</div>
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

    svg_donut = f"""<svg width="105" height="105" viewBox="0 0 100 100" style="transform: rotate(-90deg);">
        <circle cx="50" cy="50" r="42" fill="transparent" stroke="#EFEAE0" stroke-width="12"/>
        <circle cx="50" cy="50" r="42" fill="transparent" stroke="#E5A93B" stroke-width="12" stroke-dasharray="{len_p:.2f} {circumference:.2f}" stroke-dashoffset="{off_p:.2f}" stroke-linecap="round"/>
        <circle cx="50" cy="50" r="42" fill="transparent" stroke="#059669" stroke-width="12" stroke-dasharray="{len_c:.2f} {circumference:.2f}" stroke-dashoffset="{off_c:.2f}" stroke-linecap="round"/>
        <circle cx="50" cy="50" r="42" fill="transparent" stroke="#FF5A1F" stroke-width="12" stroke-dasharray="{len_f:.2f} {circumference:.2f}" stroke-dashoffset="{off_f:.2f}" stroke-linecap="round"/>
    </svg>"""

    avg_cal = round((lo + hi) / 2)

    st.markdown(
        f"""<div style="background: linear-gradient(135deg, #FDF9EE 0%, #FAF3DE 100%); border: 1.5px solid #F3E0B5; border-radius: 24px; padding: 1rem 1.1rem; margin: 0.7rem 0; display: flex; align-items: center; justify-content: space-between;">
            <div>
                <div style="font-size: 0.72rem; font-weight: 800; color: #D19428; text-transform: uppercase; letter-spacing: 0.05em;">Estimated Energy</div>
                <div style="font-family: 'Outfit', sans-serif; font-size: 1.95rem; font-weight: 900; color: #1E1B16; line-height: 1.1; margin: 2px 0 5px 0;">
                    {lo}&ndash;{hi} <span style="font-size: 0.9rem; font-weight: 600; color: #8F887C;">kcal</span>
                </div>
                <div style="display: flex; gap: 7px; font-size: 0.70rem; font-weight: 800;">
                    <span style="color: #E5A93B;">● Prot {pct_p*100:.0f}%</span>
                    <span style="color: #059669;">● Carb {pct_c*100:.0f}%</span>
                    <span style="color: #FF5A1F;">● Fat {pct_f*100:.0f}%</span>
                </div>
            </div>
            <div style="position: relative; width: 105px; height: 105px; display: flex; align-items: center; justify-content: center;">
                {svg_donut}
                <div style="position: absolute; text-align: center; transform: rotate(0deg);">
                    <div style="font-family: 'Outfit', sans-serif; font-weight: 900; font-size: 1.05rem; color: #1E1B16; line-height: 1;">{avg_cal}</div>
                    <div style="font-size: 0.60rem; font-weight: 700; color: #8F887C;">avg kcal</div>
                </div>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 7px; margin: 0.5rem 0 1rem 0;">
    <div style="background: #FAF8F3; border: 1.5px solid #EBE2CF; border-radius: 16px; padding: 10px 5px; text-align: center;">
        <div style="font-size: 0.72rem; font-weight: 700; color: #8F887C; margin-bottom: 2px;">🤍 Protein</div>
        <div style="font-family: 'Outfit', sans-serif; color: #1E1B16; font-size: 1.2rem; font-weight: 900;">{protein_g}g</div>
    </div>
    <div style="background: #FAF8F3; border: 1.5px solid #EBE2CF; border-radius: 16px; padding: 10px 5px; text-align: center;">
        <div style="font-size: 0.72rem; font-weight: 700; color: #8F887C; margin-bottom: 2px;">🌾 Carbs</div>
        <div style="font-family: 'Outfit', sans-serif; color: #1E1B16; font-size: 1.2rem; font-weight: 900;">{carbs_g}g</div>
    </div>
    <div style="background: #FAF8F3; border: 1.5px solid #EBE2CF; border-radius: 16px; padding: 10px 5px; text-align: center;">
        <div style="font-size: 0.72rem; font-weight: 700; color: #8F887C; margin-bottom: 2px;">🧈 Fat</div>
        <div style="font-family: 'Outfit', sans-serif; color: #1E1B16; font-size: 1.2rem; font-weight: 900;">{fat_g}g</div>
    </div>
</div>""",
        unsafe_allow_html=True,
    )


def render_confidence_bar(confidence):
    pct = confidence * 100
    st.markdown(
        f"""<div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 0.5rem;">
            <span style="font-size: 0.80rem; color: #8F887C; font-weight: 600;">Recognition confidence</span>
            <span style="font-family: 'Outfit', sans-serif; font-size: 0.90rem; font-weight: 800; color: #E5A93B;">{pct:.0f}%</span>
        </div>
        <div style="height: 6px; border-radius: 999px; background: #EFEAE0; overflow: hidden; margin: 0.3rem 0 0.7rem 0;">
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

_initial_nav = _get_nav_query_value()

if "stage" not in st.session_state:
    st.session_state.stage = "main" if _initial_nav in ["Home", "Menu", "Scan"] else "onboarding"
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
    st.session_state.main_section = _initial_nav if _initial_nav in ["Home", "Menu", "Scan"] else "Home"
if "scan_source" not in st.session_state:
    st.session_state.scan_source = "Camera"


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
    st.session_state.main_section = "Scan" if open_scan else "Home"
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

    machboos_kcal = onboarding_calories("01_machboos", ingredient_cache, 620)
    shawarma_kcal = onboarding_calories("10_shawarma", ingredient_cache, 430)
    karak_kcal = onboarding_calories("25_karak_chai", ingredient_cache, 130)

    st.markdown(
        f"""<div class="gulf-grid-collage">
    <div class="grid-cell">
        <img class="grid-food-photo" src="{MACHBOOS_ONBOARDING_URI}" alt="Machboos">
        <div class="grid-cell-overlay"></div>
        <div class="micro-pill" style="bottom:8px;left:8px;"><span class="pill-dot"></span><span>Machboos · {machboos_kcal} kcal</span></div>
    </div>
    <div class="grid-cell">
        <img class="grid-food-photo" src="https://images.unsplash.com/photo-1529006557810-274b9b2fc783?auto=format&fit=crop&w=700&q=85" alt="Shawarma">
        <div class="grid-cell-overlay"></div>
        <div class="micro-pill" style="top:8px;left:8px;"><span class="pill-dot"></span><span>Shawarma · {shawarma_kcal} kcal</span></div>
    </div>
    <div class="grid-cell">
        <img class="grid-food-photo" src="https://www.timeoutabudhabi.com/cloud/timeoutabudhabi/2022/08/22/Milky-Karak-Cafeteria.jpg" alt="Karak Chai">
        <div class="grid-cell-overlay"></div>
        <div class="micro-pill" style="bottom:8px;right:8px;"><span class="pill-dot"></span><span>Karak Chai · {karak_kcal} kcal</span></div>
    </div>
</div>
<div style="padding:.05rem .1rem .5rem .1rem;">
    <div style="display:inline-flex;align-items:center;gap:5px;padding:4px 8px;border-radius:999px;background:#FFF7E7;border:1px solid #F0D9A8;color:#B9780E;font-size:.68rem;font-weight:800;margin-bottom:7px;">25 Gulf dishes • AI-assisted recognition</div>
    <h1 style="font-family:'Outfit',sans-serif;font-size:1.95rem;font-weight:900;line-height:1.05;color:#1E1B16;margin:0;letter-spacing:-.035em;">Know your Gulf plate.<br><span style="color:#D99A28;">Track it smarter.</span></h1>
    <p style="color:#7C756A;font-size:.84rem;font-weight:500;margin:8px 0 6px 0;line-height:1.45;">Photograph a traditional Gulf dish, verify the AI match when needed, choose your portion, and view estimated calories and macros.</p>
</div>""",
        unsafe_allow_html=True,
    )

    if st.button("Scan your plate →", key="btn_get_started", type="primary", use_container_width=True):
        st.session_state.stage = "main"
        st.session_state.main_section = "Home"
        st.rerun()


# ============================================================================
# 8. SCREEN 1: MAIN NAVIGATION (HOME / MENU / SCAN)
# ============================================================================

elif st.session_state.stage in ["main", "upload"]:
    render_header(compact=True)
    active_section = render_main_navigation()

    if active_section == "Home":
        st.markdown(
            '<div style="font-family:\'Outfit\',sans-serif;font-size:1.0rem;font-weight:900;color:#1E1B16;margin:4px 0 6px 0;">How GulfBite works</div>',
            unsafe_allow_html=True,
        )
        render_quick_guide()

        with st.container(border=True):
            st.markdown(
                f"""<div style="padding:1px 1px 3px 1px;">
<div style="display:flex;align-items:center;justify-content:space-between;gap:10px;">
    <div>
        <div style="font-family:'Outfit',sans-serif;font-size:1.1rem;font-weight:900;color:#1E1B16;">Scan your next meal</div>
        <div style="font-size:.76rem;color:#8B8377;line-height:1.4;margin-top:3px;">Best results come from one clear plate photographed from above.</div>
    </div>
    <div style="width:50px;height:50px;border-radius:16px;background:#FFF7E7;border:1px solid #F0D8A5;display:flex;align-items:center;justify-content:center;flex:0 0 auto;">{line_icon("camera", 26)}</div>
</div>
</div>""",
                unsafe_allow_html=True,
            )
            if st.button("Scan your plate →", type="primary", use_container_width=True, key="home_scan_cta"):
                st.session_state.pending_main_section = "Scan"
                st.rerun()

        st.markdown(
            """<div style="display:grid;grid-template-columns:1fr 1fr;gap:7px;margin-top:1px;">
<div style="background:#FFFFFF;border:1px solid #EAE2D4;border-radius:15px;padding:10px;">
    <div style="font-size:.66rem;color:#948B7F;font-weight:800;text-transform:uppercase;letter-spacing:.04em;">Recognition</div>
    <div style="font-family:'Outfit',sans-serif;font-size:.95rem;font-weight:900;margin-top:2px;">25 Gulf dishes</div>
</div>
<div style="background:#FFFFFF;border:1px solid #EAE2D4;border-radius:15px;padding:10px;">
    <div style="font-size:.66rem;color:#948B7F;font-weight:800;text-transform:uppercase;letter-spacing:.04em;">Output</div>
    <div style="font-family:'Outfit',sans-serif;font-size:.95rem;font-weight:900;margin-top:2px;">Calories + macros</div>
</div>
</div>
<div style="height:10px;"></div>""",
            unsafe_allow_html=True,
        )

    elif active_section == "Menu":
        st.markdown(
            '<div style="font-family:\'Outfit\',sans-serif;font-size:1.0rem;font-weight:900;color:#1E1B16;margin:4px 0 6px 0;">Explore supported dishes</div>',
            unsafe_allow_html=True,
        )
        st.caption("Browse the 25 dishes currently recognized by the model.")
        render_category_squircle_cards()
        st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)

    else:  # 📷 Scan
        render_segmented_stepper("upload", st.session_state.get("triggered", False))

        with st.container(border=True):
            st.markdown(
                """<div style="display:flex;justify-content:space-between;align-items:flex-start;gap:10px;margin-bottom:8px;">
<div>
    <div style="font-family:'Outfit',sans-serif;font-size:1.08rem;font-weight:900;color:#1E1B16;">Scan your plate</div>
    <div style="font-size:.74rem;color:#7E7569;margin-top:1px;font-weight:600;">Camera or photo upload</div>
</div>
<span style="background:#FFF7E7;color:#B9780E;font-size:.68rem;font-weight:800;padding:4px 8px;border-radius:999px;border:1px solid #EED8A8;">Top-down works best</span>
</div>""",
                unsafe_allow_html=True,
            )

            image_to_process = render_scan_input()

            st.markdown(
                '<div style="font-size:.70rem;color:#665F56;line-height:1.35;margin-top:3px;">Tip: keep the full plate visible, use good lighting, and avoid heavy filters.</div>',
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
                            """<div style="background:#FFF5F3;border:1px solid #F3D0CB;border-radius:15px;padding:12px;margin-top:8px;">
<div style="font-family:'Outfit',sans-serif;font-weight:900;color:#B33C34;">No supported Gulf dish detected</div>
<div style="color:#7D756B;font-size:.78rem;line-height:1.4;margin-top:3px;">Try a clearer top-down image with one traditional Gulf dish filling most of the frame.</div>
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
            f"""<div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 0.5rem;">
                <div style="font-family: 'Outfit', sans-serif; font-size: 1.35rem; font-weight: 900; color: #1E1B16;">
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
                    <span style="flex:0 0 auto;margin-top:1px;">{line_icon("search", 18)}</span>
                    <span style="color: #736C61; font-size: 0.82rem; line-height: 1.35;">{reason}</span>
                </div>""",
                unsafe_allow_html=True,
            )

        if yolo_suggestion:
            st.markdown(
                f"""<div class="ingredient-badge">
                    <span style="flex:0 0 auto;">{line_icon("sparkles", 18)}</span>
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
            '<p style="font-family: \'Outfit\', sans-serif; font-size: 0.85rem; font-weight: 800; color: #1E1B16; margin: 10px 0 5px 0;">Select your dish:</p>',
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
            f"""<div style="font-family: 'Outfit', sans-serif; font-size: 1.55rem; font-weight: 900; color: #1E1B16; margin: 0.5rem 0 0.15rem 0;">
                {display_name(st.session_state.final_dish)}
            </div>
            <p style="color: #8F887C; font-size: 0.84rem; font-weight: 500; margin-bottom: 1rem;">
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
            '<div style="display:flex;justify-content:space-between;font-size:.68rem;color:#938B7F;margin-top:3px;"><span>Light meal</span><span>Typical plate</span><span>Large serving</span></div>',
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
            f"""<div style="display: flex; justify-content: space-between; align-items: center; margin-top: 0.5rem;">
                <div>
                    <div style="font-family: 'Outfit', sans-serif; font-size: 1.55rem; font-weight: 900; color: #1E1B16;">{display_name(dish)}</div>
                    <div style="color: #8F887C; font-size: 0.82rem; font-weight: 600;">Portion size: <strong style="color:#C28416;">{PORTION_LABELS[st.session_state.portion_size]}</strong></div>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

        blurb = DISH_BLURBS.get(dish)
        if blurb:
            st.markdown(
                f'<p style="color: #736C61; font-size: 0.82rem; line-height: 1.4; margin: 0.5rem 0 0 0;">{blurb}</p>',
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

        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)

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
                f'<div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #EBE2CF; padding-bottom: 7px;"><span style="color: #8F887C; font-size: 0.82rem;">YOLOv8 Feature</span><span style="color: #1E1B16; font-weight: 700; font-size: 0.85rem;">{display_name(st.session_state.yolo_suggestion)}</span></div>'
                if st.session_state.get("yolo_suggestion")
                else ""
            )

            st.markdown(
                f"""<div style="display: flex; flex-direction: column; gap: 8px; padding: 4px 0;">
<div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #EBE2CF; padding-bottom: 7px;">
<span style="color: #8F887C; font-size: 0.82rem;">CNN Classifier</span>
<span style="color: #1E1B16; font-weight: 700; font-size: 0.85rem;">{display_name(st.session_state.cnn_class)} <span style="color: #E5A93B; font-family: 'JetBrains Mono', monospace;">({st.session_state.cnn_confidence:.0%})</span></span>
</div>
{yolo_row}
<div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #EBE2CF; padding-bottom: 7px;">
<span style="color: #8F887C; font-size: 0.82rem;">Confirmed Dish</span>
<span style="color: #1E1B16; font-weight: 800; font-size: 0.85rem;">{display_name(dish)}</span>
</div>
<div style="display: flex; justify-content: space-between; align-items: center; padding-top: 1px;">
<span style="color: #8F887C; font-size: 0.82rem;">Pipeline Path</span>
<span class="tech-pill">{st.session_state.tier_used}</span>
</div>
</div>""",
                unsafe_allow_html=True,
            )

    st.write("")
    st.button("📸 Scan another plate", on_click=reset, use_container_width=True)
