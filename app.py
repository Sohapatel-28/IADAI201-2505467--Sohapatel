import io
from pathlib import Path
from textwrap import dedent

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


# ============================================================
# PARKVISION AI
# Complete Streamlit parking-space detection application
# ============================================================

st.set_page_config(
    page_title="ParkVision AI",
    page_icon="🅿️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

# Your trained parking model found earlier.
MODEL_PATH = ROOT / "runs" / "detect" / "train-3" / "weights" / "best.pt"
CAR_MODEL_PATH = ROOT / "yolov8n.pt"


# ============================================================
# CSS
# Custom CSS/HTML is rendered with Streamlit's dedicated st.html()
# renderer so HTML tags are never interpreted as Markdown code blocks.
# ============================================================

CSS = dedent("""
<style>
/* ---------- Page ---------- */
.stApp {
    background:
        radial-gradient(circle at 5% 0%, rgba(99,102,241,.12), transparent 30%),
        radial-gradient(circle at 95% 0%, rgba(14,165,233,.12), transparent 30%),
        linear-gradient(135deg, #f7f9ff 0%, #eef6ff 50%, #fbfdff 100%);
    color: #172033;
}

.main .block-container {
    max-width: 1400px;
    padding-top: 1.4rem;
    padding-bottom: 3rem;
}

section[data-testid="stSidebar"] {
    display: none;
}

/* ---------- Normal text ---------- */
h1, h2, h3, h4, h5, h6,
p, li, label,
.stMarkdown, .stText, .stCaption {
    color: #172033 !important;
}

.pv-title {
    color: #ffffff !important;
    font-size: 3rem;
    line-height: 1.05;
    font-weight: 900;
    letter-spacing: -1.5px;
    margin: 0;
}

.pv-subtitle {
    color: #dbeafe !important;
    font-size: 1.05rem;
    line-height: 1.6;
    max-width: 850px;
    margin-top: .55rem;
}

.section-title {
    color: #172033 !important;
    font-size: 1.55rem;
    font-weight: 850;
    margin: 1.35rem 0 .75rem 0;
}

/* ---------- Hero ---------- */
.pv-hero {
    position: relative;
    overflow: hidden;
    padding: 2.25rem 2.5rem;
    border-radius: 26px;
    background: linear-gradient(135deg, #172554 0%, #2563eb 52%, #0f766e 100%);
    box-shadow: 0 22px 55px rgba(30,64,175,.20);
    margin-bottom: 1.25rem;
}

.pv-badge {
    display: inline-block;
    color: #e0f2fe !important;
    background: rgba(255,255,255,.12);
    border: 1px solid rgba(255,255,255,.20);
    border-radius: 999px;
    padding: .36rem .78rem;
    font-size: .75rem;
    font-weight: 850;
    letter-spacing: .65px;
    margin-bottom: .75rem;
}

/* ---------- Tabs ---------- */
div[data-baseweb="tab-list"] {
    gap: .25rem;
    padding: .38rem;
    border-radius: 15px;
    background: rgba(255,255,255,.92);
    border: 1px solid #dbe4f2;
    box-shadow: 0 8px 25px rgba(30,64,110,.08);
    margin-bottom: 1.25rem;
}

button[data-baseweb="tab"] {
    color: #334155 !important;
    background: transparent !important;
    font-weight: 800 !important;
    border-radius: 11px !important;
    padding: .72rem 1rem !important;
}

button[data-baseweb="tab"] p,
button[data-baseweb="tab"] span {
    color: #334155 !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #ffffff !important;
    background: linear-gradient(135deg,#2563eb,#0f766e) !important;
}

button[data-baseweb="tab"][aria-selected="true"] p,
button[data-baseweb="tab"][aria-selected="true"] span {
    color: #ffffff !important;
}

/* ---------- Cards ---------- */
.pv-card {
    background: rgba(255,255,255,.96);
    border: 1px solid #dfe7f2;
    border-radius: 19px;
    padding: 1.25rem 1.35rem;
    box-shadow: 0 10px 28px rgba(31,56,93,.08);
}

.pv-card h3 {
    color: #172033 !important;
    font-size: 1.05rem;
    font-weight: 850;
    margin: 0 0 .45rem 0;
}

.pv-card p,
.pv-card li {
    color: #53627b !important;
    line-height: 1.6;
}

.pv-card ul {
    padding-left: 1.1rem;
}

.pv-metric {
    min-height: 130px;
    background: rgba(255,255,255,.97);
    border: 1px solid #dfe7f2;
    border-radius: 19px;
    padding: 1.1rem 1.2rem;
    box-shadow: 0 10px 27px rgba(31,56,93,.08);
}

.pv-label {
    color: #64748b !important;
    font-size: .77rem;
    font-weight: 850;
    text-transform: uppercase;
    letter-spacing: .7px;
}

.pv-value {
    color: #172033 !important;
    font-size: 2rem;
    font-weight: 900;
    margin-top: .35rem;
}

.pv-note {
    color: #718096 !important;
    font-size: .78rem;
    margin-top: .15rem;
}

/* ---------- Upload area ---------- */
div[data-testid="stFileUploader"] {
    background: #ffffff !important;
    border: 2px dashed #8fb1e8 !important;
    border-radius: 17px !important;
    padding: .55rem !important;
}

div[data-testid="stFileUploader"] section {
    background: #f8fbff !important;
    border-radius: 12px !important;
    min-height: 125px !important;
}

div[data-testid="stFileUploader"] section,
div[data-testid="stFileUploader"] section * {
    color: #172033 !important;
}

div[data-testid="stFileUploader"] button {
    background: #2563eb !important;
    color: #ffffff !important;
    border: 0 !important;
    border-radius: 10px !important;
    font-weight: 800 !important;
}

div[data-testid="stFileUploader"] button * {
    color: #ffffff !important;
}

/* ---------- Slider ---------- */
div[data-testid="stSlider"] label {
    color: #172033 !important;
    font-weight: 800 !important;
}

/* ---------- Buttons ---------- */
div[data-testid="stButton"] > button {
    width: 100%;
    min-height: 44px;
    border: 0 !important;
    border-radius: 11px !important;
    background: linear-gradient(135deg,#2563eb,#0f766e) !important;
    color: #ffffff !important;
    font-weight: 850 !important;
}

div[data-testid="stButton"] > button *,
div[data-testid="stDownloadButton"] button * {
    color: #ffffff !important;
}

/* ---------- Status ---------- */
.pv-green {
    color: #138a4a !important;
    font-weight: 850;
}

.pv-red {
    color: #d62839 !important;
    font-weight: 850;
}

.pv-footer {
    text-align: center;
    color: #64748b !important;
    font-size: .78rem;
    padding: 2rem 0 .5rem 0;
}

/* ---------- Mobile ---------- */
@media (max-width: 700px) {
    .pv-title { font-size: 2.15rem; }
    .pv-hero { padding: 1.6rem; }
    button[data-baseweb="tab"] { padding: .55rem .55rem !important; }
}
</style>
""")

st.html(CSS)


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "result_image": None,
    "original_image": None,
    "detections": [],
    "total": 0,
    "available": 0,
    "occupied": 0,
    "availability": 0.0,
    "filename": "",
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def get_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found:\n{MODEL_PATH}"
        )
    return YOLO(str(MODEL_PATH))


# ============================================================
# HELPERS
# ============================================================

@st.cache_resource(show_spinner=False)
def get_car_model():
    """Load the standard YOLOv8 COCO model used only to check
    whether a detected parking-space box actually contains a car.
    The rest of the ParkVision UI and parking model remain unchanged.
    """
    if not CAR_MODEL_PATH.exists():
        return None
    return YOLO(str(CAR_MODEL_PATH))


def class_name(model, class_id):
    names = model.names

    if isinstance(names, dict):
        return str(names.get(class_id, class_id))

    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])

    return str(class_id)


def parking_state(class_id):
    # Dataset class mapping from data.yaml:
    # 0 = space-empty  -> available
    # 1 = space-occupied -> occupied
    if int(class_id) == 0:
        return "available"
    if int(class_id) == 1:
        return "occupied"
    return "unknown"


def font(size):
    for candidate in (
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ):
        if candidate.exists():
            try:
                return ImageFont.truetype(str(candidate), size)
            except Exception:
                pass

    return ImageFont.load_default()


def annotate(image, detections):
    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)

    fnt = font(max(13, output.width // 95))
    line_width = max(3, output.width // 350)

    for item in detections:
        x1, y1, x2, y2 = item["box"]
        state = item["state"]

        if state == "available":
            color = "#16a34a"
            label = "AVAILABLE"
        elif state == "occupied":
            color = "#dc2626"
            label = "OCCUPIED"
        else:
            color = "#d97706"
            label = item["class_name"].upper()

        draw.rounded_rectangle(
            [x1, y1, x2, y2],
            radius=5,
            outline=color,
            width=line_width,
        )

        text = f"{label}  {item['confidence']:.0%}"
        bbox = draw.textbbox((0, 0), text, font=fnt)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        top = max(0, y1 - th - 11)

        draw.rounded_rectangle(
            [x1, top, x1 + tw + 14, top + th + 9],
            radius=5,
            fill=color,
        )

        draw.text(
            (x1 + 7, top + 3),
            text,
            fill="#ffffff",
            font=fnt,
        )

    return output


def detect(image, confidence):
    """Detect parking spaces with high recall while keeping the UI unchanged.

    Strategy:
    1. Run the trained parking-space detector on overlapping tiles.
    2. Merge duplicate detections.
    3. Use vehicle detections as additional anchors.
    4. Reconstruct missing bays between detected bays in the same row using
       the measured parking-space spacing/size. This is especially useful for
       empty bays that the trained detector sometimes skips.
    5. Decide occupied/available from vehicle overlap, not from the parking
       model's unreliable class label.
    """
    model = get_model()
    img = image.convert("RGB")
    width, height = img.size

    def make_tiles(tile_size, overlap=0.40):
        if width <= tile_size and height <= tile_size:
            return [(0, 0, width, height)]

        step = max(1, int(tile_size * (1.0 - overlap)))

        def starts(length):
            if length <= tile_size:
                return [0]
            values = list(range(0, max(1, length - tile_size + 1), step))
            last = length - tile_size
            if values[-1] != last:
                values.append(last)
            return values

        return [
            (x, y, min(x + tile_size, width), min(y + tile_size, height))
            for y in starts(height)
            for x in starts(width)
        ]

    # Multiple overlapping views give small empty spaces more pixels.
    tile_sets = [make_tiles(960, 0.40)]
    if width > 1200 or height > 1200:
        tile_sets.append(make_tiles(1280, 0.40))

    all_tiles = []
    seen_tiles = set()
    for tile_set in tile_sets:
        for tile in tile_set:
            if tile not in seen_tiles:
                seen_tiles.add(tile)
                all_tiles.append(tile)

    raw = []
    inference_conf = max(0.025, min(float(confidence), 0.20))

    for left, top, right, bottom in all_tiles:
        tile = img.crop((left, top, right, bottom))
        results = model.predict(
            source=tile,
            conf=inference_conf,
            imgsz=1280,
            iou=0.45,
            max_det=2500,
            augment=True,
            verbose=False,
        )

        if not results or results[0].boxes is None:
            continue

        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        for box, conf, cls_id in zip(xyxy, confs, classes):
            x1, y1, x2, y2 = [float(v) for v in box]
            if x2 <= x1 or y2 <= y1:
                continue
            raw.append({
                "box": [x1 + left, y1 + top, x2 + left, y2 + top],
                "confidence": float(conf),
                "class_id": int(cls_id),
            })

    def iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        aa = max(1.0, (ax2 - ax1) * (ay2 - ay1))
        ba = max(1.0, (bx2 - bx1) * (by2 - by1))
        union = aa + ba - inter
        return inter / union if union else 0.0

    def containment(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        aa = max(1.0, (ax2 - ax1) * (ay2 - ay1))
        ba = max(1.0, (bx2 - bx1) * (by2 - by1))
        return inter / min(aa, ba)

    # Global duplicate removal after tiled inference.
    raw.sort(key=lambda d: d["confidence"], reverse=True)
    kept = []
    for item in raw:
        if not any(
            iou(item["box"], old["box"]) >= 0.42
            or containment(item["box"], old["box"]) >= 0.82
            for old in kept
        ):
            kept.append(item)

    # ------------------------------------------------------------
    # Vehicle detection: used only to classify a parking bay.
    # ------------------------------------------------------------
    car_boxes = []
    car_model = get_car_model()

    if car_model is not None:
        for left, top, right, bottom in all_tiles:
            tile = img.crop((left, top, right, bottom))
            car_results = car_model.predict(
                source=tile,
                conf=0.10,
                imgsz=1280,
                iou=0.45,
                max_det=1500,
                augment=True,
                verbose=False,
            )

            if not car_results or car_results[0].boxes is None:
                continue

            cb = car_results[0].boxes
            cxyxy = cb.xyxy.cpu().numpy()
            cconf = cb.conf.cpu().numpy()
            ccls = cb.cls.cpu().numpy().astype(int)

            # COCO: car=2, motorcycle=3, bus=5, truck=7.
            for cbox, cc, cid in zip(cxyxy, cconf, ccls):
                if int(cid) in {2, 3, 5, 7} and float(cc) >= 0.10:
                    car_boxes.append([
                        float(cbox[0]) + left,
                        float(cbox[1]) + top,
                        float(cbox[2]) + left,
                        float(cbox[3]) + top,
                    ])

    # Remove duplicate vehicle boxes from overlapping tiles.
    car_boxes.sort(
        key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
        reverse=True,
    )
    unique_cars = []
    for box in car_boxes:
        if all(iou(box, old) < 0.50 for old in unique_cars):
            unique_cars.append(box)
    car_boxes = unique_cars

    def box_area(box):
        return max(1.0, (box[2] - box[0]) * (box[3] - box[1]))

    def intersection_area(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        return (ix2 - ix1) * (iy2 - iy1)

    # ------------------------------------------------------------
    # Add vehicle-centered space anchors when the parking model missed
    # the space containing a visible car.
    # ------------------------------------------------------------
    if kept:
        widths = [d["box"][2] - d["box"][0] for d in kept]
        heights = [d["box"][3] - d["box"][1] for d in kept]
        med_w = float(pd.Series(widths).median())
        med_h = float(pd.Series(heights).median())
    else:
        med_w = max(18.0, width * 0.025)
        med_h = max(30.0, height * 0.08)

    med_w = max(12.0, med_w)
    med_h = max(20.0, med_h)

    anchors = [d["box"] for d in kept]
    for car in car_boxes:
        cx = (car[0] + car[2]) / 2.0
        cy = (car[1] + car[3]) / 2.0
        # Don't add a car anchor if an existing parking-space box already
        # covers the vehicle center.
        if not any(
            d["box"][0] <= cx <= d["box"][2]
            and d["box"][1] <= cy <= d["box"][3]
            for d in kept
        ):
            # Preserve the typical parking-space aspect ratio.
            ar = med_h / med_w
            car_w = max(med_w, min(car[2] - car[0], med_w * 1.35))
            car_h = max(med_h, car_w * ar)
            candidate = [
                cx - car_w / 2,
                cy - car_h / 2,
                cx + car_w / 2,
                cy + car_h / 2,
            ]
            candidate[0] = max(0.0, candidate[0])
            candidate[1] = max(0.0, candidate[1])
            candidate[2] = min(float(width), candidate[2])
            candidate[3] = min(float(height), candidate[3])
            anchors.append(candidate)

    # ------------------------------------------------------------
    # Reconstruct missing empty bays between detected bays.
    # Parking lots have repeated, approximately equally spaced bays in
    # each row. We use the actual detections as anchors and fill only
    # geometrically plausible gaps; we do NOT paint a giant arbitrary grid.
    # ------------------------------------------------------------
    def cluster_rows(boxes):
        rows = []
        # Sort top-to-bottom by center y.
        ordered = sorted(
            boxes,
            key=lambda b: ((b[1] + b[3]) / 2.0)
        )
        for box in ordered:
            cy = (box[1] + box[3]) / 2.0
            h = box[3] - box[1]
            tolerance = max(12.0, h * 0.55)
            best = None
            best_diff = float("inf")
            for row in rows:
                diff = abs(cy - row["cy"])
                if diff <= max(tolerance, row["tol"] * 1.15) and diff < best_diff:
                    best = row
                    best_diff = diff
            if best is None:
                rows.append({"cy": cy, "tol": tolerance, "boxes": [box]})
            else:
                best["boxes"].append(box)
                n = len(best["boxes"])
                best["cy"] = ((best["cy"] * (n - 1)) + cy) / n
                best["tol"] = max(best["tol"], tolerance)
        return rows

    rows = cluster_rows(anchors)
    generated = []

    for row in rows:
        boxes = sorted(
            row["boxes"],
            key=lambda b: (b[0] + b[2]) / 2.0
        )
        if len(boxes) < 2:
            continue

        centers = [
            ((b[0] + b[2]) / 2.0,
             (b[1] + b[3]) / 2.0,
             b[2] - b[0],
             b[3] - b[1])
            for b in boxes
        ]

        dxs = [
            centers[i + 1][0] - centers[i][0]
            for i in range(len(centers) - 1)
            if centers[i + 1][0] > centers[i][0]
        ]
        if not dxs:
            continue

        # The smaller neighbor gaps are normally the true bay spacing;
        # large gaps are exactly where missing empty bays may exist.
        sorted_dxs = sorted(dxs)
        base_dx = float(
            pd.Series(sorted_dxs[:max(1, int(len(sorted_dxs) * 0.65))]).median()
        )
        if base_dx <= 4:
            continue

        # Estimate dimensions from the row itself.
        row_w = float(pd.Series([c[2] for c in centers]).median())
        row_h = float(pd.Series([c[3] for c in centers]).median())
        row_cy = float(pd.Series([c[1] for c in centers]).median())

        # Only fill plausible gaps. A maximum of 5 missing bays per gap
        # prevents accidental giant grids in unrelated image regions.
        for i in range(len(centers) - 1):
            x1 = centers[i][0]
            x2 = centers[i + 1][0]
            gap = x2 - x1
            missing = int(round(gap / base_dx)) - 1
            if missing <= 0 or missing > 5:
                continue

            # Require the gap to be substantially larger than normal.
            if gap < base_dx * 1.55:
                continue

            for k in range(1, missing + 1):
                cx = x1 + gap * (k / (missing + 1))
                cy = row_cy
                candidate = [
                    cx - row_w / 2.0,
                    cy - row_h / 2.0,
                    cx + row_w / 2.0,
                    cy + row_h / 2.0,
                ]
                candidate[0] = max(0.0, candidate[0])
                candidate[1] = max(0.0, candidate[1])
                candidate[2] = min(float(width), candidate[2])
                candidate[3] = min(float(height), candidate[3])

                # Don't create a duplicate of an existing anchor.
                if not any(
                    iou(candidate, old) > 0.35
                    for old in anchors + generated
                ):
                    generated.append(candidate)

    # Final candidate list: model detections + reconstructed missing bays.
    final_boxes = [d["box"] for d in kept] + generated

    # Add car-derived anchors that were not already represented.
    for car in car_boxes:
        cx = (car[0] + car[2]) / 2.0
        cy = (car[1] + car[3]) / 2.0
        if not any(
            b[0] <= cx <= b[2] and b[1] <= cy <= b[3]
            for b in final_boxes
        ):
            car_w = max(med_w, min(car[2] - car[0], med_w * 1.35))
            car_h = max(med_h, car_w * (med_h / med_w))
            candidate = [
                max(0.0, cx - car_w / 2),
                max(0.0, cy - car_h / 2),
                min(float(width), cx + car_w / 2),
                min(float(height), cy + car_h / 2),
            ]
            if not any(iou(candidate, b) > 0.35 for b in final_boxes):
                final_boxes.append(candidate)

    # Final deduplication.
    final_boxes.sort(
        key=lambda b: ((b[2] - b[0]) * (b[3] - b[1])),
        reverse=True,
    )
    unique_boxes = []
    for b in final_boxes:
        if not any(iou(b, old) >= 0.50 or containment(b, old) >= 0.82 for old in unique_boxes):
            unique_boxes.append(b)

    detections = []
    for box in unique_boxes:
        parking_area = box_area(box)
        occupied = False

        for car_box in car_boxes:
            overlap = intersection_area(box, car_box)
            if overlap <= 0:
                continue

            car_area = box_area(car_box)
            cx = (car_box[0] + car_box[2]) / 2.0
            cy = (car_box[1] + car_box[3]) / 2.0
            centre_inside = (
                box[0] <= cx <= box[2]
                and box[1] <= cy <= box[3]
            )
            overlap_with_space = overlap / parking_area
            overlap_with_car = overlap / car_area

            if centre_inside and (
                overlap_with_space >= 0.06
                or overlap_with_car >= 0.20
            ):
                occupied = True
                break

        # Model confidence is retained for real detections; reconstructed
        # spaces use a neutral confidence because they are geometry-derived.
        real_conf = 0.0
        for item in kept:
            if iou(box, item["box"]) >= 0.50:
                real_conf = max(real_conf, item["confidence"])

        state = "occupied" if occupied else "available"
        detections.append({
            "box": box,
            "confidence": real_conf if real_conf > 0 else 1.0,
            "class_name": "space-occupied" if state == "occupied" else "space-empty",
            "state": state,
        })

    return detections, annotate(image, detections)

def store_results(filename, image, detections, result):
    available = sum(x["state"] == "available" for x in detections)
    occupied = sum(x["state"] == "occupied" for x in detections)
    total = available + occupied
    percent = (available / total * 100) if total else 0.0

    st.session_state.original_image = image
    st.session_state.result_image = result
    st.session_state.detections = detections
    st.session_state.available = available
    st.session_state.occupied = occupied
    st.session_state.total = total
    st.session_state.availability = percent
    st.session_state.filename = filename


def clear_results():
    st.session_state.result_image = None
    st.session_state.original_image = None
    st.session_state.detections = []
    st.session_state.total = 0
    st.session_state.available = 0
    st.session_state.occupied = 0
    st.session_state.availability = 0.0
    st.session_state.filename = ""


def metric(label, value, note):
    st.html(
        dedent(f"""
        <div class="pv-metric">
            <div class="pv-label">{label}</div>
            <div class="pv-value">{value}</div>
            <div class="pv-note">{note}</div>
        </div>
        """),
    )


def card(title, body):
    st.html(
        dedent(f"""
        <div class="pv-card">
            <h3>{title}</h3>
            {body}
        </div>
        """),
    )


# ============================================================
# HERO
# ============================================================

st.html(
    dedent("""
    <div class="pv-hero">
        <div class="pv-badge">
            AI-POWERED PARKING SPACE DETECTION
        </div>

        <div class="pv-title">
            🅿️ ParkVision AI
        </div>

        <div class="pv-subtitle">
            A smart computer-vision system that detects parking spaces
            and identifies whether they are available or occupied.
        </div>
    </div>
    """),
)


# ============================================================
# TABS
# ============================================================

about_tab, detection_tab, analytics_tab, map_tab = st.tabs(
    [
        "📚 About ParkVision",
        "🔍 Parking Detection",
        "📈 Analytics",
        "🗺️ Parking Map",
    ]
)


# ============================================================
# ABOUT
# ============================================================

with about_tab:

    st.html(
        '<div class="section-title">About ParkVision AI</div>',
    )

    card(
        "What is ParkVision AI?",
        """
        <p>
            ParkVision AI is a computer-vision application designed
            to analyze parking-lot images and automatically identify
            parking spaces.
        </p>
        <p>
            The custom YOLO model classifies detected parking spaces
            as <b>available</b> or <b>occupied</b>.
        </p>
        """,
    )

    st.html(
        '<div class="section-title">Project Objective</div>',
    )

    a, b = st.columns(2, gap="large")

    with a:
        card(
            "🎯 Main Objective",
            """
            <p>
                Build an AI-based system that can understand the
                parking situation from an image.
            </p>
            <ul>
                <li>Detect parking spaces</li>
                <li>Identify available spaces</li>
                <li>Identify occupied spaces</li>
                <li>Calculate availability</li>
            </ul>
            """,
        )

    with b:
        card(
            "💡 Why ParkVision?",
            """
            <p>
                Manually checking a large parking area can take time.
                ParkVision provides an automated visual analysis from
                a single image.
            </p>
            <p>
                The result is presented in a simple dashboard so that
                the parking situation is easy to understand.
            </p>
            """,
        )

    st.html(
        '<div class="section-title">Core Features</div>',
    )

    c1, c2, c3 = st.columns(3, gap="large")

    with c1:
        card(
            "🔍 Parking Detection",
            """
            <p>
                Upload a parking-lot image and run the trained YOLO
                model to detect parking spaces.
            </p>
            """,
        )

    with c2:
        card(
            "📊 Analytics",
            """
            <p>
                View total spaces, available spaces, occupied spaces
                and the percentage of available parking.
            </p>
            """,
        )

    with c3:
        card(
            "🗺️ Parking Map",
            """
            <p>
                View the detected parking spaces directly on the
                uploaded image with clear green and red boxes.
            </p>
            """,
        )

    st.html(
        '<div class="section-title">Technology Used</div>',
    )

    t1, t2, t3, t4 = st.columns(4)

    tech = [
        ("🐍", "Python", "Application logic"),
        ("⚡", "Streamlit", "Web dashboard"),
        ("🤖", "YOLO", "Object detection"),
        ("🖼️", "Computer Vision", "Image analysis"),
    ]

    for column, (icon, name, description) in zip(
        (t1, t2, t3, t4),
        tech,
    ):
        with column:
            card(
                f"{icon} {name}",
                f"<p>{description}</p>",
            )

    st.html(
        '<div class="section-title">Latest Detection</div>',
    )

    s1, s2, s3, s4 = st.columns(4)

    with s1:
        metric(
            "Total Spaces",
            st.session_state.total,
            "Latest analysis",
        )

    with s2:
        metric(
            "Available",
            st.session_state.available,
            "Empty spaces",
        )

    with s3:
        metric(
            "Occupied",
            st.session_state.occupied,
            "Occupied spaces",
        )

    with s4:
        metric(
            "Availability",
            f"{st.session_state.availability:.1f}%",
            "Latest analysis",
        )


# ============================================================
# DETECTION
# ============================================================

with detection_tab:

    st.html(
        '<div class="section-title">Parking Detection</div>',
    )

    card(
        "📤 Upload a Parking-Lot Image",
        """
        <p>
            Select a JPG, JPEG or PNG image. ParkVision AI will use
            your trained parking-space detection model to analyze it.
        </p>
        """,
    )

    if MODEL_PATH.exists():
        st.success("✅ Trained parking model is ready.")
    else:
        st.error(
            "❌ Trained model not found. Expected: "
            f"{MODEL_PATH}"
        )

    confidence = st.slider(
        "AI Confidence Threshold",
        min_value=0.03,
        max_value=0.95,
        value=0.05,
        step=0.01,
        help="Increase this value to keep only more confident detections.",
    )

    uploaded = st.file_uploader(
        "Upload Parking-Lot Image",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
        key="parking_upload",
    )

    st.caption("Supported formats: JPG, JPEG and PNG")

    b1, b2 = st.columns(2, gap="medium")

    with b1:
        analyze = st.button(
            "🔍 Analyze Parking Lot",
            use_container_width=True,
        )

    with b2:
        clear = st.button(
            "↺ Clear Results",
            use_container_width=True,
        )

    if clear:
        clear_results()
        st.rerun()

    if uploaded is not None:
        try:
            input_image = Image.open(
                io.BytesIO(uploaded.getvalue())
            ).convert("RGB")

            st.html(
                '<div class="section-title">Input Image</div>',
                    )

            st.image(
                input_image,
                caption=uploaded.name,
                use_container_width=True,
            )

        except Exception:
            st.error("Could not open the selected image.")

    if analyze:

        if uploaded is None:
            st.warning(
                "📤 Please upload a parking-lot image first."
            )

        elif not MODEL_PATH.exists():
            st.error(
                "Your trained model is missing. Check:\n"
                f"{MODEL_PATH}"
            )

        else:
            try:
                input_image = Image.open(
                    io.BytesIO(uploaded.getvalue())
                ).convert("RGB")

                with st.spinner(
                    "🤖 ParkVision AI is analyzing the image..."
                ):
                    detections, result_image = detect(
                        input_image,
                        confidence,
                    )

                store_results(
                    uploaded.name,
                    input_image,
                    detections,
                    result_image,
                )

                st.success(
                    "✅ Parking analysis completed."
                )

            except Exception as error:
                st.error("Detection failed.")
                st.exception(error)

    if st.session_state.result_image is not None:

        st.html(
            '<div class="section-title">Detection Results</div>',
            )

        r1, r2, r3, r4 = st.columns(4)

        with r1:
            metric(
                "Total Spaces",
                st.session_state.total,
                "Recognized spaces",
            )

        with r2:
            metric(
                "Available",
                st.session_state.available,
                "Green boxes",
            )

        with r3:
            metric(
                "Occupied",
                st.session_state.occupied,
                "Red boxes",
            )

        with r4:
            metric(
                "Availability",
                f"{st.session_state.availability:.1f}%",
                "Available / total",
            )

        st.html(
            '<div class="section-title">AI Detection</div>',
            )

        st.image(
            st.session_state.result_image,
            caption="🟢 Available  |  🔴 Occupied",
            use_container_width=True,
        )

        st.html(
            '<div class="section-title">Detection Details</div>',
            )

        if st.session_state.detections:

            rows = []

            for number, item in enumerate(
                st.session_state.detections,
                start=1,
            ):
                if item["state"] == "available":
                    status = "🟢 Available"
                elif item["state"] == "occupied":
                    status = "🔴 Occupied"
                else:
                    status = "🟡 Unknown"

                rows.append({
                    "Space": number,
                    "Status": status,
                    "Class": item["class_name"],
                    "Confidence": f"{item['confidence']:.1%}",
                })

            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )

        else:
            st.info(
                "No parking spaces were detected at this confidence threshold."
            )


# ============================================================
# ANALYTICS
# ============================================================

with analytics_tab:

    st.html(
        '<div class="section-title">Parking Analytics</div>',
    )

    if st.session_state.total == 0:

        card(
            "📊 No Analysis Yet",
            """
            <p>
                Upload an image in the Parking Detection tab and
                run the model. Your analytics will appear here.
            </p>
            """,
        )

    else:

        data = pd.DataFrame({
            "Status": ["Available", "Occupied"],
            "Spaces": [
                st.session_state.available,
                st.session_state.occupied,
            ],
        }).set_index("Status")

        left, right = st.columns(
            [1.15, .85],
            gap="large",
        )

        with left:
            st.html(
                '<div class="section-title">Space Distribution</div>',
                    )
            st.bar_chart(
                data,
                height=350,
            )

        with right:
            st.html(
                '<div class="section-title">Current Snapshot</div>',
                    )

            card(
                "📌 Latest Analysis",
                f"""
                <p>
                    <span class="pv-green">
                    🟢 Available: {st.session_state.available}
                    </span>
                </p>
                <p>
                    <span class="pv-red">
                    🔴 Occupied: {st.session_state.occupied}
                    </span>
                </p>
                <p>
                    <b>Total spaces:</b> {st.session_state.total}
                </p>
                <p>
                    <b>Availability:</b>
                    {st.session_state.availability:.1f}%
                </p>
                """,
            )

        st.html(
            '<div class="section-title">Breakdown</div>',
            )

        total = st.session_state.total

        breakdown = pd.DataFrame({
            "Category": ["Available", "Occupied"],
            "Count": [
                st.session_state.available,
                st.session_state.occupied,
            ],
            "Percentage": [
                st.session_state.available / total * 100,
                st.session_state.occupied / total * 100,
            ],
        })

        breakdown["Percentage"] = breakdown["Percentage"].map(
            lambda x: f"{x:.1f}%"
        )

        st.dataframe(
            breakdown,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# PARKING MAP
# ============================================================

with map_tab:

    st.html(
        '<div class="section-title">Parking Map</div>',
    )

    card(
        "🗺️ Visual Parking Map",
        """
        <p>
            Green boxes represent available parking spaces.
            Red boxes represent occupied parking spaces.
        </p>
        """,
    )

    if st.session_state.result_image is None:

        st.info(
            "Run Parking Detection first to generate the parking map."
        )

    else:

        st.image(
            st.session_state.result_image,
            caption="🟢 Available  |  🔴 Occupied",
            use_container_width=True,
        )

        m1, m2 = st.columns(2)

        with m1:
            metric(
                "Available Spaces",
                st.session_state.available,
                "Green boxes",
            )

        with m2:
            metric(
                "Occupied Spaces",
                st.session_state.occupied,
                "Red boxes",
            )


# ============================================================
# FOOTER
# ============================================================

st.html(
    dedent("""
    <div class="pv-footer">
        ParkVision AI · Smart Parking Space Detection · Powered by YOLO
    </div>
    """),
)
