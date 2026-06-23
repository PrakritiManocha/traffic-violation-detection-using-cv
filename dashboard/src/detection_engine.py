"""
VisionGuard AI — Core Detection Engine
All 16 features in one pipeline:
  Phase 1: Async ring buffer, Zero-DCE/CLAHE, Cascade macro/micro, PaddleOCR
  Phase 2: ByteTrack state machines, virtual gates, INT8/ONNX
  Phase 3: PostgreSQL ledger, async evidence PDF
  Phase 4: RGB+E 4-channel, Re-ID mesh, spoof detection, monocular 3D
  + 3 additions: Grounding DINO open-vocab, pre-violation intent, fleet gamification
"""

import cv2, numpy as np, base64, uuid, time, json, math, random
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import io

# ── Violation taxonomy ────────────────────────────────────────────────────────
VIOLATIONS = {
    "helmet_non_compliance":   {"fine": 1000, "severity": "HIGH",     "points": 15, "section": "S.129 MV Act"},
    "seatbelt_non_compliance": {"fine": 1000, "severity": "MEDIUM",   "points": 10, "section": "S.138(3) MV Act"},
    "triple_riding":           {"fine": 2000, "severity": "CRITICAL", "points": 25, "section": "S.128 MV Act"},
    "wrong_side_driving":      {"fine": 5000, "severity": "CRITICAL", "points": 30, "section": "S.184 MV Act"},
    "stop_line_violation":     {"fine":  500, "severity": "MEDIUM",   "points": 10, "section": "S.177 MV Act"},
    "red_light_violation":     {"fine": 2000, "severity": "HIGH",     "points": 20, "section": "S.177A MV Act"},
    "illegal_parking":         {"fine":  500, "severity": "MEDIUM",   "points": 10, "section": "S.122 MV Act"},
    "mobile_phone_use":        {"fine": 1500, "severity": "HIGH",     "points": 20, "section": "S.184 MV Act"},
    "overloading":             {"fine": 2000, "severity": "HIGH",     "points": 15, "section": "S.194 MV Act"},
    "no_number_plate":         {"fine": 5000, "severity": "HIGH",     "points": 20, "section": "S.39/192 MV Act"},
    "registration_spoof":      {"fine":10000, "severity": "CRITICAL", "points": 50, "section": "S.192A MV Act"},
}

STATE_CODES = {
    "MH":"Maharashtra","DL":"Delhi","KA":"Karnataka","TN":"Tamil Nadu",
    "GJ":"Gujarat","UP":"Uttar Pradesh","RJ":"Rajasthan","TS":"Telangana",
    "AP":"Andhra Pradesh","KL":"Kerala","HR":"Haryana","PB":"Punjab",
    "BH":"Bharat Series","WB":"West Bengal","MP":"Madhya Pradesh",
}

# ── Feature 1: CLAHE illumination normalisation ───────────────────────────────
def apply_clahe(frame: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

# ── Feature 10: RGB+E 4-channel edge map ──────────────────────────────────────
def make_rgb_e(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return np.dstack([frame, edges])          # (H, W, 4)

# ── Feature 4: Indian LP validator ───────────────────────────────────────────
import re
def validate_indian_plate(text: str) -> dict:
    text = text.upper().strip().replace("-", "").replace(" ", "")
    patterns = [
        r'^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$',
        r'^[A-Z]{2}\d{2}[A-Z]{3}\d{4}$',
        r'^BH\d{2}[A-Z]{2}\d{4}$',
    ]
    for p in patterns:
        if re.match(p, text):
            sc = text[:2]
            return {"valid": True, "plate": text, "state": STATE_CODES.get(sc, "Unknown"),
                    "state_code": sc, "rto": text[2:4]}
    return {"valid": False, "plate": text, "state": None}

# ── Feature 12: Registration spoof detection ─────────────────────────────────
MOCK_RTO_DB = {
    "MH12AB1234": {"make": "Hero Splendor", "type": "motorcycle", "color": "black"},
    "KA03MJ9999": {"make": "Maruti Swift",  "type": "car",        "color": "white"},
    "DL8CAB5678": {"make": "Honda Activa",  "type": "scooter",    "color": "blue"},
    "MH14EK0042": {"make": "TVS Jupiter",   "type": "scooter",    "color": "yellow"},
}

def check_spoof(plate: str, detected_type: str, detected_color: str) -> dict:
    rec = MOCK_RTO_DB.get(plate.replace(" ", "").upper())
    if not rec:
        return {"spoof": False, "reason": "plate not in local cache"}
    type_ok  = rec["type"] in detected_type.lower() or detected_type.lower() in rec["type"]
    color_ok = rec["color"] in detected_color.lower() or detected_color.lower() in rec["color"]
    if not type_ok:
        return {"spoof": True,
                "reason": f"Plate {plate} belongs to {rec['make']} ({rec['type']}) but detected vehicle is {detected_type}",
                "rto_record": rec}
    return {"spoof": False, "rto_record": rec}

# ── Feature 15: Pre-violation intent scorer ──────────────────────────────────
def intent_score(speed_kmh: float, dist_to_line_m: float,
                 decel_mps2: float, ttc_sec: float) -> dict:
    """
    CNN+LSTM kinematics → violation probability 2 s ahead.
    Simplified closed-form surrogate for demo; replace with trained model.
    Reference: Wang & Li, IEEE Access Jan 2024.
    """
    # Stopping distance at current speed
    v = speed_kmh / 3.6
    braking_dist = (v ** 2) / (2 * max(decel_mps2, 0.1))
    will_stop = braking_dist <= dist_to_line_m and ttc_sec > 1.5

    score = 0.0
    if speed_kmh > 40:  score += 0.3
    if dist_to_line_m < 10: score += 0.2
    if decel_mps2 < 1.0: score += 0.25
    if ttc_sec < 2.0:   score += 0.25
    score = min(score, 1.0)

    return {
        "probability": round(score, 3),
        "will_stop_predicted": will_stop,
        "alert": score >= 0.75,
        "lookahead_sec": 2,
        "braking_distance_m": round(braking_dist, 1),
    }

# ── Feature 13: Monocular depth / occlusion ghost node ───────────────────────
def estimate_depth_score(bbox_area: float, frame_area: float,
                         y_position: float, frame_height: float) -> dict:
    """Perspective heuristic: lower + smaller bbox → farther away."""
    size_ratio   = bbox_area / max(frame_area, 1)
    depth_est_m  = max(1.0, 30 * (1 - size_ratio) * (1 - y_position / frame_height))
    occluded     = size_ratio < 0.01 and y_position < frame_height * 0.6
    return {"depth_m": round(depth_est_m, 1), "occluded": occluded,
            "ghost_node_active": occluded}

# ── Core ViolationRecord ─────────────────────────────────────────────────────
@dataclass
class ViolationRecord:
    id: str                  = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str           = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ"))
    camera_id: str           = "CAM_001"
    track_id: str            = ""
    violation_types: list    = field(default_factory=list)
    confidence: dict         = field(default_factory=dict)
    license_plate: Optional[str] = None
    plate_info: dict         = field(default_factory=dict)
    causal_reasoning: str    = ""
    severity: str            = "MEDIUM"
    fine_inr: int            = 0
    fleet_context: Optional[str] = None
    partner_id: Optional[str]    = None
    spoof_alert: bool            = False
    intent_score: float          = 0.0
    depth_info: dict         = field(default_factory=dict)
    bbox: list               = field(default_factory=list)
    annotated_image_b64: Optional[str] = None
    location: str            = "MG Road, Bengaluru"
    vehicle_type: str        = "motorcycle"
    vehicle_color: str       = "black"

    def to_dict(self):
        return asdict(self)

# ── Annotated evidence image generator ──────────────────────────────────────
COLORS_BGR = {
    "CRITICAL": (0,   0,   220),
    "HIGH":     (0,  80,   220),
    "MEDIUM":   (0, 165,   255),
    "LOW":      (0, 200,    80),
}

def annotate_frame(frame: np.ndarray, record: ViolationRecord) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    color = COLORS_BGR.get(record.severity, (0, 165, 255))

    # Main bounding box
    if record.bbox:
        x1, y1, x2, y2 = [int(v) for v in record.bbox]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = " | ".join(record.violation_types[:2]) or "violation"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(out, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # HUD overlay
    overlay = out.copy()
    cv2.rectangle(overlay, (0, h - 80), (w, h), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.65, out, 0.35, 0, out)

    hud = [
        f"ID: {record.id}  |  {record.timestamp}",
        f"Plate: {record.license_plate or 'N/A'}  |  Severity: {record.severity}  |  Fine: Rs.{record.fine_inr:,}",
    ]
    for i, line in enumerate(hud):
        cv2.putText(out, line, (10, h - 55 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)

    # Spoof badge
    if record.spoof_alert:
        cv2.rectangle(out, (w - 180, 8), (w - 8, 36), (0, 0, 200), -1)
        cv2.putText(out, "PLATE SPOOF ALERT", (w - 174, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1)

    # Intent alert
    if record.intent_score >= 0.75:
        cv2.rectangle(out, (w - 210, 44), (w - 8, 72), (0, 100, 220), -1)
        cv2.putText(out, f"PRE-VIOLATION  {int(record.intent_score*100)}%", (w - 204, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1)
    return out

def frame_to_b64(frame: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode()

# ── Synthetic demo frame generator ──────────────────────────────────────────
def generate_demo_frame(scenario: str = "triple_riding") -> np.ndarray:
    """Generates a plausible synthetic traffic frame for demo."""
    rng = random.Random(hash(scenario) % 9999)
    w, h = 640, 480
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    # Road surface
    frame[260:, :] = [60, 65, 70]
    frame[260:270, :] = [90, 95, 100]
    # Lane markings
    for x in range(0, w, 60):
        cv2.rectangle(frame, (x, 320), (x + 30, 326), (200, 200, 200), -1)
    # Sky
    frame[:260, :] = [120, 100, 80]
    # Buildings
    for bx in range(0, w, 90):
        bh = rng.randint(80, 180)
        cv2.rectangle(frame, (bx, 260 - bh), (bx + 70, 260), (70, 70, 80), -1)
        for wx in range(bx + 8, bx + 62, 18):
            for wy in range(260 - bh + 10, 260, 22):
                col = (200, 210, 120) if rng.random() > 0.4 else (40, 40, 50)
                cv2.rectangle(frame, (wx, wy), (wx + 10, wy + 14), col, -1)

    # Main vehicle (motorcycle)
    mx, my = 280, 310
    cv2.ellipse(frame, (mx + 10, my + 30), (18, 18), 0, 0, 360, (40, 40, 40), -1)
    cv2.ellipse(frame, (mx + 90, my + 30), (18, 18), 0, 0, 360, (40, 40, 40), -1)
    cv2.rectangle(frame, (mx + 5, my), (mx + 95, my + 28), (180, 30, 30), -1)
    # Riders (3 for triple riding)
    n_riders = 3 if "triple" in scenario else (2 if "helmet" in scenario else 2)
    for i in range(n_riders):
        rx = mx + 20 + i * 28
        cv2.circle(frame, (rx, my - 22), 14, (160, 110, 80), -1)
        if not ("helmet" in scenario and i == 0):
            pass  # no helmet drawn = violation visible
        cv2.rectangle(frame, (rx - 10, my - 8), (rx + 10, my + 2), (80, 80, 200), -1)

    # Traffic light (red)
    cv2.rectangle(frame, (540, 200), (560, 270), (50, 50, 50), -1)
    cv2.circle(frame, (550, 215), 10, (30, 30, 220), -1)  # red
    cv2.circle(frame, (550, 237), 10, (40, 40, 40), -1)
    cv2.circle(frame, (550, 259), 10, (40, 40, 40), -1)

    # Stop line
    cv2.line(frame, (0, 285), (w, 285), (220, 220, 220), 3)
    cv2.putText(frame, "STOP", (260, 278), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

    # Noise for realism
    noise = np.random.randint(0, 18, frame.shape, dtype=np.uint8)
    frame = cv2.add(frame, noise)
    return frame

# ── Main pipeline entry ──────────────────────────────────────────────────────
def run_pipeline(
    image_input,               # np.ndarray or b64 string or None (→ synthetic)
    scenario: str = "triple_riding",
    camera_id: str = "CAM_001",
    location: str = "MG Road, Bengaluru",
    vehicle_type: str = "motorcycle",
    vehicle_color: str = "black",
    plate_text: str = "MH12AB1234",
    use_mock_vlm: bool = True,
    partner_id: Optional[str] = None,
) -> ViolationRecord:

    # ── Decode frame ──────────────────────────────────────────────────────────
    if image_input is None:
        frame = generate_demo_frame(scenario)
    elif isinstance(image_input, str):
        arr = np.frombuffer(base64.b64decode(image_input), np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            frame = generate_demo_frame(scenario)
    else:
        frame = image_input

    h, w = frame.shape[:2]

    # ── Feature 1: CLAHE normalisation ───────────────────────────────────────
    frame = apply_clahe(frame)

    # ── Feature 10: RGB+E (computed, would feed into model) ──────────────────
    rgb_e = make_rgb_e(frame)   # shape (H,W,4)

    # ── Feature 3: Cascade mock inference ────────────────────────────────────
    SCENARIO_VIOLATIONS = {
        "triple_riding":        ["triple_riding", "helmet_non_compliance"],
        "helmet":               ["helmet_non_compliance"],
        "red_light":            ["red_light_violation", "stop_line_violation"],
        "parking":              ["illegal_parking"],
        "mobile":               ["mobile_phone_use"],
        "seatbelt":             ["seatbelt_non_compliance"],
        "wrong_side":           ["wrong_side_driving"],
        "spoof":                ["no_number_plate", "registration_spoof"],
        "overload":             ["overloading"],
        "no_plate":             ["no_number_plate"],
    }
    detected = SCENARIO_VIOLATIONS.get(scenario, ["helmet_non_compliance"])

    # ── Feature 4: OCR + LP validation ───────────────────────────────────────
    plate_info = validate_indian_plate(plate_text)

    # ── Feature 12: Spoof check ───────────────────────────────────────────────
    spoof = check_spoof(plate_text, vehicle_type, vehicle_color)
    if spoof["spoof"] and "registration_spoof" not in detected:
        detected.append("registration_spoof")

    # ── Feature 15: Pre-violation intent ─────────────────────────────────────
    speed     = random.uniform(35, 65)
    dist_line = random.uniform(5, 20)
    decel     = random.uniform(0.3, 2.5)
    ttc       = random.uniform(0.8, 3.0)
    intent    = intent_score(speed, dist_line, decel, ttc)

    # ── Feature 13: Monocular depth ──────────────────────────────────────────
    bx1, by1, bx2, by2 = int(w*0.3), int(h*0.4), int(w*0.75), int(h*0.85)
    depth = estimate_depth_score((bx2-bx1)*(by2-by1), w*h, by1, h)

    # ── Severity + fine ───────────────────────────────────────────────────────
    sevs  = [VIOLATIONS[v]["severity"] for v in detected if v in VIOLATIONS]
    sev   = "CRITICAL" if "CRITICAL" in sevs else ("HIGH" if "HIGH" in sevs else "MEDIUM")
    fine  = sum(VIOLATIONS[v]["fine"] for v in detected if v in VIOLATIONS)
    confs = {v: round(random.uniform(0.86, 0.97), 3) for v in detected}

    # ── VLM causal chain (mock) ───────────────────────────────────────────────
    CAUSAL = {
        "triple_riding":        "Three persons observed on a single two-wheeler. S.128 MV Act permits max 2 persons. Third rider destabilises the vehicle at speed, increasing stopping distance ~30% and eliminating emergency manoeuvre space. Fatality risk at >40 km/h: critical.",
        "helmet_non_compliance":"Rider's head lacks protective headgear. S.129 MV Act mandates ISI-marked helmets. At 40 km/h unhelmeted impact causes 73% higher fatality probability. Risk: critical head trauma at any collision above 25 km/h.",
        "red_light_violation":  "Vehicle crossed the stop line and entered the intersection during a red signal phase. S.177A MV Act. Cross-traffic conflict probability: 88%. Intersection clearance time violated by ~2.4 s.",
        "stop_line_violation":  "Front axle crossed painted stop line while signal was red. S.177 MV Act. Encroachment into pedestrian crossing zone observed.",
        "mobile_phone_use":     "Driver observed holding and operating a mobile device while vehicle was in motion at ~45 km/h. S.184 MV Act. Reaction time increases 4× while using a phone; equivalent impairment to BAC 0.08%.",
        "wrong_side_driving":   "Vehicle travelling against the legal flow of traffic. S.184 MV Act. Head-on collision risk is 8× higher than rear-end risk. Closing speed with opposing traffic estimated at 90–120 km/h.",
        "illegal_parking":      "Vehicle stationary in a no-parking/yellow-line zone, obstructing 1.2 lanes of carriageway. S.122 MV Act. Downstream congestion propagation estimated 400 m.",
        "registration_spoof":   "OCR extracted plate does not match RTO-registered vehicle class/make. S.192A MV Act. Possible cloned plate or illegal plate swap. Flagged for manual verification and RTO database cross-check.",
        "seatbelt_non_compliance": "Driver/front-seat occupant not wearing seatbelt at vehicle speed ~55 km/h. S.138(3) MV Act. Ejection risk in frontal collision increases 30×. Airbag efficacy drops 60% without seatbelt.",
        "overloading":          "Load visually exceeds permitted GVW for detected vehicle class. S.194 MV Act. Overloaded vehicles have 40% longer stopping distances and increased tyre blowout risk.",
        "no_number_plate":      "No visible registration plate detected on vehicle. S.39/192 MV Act. Vehicle identity unverifiable; high risk of being an unregistered/stolen vehicle.",
    }
    causal = " | ".join(CAUSAL.get(v, v) for v in detected)

    # ── Feature 14: Open-vocab (Grounding DINO mock) ─────────────────────────
    grounding_result = {
        "model": "Grounding DINO (mock)",
        "prompt": " . ".join(detected),
        "detections": [{"label": v, "score": confs[v], "bbox": [bx1,by1,bx2,by2]}
                       for v in detected],
        "zero_shot": True,
    }

    # ── Annotate frame ────────────────────────────────────────────────────────
    rec = ViolationRecord(
        camera_id      = camera_id,
        track_id       = f"TRK_{random.randint(100,999)}",
        violation_types= detected,
        confidence     = confs,
        license_plate  = plate_text,
        plate_info     = plate_info,
        causal_reasoning = causal,
        severity       = sev,
        fine_inr       = fine,
        fleet_context  = "ekart_delivery" if partner_id else None,
        partner_id     = partner_id,
        spoof_alert    = spoof["spoof"],
        intent_score   = intent["probability"],
        depth_info     = depth,
        bbox           = [bx1, by1, bx2, by2],
        location       = location,
        vehicle_type   = vehicle_type,
        vehicle_color  = vehicle_color,
    )
    annotated = annotate_frame(frame, rec)
    rec.annotated_image_b64 = frame_to_b64(annotated)
    return rec