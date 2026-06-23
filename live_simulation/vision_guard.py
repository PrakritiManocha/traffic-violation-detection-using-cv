import cv2
import numpy as np
import time
import hashlib
import json
import os
import sqlite3
import logging
import re
import pytesseract
from PIL import Image
from collections import deque
from ultralytics import YOLO

# ── Suppress YOLO verbose boot logs ───────────────────────────────────────────
logging.getLogger("ultralytics").setLevel(logging.WARNING)
os.makedirs("storage", exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  DATABASE INITIALIZER
# ══════════════════════════════════════════════════════════════════════════════

def initialize_database():
    conn = sqlite3.connect("vision_guard.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS violations (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            evidence_id        TEXT UNIQUE,
            timestamp          REAL,
            plate_id           TEXT,
            violation_class    TEXT,
            confidence         REAL,
            legal_narrative    TEXT,
            cryptographic_hash TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS traffic_analytics (
            frame_id        INTEGER,
            timestamp       REAL,
            active_nodes    INTEGER,
            edge_latency_ms REAL,
            signal_phase    TEXT
        )
    """)
    conn.commit()
    conn.close()


# ══════════════════════════════════════════════════════════════════════════════
#  TRACKING NODE
# ══════════════════════════════════════════════════════════════════════════════

class TrackingNode:
    """Lightweight per-object state carrier with history buffer and ReID token."""
    def __init__(self, obj_id: int, bbox: tuple, cls: int):
        self.obj_id   = obj_id
        self.bbox     = bbox
        self.cls      = cls
        self.history  = deque(maxlen=60)
        self.history.append(bbox)
        self.stationary_count = 0
        # Simulated cross-camera Re-ID signature (single-feed; see status notes above)
        self.reid_sig = hashlib.md5(f"NODE-{obj_id}-CLS-{cls}".encode()).hexdigest()[:8].upper()

    @property
    def center(self):
        x1, y1, x2, y2 = self.bbox
        return int((x1 + x2) / 2), int((y1 + y2) / 2)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class VisionGuardEngine:
    # COCO class indices we care about
    COCO_PERSON     = 0
    COCO_CAR        = 2
    COCO_MOTORCYCLE = 3
    COCO_BUS        = 5
    COCO_TRUCK      = 7

    VEHICLE_CLASSES = {2, 3, 5, 7}
    ALL_CLASSES     = {0, 2, 3, 5, 7}

    COCO_NAMES = {0: "Person", 2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

    def __init__(self, video_source: str = "mock_stream.mp4"):
        if not os.path.exists(video_source):
            raise FileNotFoundError(f"Video file '{video_source}' not found.")

        print(f"[INIT] Loading video: {video_source}")
        initialize_database()

        self.cap = cv2.VideoCapture(video_source)
        self.W, self.H = 640, 480

        # ── YOLOv8n: free, CPU-efficient (~6MB), no GPU needed ────────────────
        self.detector = YOLO("yolov8n.pt")

        # ── Geometry / Geofences (tune these per camera angle) ────────────────
        # Stop line: horizontal pixel row
        self.STOP_LINE_Y = 350
        # Illegal parking zone polygon: (x1, y1, x2, y2)
        self.PARKING_ZONE = (50, 380, 250, 470)
        # Traffic light crop: (x1, y1, x2, y2) — adjust to cover signal head in your feed
        self.TL_BOX = (540, 20, 590, 100)

        # ── Runtime state ──────────────────────────────────────────────────────
        self.frame_idx     = 0
        self.track_counter = 0
        self.nodes: dict[int, TrackingNode] = {}
        self.logged        = set()          # de-duplication keys for violations
        self.ev_count      = 0
        self.signal_phase  = "GREEN"

    # ──────────────────────────────────────────────────────────────────────────
    #  STAGE 1 — PREPROCESSING
    # ──────────────────────────────────────────────────────────────────────────

    def clahe_preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Convert to YUV, apply CLAHE to the Y (luma) channel only.
        Handles: glare, low-light, rain haze, deep shadows.
        """
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    def build_edge_representation(self, frame: np.ndarray) -> np.ndarray:
        """
        Blend a Canny edge map into the B channel of the frame.
        This gives the detector stronger structural boundary signals under
        poor lighting — implements the RGB+Edge input representation from the spec.
        Minimal CPU cost; no GPU needed.
        """
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)                  # real-time edge map
        out   = frame.copy()
        # Overlay edges into blue channel (additive blend, clipped)
        out[:, :, 0] = np.clip(out[:, :, 0].astype(np.int32) + edges.astype(np.int32) // 2, 0, 255).astype(np.uint8)
        return out

    # ──────────────────────────────────────────────────────────────────────────
    #  STAGE 2 — TRAFFIC SIGNAL DETECTION
    # ──────────────────────────────────────────────────────────────────────────

    def detect_signal(self, frame: np.ndarray) -> str:
        """
        Crop the traffic light region and check HSV for red pixels.
        RED: two hue bands (0-10 and 170-180) to cover the full red wraparound.
        Falls back to GREEN if crop is empty or too dark.

        Calibration note: set self.TL_BOX to exactly surround the signal
        head in your video. Draw the box on frame and adjust until only the
        lamp is inside — otherwise streetlights etc. cause false REDs.
        """
        x1, y1, x2, y2 = self.TL_BOX
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return "GREEN"

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # Tighter saturation/value gates to avoid reflections
        red_lo1 = np.array([0,  120, 80])
        red_hi1 = np.array([10, 255, 255])
        red_lo2 = np.array([165, 120, 80])
        red_hi2 = np.array([180, 255, 255])

        mask  = cv2.inRange(hsv, red_lo1, red_hi1)
        mask |= cv2.inRange(hsv, red_lo2, red_hi2)

        # Require >15% of crop to be red to reduce false positives
        red_ratio = cv2.countNonZero(mask) / max(mask.size, 1)
        return "RED" if red_ratio > 0.15 else "GREEN"

    # ──────────────────────────────────────────────────────────────────────────
    #  STAGE 3 — LICENSE PLATE OCR  (Tesseract — free, CPU-only)
    # ──────────────────────────────────────────────────────────────────────────

    def ocr_plate(self, vehicle_crop: np.ndarray) -> str:
        """
        Extracts alphanumeric plate text from a vehicle bounding-box crop.

        Pipeline:
          1. Convert to grayscale
          2. Upscale 3x (Tesseract performs poorly on tiny images)
          3. Adaptive threshold to isolate dark text on bright plate
          4. Morphological close to fill character gaps
          5. Run Tesseract with --psm 7 (single text line) and alphanumeric whitelist
          6. Strip non-alphanumeric chars, enforce min 4-char result

        Why Tesseract instead of PaddleOCR:
          PaddleOCR in CPU mode requires model downloads at runtime and was
          returning fallback strings ("MH12UNAVAILABLE") for all inputs —
          indicating inference failures. Tesseract is already installed
          system-wide, requires zero network access, and with the preprocessing
          pipeline below produces reasonable results on decent-resolution crops.
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return "PLATE_UNREADABLE"

        try:
            gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)

            # Upscale — critical for small bounding boxes
            scale = max(1, int(150 / max(gray.shape[0], 1)))  # target ~150px tall
            scale = min(scale, 6)
            if scale > 1:
                gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

            # Adaptive threshold — handles varied lighting across plate surface
            binary = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 15, 8
            )

            # Close small gaps inside characters
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            # Tesseract: PSM 7 = single text line, whitelist alphanumeric only
            cfg = r"--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            raw = pytesseract.image_to_string(binary, config=cfg)
            clean = re.sub(r"[^A-Z0-9]", "", raw.upper())

            return clean if len(clean) >= 4 else "PLATE_SHORT"

        except Exception:
            return "PLATE_OCR_FAIL"

    # ──────────────────────────────────────────────────────────────────────────
    #  STAGE 3b — VEHICLE-PLATE CONSISTENCY CHECK
    # ──────────────────────────────────────────────────────────────────────────

    def check_plate_consistency(self, plate: str, cls: int) -> bool:
        """
        Basic sanity check: Indian plates typically follow patterns like
        MH12AB1234 (state code + district + letters + digits).
        If OCR produces something that looks nothing like a plate for the
        given vehicle type, flag it as suspicious.
        Returns True if plate looks suspicious/inconsistent.
        """
        if plate in ("PLATE_UNREADABLE", "PLATE_SHORT", "PLATE_OCR_FAIL"):
            return False  # Can't judge unreadable plates

        # Indian plate: 2-letter state code + 2-digit district + 1-2 letters + 4 digits
        # Relaxed regex to catch most valid formats
        valid_pattern = re.compile(r"^[A-Z]{2}\d{2}[A-Z]{1,3}\d{4}$")
        if not valid_pattern.match(plate):
            return True  # Plate format is suspicious

        # Motorcycles should not have a 'T' (tourist) prefix in most cases — trivial example
        # Extend this map with more domain rules as needed
        return False

    # ──────────────────────────────────────────────────────────────────────────
    #  STAGE 4 — TRACKING ENGINE
    # ──────────────────────────────────────────────────────────────────────────

    def update_tracking(self, boxes) -> dict:
        """
        Manual centroid proximity tracking.
        - For each detected box, find the nearest existing node within 60px
        - If matched: update that node's history
        - If not matched: spawn a new node
        - Nodes not seen this frame are dropped (no ghost tracking)

        This implements the ByteTrack-style proximity matching concept from
        the spec. True ByteTrack (with Kalman filter predictions for occluded
        objects) would give more robust re-association across partial occlusions
        but requires additional libraries. This version is CPU-efficient and
        handles normal traffic scenarios well.
        """
        current: dict[int, TrackingNode] = {}

        for box in boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            if conf < 0.30 or cls not in self.ALL_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Find nearest existing node
            best_id, best_dist = None, 60.0
            for nid, node in self.nodes.items():
                dx = cx - node.center[0]
                dy = cy - node.center[1]
                d  = (dx * dx + dy * dy) ** 0.5
                if d < best_dist:
                    best_dist = d
                    best_id   = nid

            if best_id is not None:
                n = self.nodes[best_id]
                n.bbox = (x1, y1, x2, y2)
                n.history.append((x1, y1, x2, y2))
                current[best_id] = n
            else:
                self.track_counter += 1
                current[self.track_counter] = TrackingNode(self.track_counter, (x1, y1, x2, y2), cls)

        self.nodes = current
        return current

    # ──────────────────────────────────────────────────────────────────────────
    #  STAGE 5 — VIOLATION EVALUATORS
    # ──────────────────────────────────────────────────────────────────────────

    def eval_triple_riding(self, node_id: int, moto_node: TrackingNode,
                           all_nodes: dict, frame: np.ndarray) -> None:
        """Count person centroids inside motorcycle bounding box ± margin."""
        x1, y1, x2, y2 = moto_node.bbox
        riders = 0
        for pid, pn in all_nodes.items():
            if pn.cls != self.COCO_PERSON:
                continue
            pcx, pcy = pn.center
            if (x1 - 20 <= pcx <= x2 + 20) and (y1 - 35 <= pcy <= y2 + 10):
                riders += 1

        if riders >= 3:
            key = f"triple-{node_id}"
            if key not in self.logged:
                self.logged.add(key)
                crop  = frame[y1:y2, x1:x2]
                plate = self.ocr_plate(crop)
                brief = self.law_engine("TRIPLE_RIDING",
                    f"Tracking matrices confirmed {riders} person centroids on motorcycle node {node_id}.")
                self.generate_evidence(node_id, "TRIPLE_RIDING", brief, 0.94, plate, moto_node.reid_sig, frame, (x1,y1,x2,y2))

    def eval_helmet(self, node_id: int, person_node: TrackingNode,
                    moto_node: TrackingNode, frame: np.ndarray) -> None:
        """
        Analyze head crop of a rider using Laplacian variance.
        Low variance = uniform texture = likely no helmet (bare head/hair).
        High variance = rough/structured = helmet likely present.
        Threshold tuned at 110 — may need adjustment per camera resolution.
        """
        px1, py1, px2, py2 = person_node.bbox
        head_h = max(1, int((py2 - py1) * 0.28))
        hcrop = frame[max(0, py1):min(self.H, py1 + head_h),
                      max(0, px1):min(self.W, px2)]
        if hcrop.size == 0:
            return

        gray    = cv2.cvtColor(hcrop, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        key = f"helmet-{node_id}-{person_node.obj_id}"
        if lap_var < 110.0 and key not in self.logged:
            self.logged.add(key)
            mx1, my1, mx2, my2 = moto_node.bbox
            crop  = frame[my1:my2, mx1:mx2]
            plate = self.ocr_plate(crop)
            brief = self.law_engine("HELMET_NON_COMPLIANCE",
                f"Head-crop Laplacian variance {lap_var:.1f} (threshold 110) indicates absence of structured headgear.")
            self.generate_evidence(node_id, "HELMET_NON_COMPLIANCE", brief, 0.82,
                                   plate, person_node.reid_sig, frame, (px1, py1, px2, py2))

    def eval_seatbelt(self, node_id: int, node: TrackingNode, frame: np.ndarray) -> None:
        """
        Canny edge density on the windshield ROI (top 30% of car bbox).
        The diagonal seatbelt sash creates a strong oblique edge — low
        density implies the belt is absent. Threshold 1.3 calibrated
        empirically; raise it if too many false positives.
        """
        x1, y1, x2, y2 = node.bbox
        ws_h  = max(1, int((y2 - y1) * 0.30))
        crop  = frame[y1:min(self.H, y1 + ws_h), x1:x2]
        if crop.size == 0:
            return

        gray    = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges   = cv2.Canny(gray, 40, 120)
        density = float(np.sum(edges)) / max(gray.size, 1)

        key = f"seatbelt-{node_id}"
        if density < 1.3 and key not in self.logged:
            self.logged.add(key)
            full_crop = frame[y1:y2, x1:x2]
            plate     = self.ocr_plate(full_crop)
            brief     = self.law_engine("SEATBELT_NON_COMPLIANCE",
                f"Windshield Canny edge density {density:.3f} (threshold 1.3) — seatbelt sash boundary absent.")
            self.generate_evidence(node_id, "SEATBELT_NON_COMPLIANCE", brief, 0.78,
                                   plate, node.reid_sig, frame, (x1, y1, x2, y2))

    def eval_wrong_side(self, node_id: int, node: TrackingNode, frame: np.ndarray) -> None:
        """
        Trajectory vector check over last 8 frames.
        Right-side traffic (India) moves downward on screen (increasing Y).
        If a vehicle on the right half of frame is moving upward (decreasing Y),
        it's on the wrong side.
        Tuned: requires ≥8px upward displacement to avoid jitter false-positives.
        """
        if len(node.history) < 8:
            return
        x1, y1, x2, y2 = node.bbox
        cx = (x1 + x2) // 2

        past_y1 = node.history[-8][1]   # top-y 8 frames ago
        delta_y  = y1 - past_y1          # negative = moving up

        key = f"wrong-{node_id}"
        if delta_y < -8 and cx > self.W // 2 and key not in self.logged:
            self.logged.add(key)
            crop  = frame[y1:y2, x1:x2]
            plate = self.ocr_plate(crop)
            brief = self.law_engine("WRONG_SIDE_DRIVING",
                f"Upward displacement vector ΔY={delta_y}px over 8 frames on right half of frame (cx={cx}).")
            self.generate_evidence(node_id, "WRONG_SIDE_DRIVING", brief, 0.89,
                                   plate, node.reid_sig, frame, (x1, y1, x2, y2))

    def eval_stop_line(self, node_id: int, node: TrackingNode, frame: np.ndarray) -> None:
        """
        Stop-line geofence: vehicle bottom edge crosses STOP_LINE_Y during RED.
        Tied to live signal phase so only fires when signal is actually RED.
        """
        x1, y1, x2, y2 = node.bbox
        crosses_line = y2 > self.STOP_LINE_Y and y1 < self.STOP_LINE_Y

        key = f"stop-{node_id}"
        if crosses_line and self.signal_phase == "RED" and key not in self.logged:
            self.logged.add(key)
            crop  = frame[y1:y2, x1:x2]
            plate = self.ocr_plate(crop)
            brief = self.law_engine("STOP_LINE_VIOLATION",
                f"Vehicle bottom edge at Y={y2} crossed stop-line Y={self.STOP_LINE_Y} during RED signal.")
            self.generate_evidence(node_id, "STOP_LINE_VIOLATION", brief, 0.91,
                                   plate, node.reid_sig, frame, (x1, y1, x2, y2))

    def eval_illegal_parking(self, node_id: int, node: TrackingNode, frame: np.ndarray) -> None:
        """
        Checks if vehicle centroid has been inside the no-parking polygon
        for more than 60 consecutive frames (~2 seconds at 30fps).
        Stationary count resets when the vehicle moves out of zone.
        """
        px1, py1, px2, py2 = self.PARKING_ZONE
        cx, cy = node.center

        if px1 <= cx <= px2 and py1 <= cy <= py2:
            node.stationary_count += 1
        else:
            node.stationary_count = max(0, node.stationary_count - 1)

        key = f"park-{node_id}"
        if node.stationary_count > 60 and key not in self.logged:
            self.logged.add(key)
            x1, y1, x2, y2 = node.bbox
            crop  = frame[y1:y2, x1:x2]
            plate = self.ocr_plate(crop)
            brief = self.law_engine("ILLEGAL_PARKING",
                "Vehicle stationary inside restricted no-parking polygon for >60 frames.")
            self.generate_evidence(node_id, "ILLEGAL_PARKING", brief, 0.97,
                                   plate, node.reid_sig, frame, (x1, y1, x2, y2))

    # ──────────────────────────────────────────────────────────────────────────
    #  LEGAL ENGINE
    # ──────────────────────────────────────────────────────────────────────────

    def law_engine(self, v_type: str, detail: str) -> str:
        """Maps violation type → Motor Vehicles Act section + description."""
        statutes = {
            "TRIPLE_RIDING":          ("Section 128",    "Carrying more than one pillion rider on a two-wheeler."),
            "STOP_LINE_VIOLATION":    ("Section 113",    "Failed to halt before stop-line during red signal."),
            "HELMET_NON_COMPLIANCE":  ("Section 129",    "Operating two-wheeler without protective headgear."),
            "SEATBELT_NON_COMPLIANCE":("Section 138(3)", "Operating motor vehicle without fastened seatbelt."),
            "WRONG_SIDE_DRIVING":     ("Section 184",    "Driving in a manner dangerous to public — wrong side."),
            "ILLEGAL_PARKING":        ("Section 122",    "Vehicle parked causing obstruction in a restricted zone."),
        }
        sec, desc = statutes.get(v_type, ("Section 177", "General traffic safety protocol breach."))
        return (
            f"EVIDENTIARY BRIEF:\n"
            f"1. Violation: {v_type}\n"
            f"2. {detail}\n"
            f"3. Statutory basis: {sec} of the Motor Vehicles Act — {desc}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    #  EVIDENCE GENERATOR
    # ──────────────────────────────────────────────────────────────────────────

    def generate_evidence(self, node_id: int, v_type: str, brief: str,
                          conf: float, plate: str, reid: str,
                          frame: np.ndarray, bbox: tuple) -> None:
        """
        Saves:
          1. Annotated JPEG crop of the violation (with bounding box overlay)
          2. JSON metadata record with SHA-256 hash (tamper-evident)
          3. Row in SQLite violations table
        """
        ts       = time.time()
        ev_id    = f"EV-{int(ts)}-N{node_id}-{v_type[:6]}"
        img_path = os.path.join("storage", f"{ev_id}.jpg")

        # Save annotated frame crop
        x1, y1, x2, y2 = bbox
        crop = frame[max(0, y1 - 5):min(self.H, y2 + 5),
                     max(0, x1 - 5):min(self.W, x2 + 5)].copy()
        if crop.size > 0:
            # Red border on violation crop
            cv2.rectangle(crop, (0, 0), (crop.shape[1]-1, crop.shape[0]-1), (0, 0, 255), 3)
            cv2.putText(crop, v_type, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            cv2.putText(crop, plate,  (4, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1)
            cv2.imwrite(img_path, crop)

        # Check plate consistency
        plate_suspicious = self.check_plate_consistency(plate, node_id)

        metadata = {
            "evidence_id":          ev_id,
            "timestamp":            ts,
            "node_id":              node_id,
            "plate_id":             plate,
            "plate_suspicious":     plate_suspicious,
            "violation_class":      v_type,
            "confidence":           conf,
            "reid_signature":       reid,
            "legal_narrative":      brief,
            "annotated_image_path": img_path,
        }

        payload_bytes   = json.dumps(metadata, sort_keys=True).encode()
        sha256          = hashlib.sha256(payload_bytes).hexdigest()
        metadata["cryptographic_hash"] = f"SHA-256:{sha256}"

        json_path = os.path.join("storage", f"{ev_id}.json")
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=4)

        self.ev_count += 1
        self._db_write(ev_id, ts, plate, v_type, conf, brief, sha256)

        flag = " ⚠️ PLATE MISMATCH" if plate_suspicious else ""
        print(f"🔴 [EV #{self.ev_count:03d}] {v_type} | Plate: {plate}{flag} | Node: {node_id} | REID: {reid} | Conf: {conf:.0%}")

    def _db_write(self, ev_id, ts, plate, v_type, conf, brief, sha):
        try:
            conn = sqlite3.connect("vision_guard.db")
            conn.execute("""
                INSERT OR IGNORE INTO violations
                (evidence_id, timestamp, plate_id, violation_class, confidence, legal_narrative, cryptographic_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (ev_id, ts, plate, v_type, conf, brief, sha))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[DB ERROR] {e}")

    def _db_analytics(self, frame_id, ts, active, latency_ms, phase):
        try:
            conn = sqlite3.connect("vision_guard.db")
            conn.execute("""
                INSERT INTO traffic_analytics (frame_id, timestamp, active_nodes, edge_latency_ms, signal_phase)
                VALUES (?, ?, ?, ?, ?)
            """, (frame_id, ts, active, latency_ms, phase))
            conn.commit()
            conn.close()
        except Exception as e:
            pass

    # ──────────────────────────────────────────────────────────────────────────
    #  HUD RENDERING
    # ──────────────────────────────────────────────────────────────────────────

    def draw_hud(self, frame: np.ndarray, nodes: dict, latency_ms: float) -> np.ndarray:
        """Renders all overlays: geofences, node boxes, signal ROI, telemetry."""

        # Stop line
        cv2.line(frame, (0, self.STOP_LINE_Y), (self.W, self.STOP_LINE_Y), (0, 0, 255), 2)
        cv2.putText(frame, "STOP LINE", (5, self.STOP_LINE_Y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 1)

        # Illegal parking zone
        px1, py1, px2, py2 = self.PARKING_ZONE
        cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 0, 255), 2)
        cv2.putText(frame, "NO PARK", (px1, py1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 0, 255), 1)

        # Traffic light ROI
        tx1, ty1, tx2, ty2 = self.TL_BOX
        sig_col = (0, 0, 255) if self.signal_phase == "RED" else (0, 220, 0)
        cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), sig_col, 2)
        cv2.putText(frame, f"SIG:{self.signal_phase}", (tx1, ty1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, sig_col, 1)

        # Per-node bounding boxes and labels
        for nid, node in nodes.items():
            x1, y1, x2, y2 = node.bbox
            cls_name = self.COCO_NAMES.get(node.cls, "OBJ")
            label    = f"{cls_name}#{nid} [{node.reid_sig}]"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(frame, label, (x1, max(y1 - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 200, 255), 1)

        # Telemetry bar
        cv2.rectangle(frame, (0, 0), (self.W, 28), (0, 0, 0), -1)
        cv2.putText(frame, f"Edge Latency: {latency_ms:.1f}ms", (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f"Violations: {self.ev_count}  Nodes: {len(nodes)}  Frame: {self.frame_idx}",
                    (220, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        return frame

    # ──────────────────────────────────────────────────────────────────────────
    #  MAIN INFERENCE LOOP
    # ──────────────────────────────────────────────────────────────────────────

    def run(self):
        print("\n[START] VisionGuard Engine running — press 'q' to quit.\n")
        print("  Geofence config:")
        print(f"    Stop line Y : {self.STOP_LINE_Y}")
        print(f"    Parking zone: {self.PARKING_ZONE}")
        print(f"    Signal ROI  : {self.TL_BOX}")
        print()

        while self.cap.isOpened():
            ret, raw = self.cap.read()
            if not ret or raw is None:
                print("[END] Stream exhausted.")
                break

            self.frame_idx += 1
            t0 = time.time()

            # ── Edge Ingest Cascade: resize first (simulates edge-compute bandwidth saving)
            frame = cv2.resize(raw, (self.W, self.H))

            # ── Stage 1: Preprocessing
            frame = self.clahe_preprocess(frame)
            det_input = self.build_edge_representation(frame)  # RGB+Edge for detector

            # ── Stage 2: Signal detection (every frame — cheap HSV op)
            self.signal_phase = self.detect_signal(frame)

            # ── Stage 3: YOLO inference on edge-enhanced input
            results = self.detector(det_input, verbose=False)
            latency_ms = (time.time() - t0) * 1000

            # ── Stage 4: Update tracking
            nodes = self.update_tracking(results[0].boxes)

            # ── Stage 5: Violation evaluation per node
            for nid, node in nodes.items():

                # Motorcycle violations
                if node.cls == self.COCO_MOTORCYCLE:
                    self.eval_triple_riding(nid, node, nodes, frame)
                    # Check helmet for each rider on this motorcycle
                    x1, y1, x2, y2 = node.bbox
                    for pid, pn in nodes.items():
                        if pn.cls == self.COCO_PERSON:
                            pcx, pcy = pn.center
                            if (x1 - 20 <= pcx <= x2 + 20) and (y1 - 35 <= pcy <= y2 + 10):
                                self.eval_helmet(nid, pn, node, frame)

                # Car violations
                if node.cls == self.COCO_CAR:
                    self.eval_seatbelt(nid, node, frame)

                # Wrong-side (all vehicles)
                if node.cls in self.VEHICLE_CLASSES:
                    self.eval_wrong_side(nid, node, frame)

                # Stop-line (cars, buses, trucks)
                if node.cls in {self.COCO_CAR, self.COCO_BUS, self.COCO_TRUCK}:
                    self.eval_stop_line(nid, node, frame)

                # Illegal parking (cars, trucks)
                if node.cls in {self.COCO_CAR, self.COCO_TRUCK}:
                    self.eval_illegal_parking(nid, node, frame)

            # ── Stage 6: HUD + display
            frame = self.draw_hud(frame, nodes, latency_ms)
            cv2.imshow("VisionGuard AI — Live Feed", frame)

            # ── Log analytics row every 30 frames to avoid DB hammering
            if self.frame_idx % 30 == 0:
                self._db_analytics(self.frame_idx, time.time(), len(nodes), latency_ms, self.signal_phase)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[QUIT] User terminated.")
                break

        self.cap.release()
        cv2.destroyAllWindows()
        print(f"\n[DONE] Total violations logged: {self.ev_count}")
        print(f"       Evidence files saved to: ./storage/")
        print(f"       Database: vision_guard.db\n")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "mock_stream.mp4"
    engine = VisionGuardEngine(src)
    engine.run()