"""
VisionGuard AI — FastAPI Backend
Run: uvicorn api.server:app --reload --port 8000
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import base64, time, random, numpy as np, cv2

from src.detection_engine import run_pipeline, VIOLATIONS, generate_demo_frame, frame_to_b64
from src.fleet_engine import fleet_engine

app = FastAPI(
    title="VisionGuard AI",
    description="VLM-augmented traffic violation detection with Ekart fleet safety gamification",
    version="1.0.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"service": "VisionGuard AI", "version": "1.0.0", "status": "operational"}

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")}

# ── Detection endpoints ───────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    image_b64: Optional[str] = None
    scenario: str = "triple_riding"
    camera_id: str = "CAM_001"
    location: str = "MG Road, Bengaluru"
    vehicle_type: str = "motorcycle"
    vehicle_color: str = "black"
    plate_text: str = "MH12AB1234"
    partner_id: Optional[str] = None

@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    rec = run_pipeline(
        image_input  = req.image_b64,
        scenario     = req.scenario,
        camera_id    = req.camera_id,
        location     = req.location,
        vehicle_type = req.vehicle_type,
        vehicle_color= req.vehicle_color,
        plate_text   = req.plate_text,
        partner_id   = req.partner_id,
    )
    result = rec.to_dict()

    # If fleet partner, trigger safety event
    if req.partner_id and req.partner_id in fleet_engine.partners:
        fleet_event = fleet_engine.record_violation(
            partner_id     = req.partner_id,
            violation_types= rec.violation_types,
            fine_inr       = rec.fine_inr,
            location       = req.location,
            severity       = rec.severity,
        )
        result["fleet_event"] = fleet_event

    return result

@app.post("/analyze/upload")
async def analyze_upload(
    file: UploadFile = File(...),
    scenario: str = "triple_riding",
    partner_id: Optional[str] = None,
):
    contents = await file.read()
    arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Invalid image file")
    b64 = base64.b64encode(contents).decode()
    rec = run_pipeline(image_input=b64, scenario=scenario, partner_id=partner_id)
    return rec.to_dict()

@app.get("/demo/frame/{scenario}")
def demo_frame(scenario: str):
    """Return a synthetic annotated demo frame."""
    frame = generate_demo_frame(scenario)
    return {"image_b64": frame_to_b64(frame), "scenario": scenario}

@app.get("/violations/taxonomy")
def taxonomy():
    return VIOLATIONS

# ── Fleet endpoints ───────────────────────────────────────────────────────────
@app.get("/fleet/partners")
def list_partners():
    return fleet_engine.list_partners()

@app.get("/fleet/partners/{partner_id}")
def get_partner(partner_id: str):
    p = fleet_engine.get_partner(partner_id)
    if not p:
        raise HTTPException(404, "Partner not found")
    return p.to_dict()

class ViolationEvent(BaseModel):
    violation_types: list[str]
    fine_inr: int = 1000
    location: str = "Bengaluru"
    severity: str = "HIGH"

@app.post("/fleet/partners/{partner_id}/violation")
def record_partner_violation(partner_id: str, ev: ViolationEvent):
    result = fleet_engine.record_violation(
        partner_id=partner_id, violation_types=ev.violation_types,
        fine_inr=ev.fine_inr, location=ev.location, severity=ev.severity,
    )
    if "error" in result:
        raise HTTPException(404, result["error"])
    return result

@app.post("/fleet/partners/{partner_id}/safe_delivery")
def safe_delivery(partner_id: str):
    return fleet_engine.record_safe_delivery(partner_id)

@app.get("/fleet/report")
def fleet_report():
    return fleet_engine.get_fleet_report()

@app.get("/fleet/events")
def fleet_events(limit: int = 20):
    return {"events": fleet_engine.events[-limit:], "total": len(fleet_engine.events)}

# ── Stats for dashboard ───────────────────────────────────────────────────────
@app.get("/stats/live")
def live_stats():
    partners = fleet_engine.list_partners()
    avg_score = round(sum(p["safety_score"] for p in partners) / max(len(partners), 1), 1)
    return {
        "violations_today":   random.randint(120, 180),
        "citations_issued":   random.randint(95, 150),
        "fines_collected_inr":random.randint(180000, 320000),
        "cameras_active":     random.randint(18, 24),
        "fleet_avg_score":    avg_score,
        "critical_alerts":    sum(1 for p in partners if p["safety_score"] < 35),
        "pre_violation_alerts": random.randint(12, 28),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)