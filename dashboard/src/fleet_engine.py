"""
VisionGuard AI — Ekart Fleet Safety Gamification Engine (Feature 16)

Per-partner safety scores, streaks, insurance API link, coaching triggers.
"""

import time, uuid, random
from dataclasses import dataclass, field, asdict
from typing import Optional

# ── Score deductions per violation ───────────────────────────────────────────
DEDUCTIONS = {
    "helmet_non_compliance":   15,
    "seatbelt_non_compliance": 10,
    "triple_riding":           25,
    "wrong_side_driving":      30,
    "red_light_violation":     20,
    "stop_line_violation":     10,
    "illegal_parking":         10,
    "mobile_phone_use":        20,
    "overloading":             15,
    "no_number_plate":         20,
    "registration_spoof":      50,
}

# ── Intervention policy ───────────────────────────────────────────────────────
INTERVENTIONS = {
    80: "advisory_push",
    65: "mandatory_module",
    50: "supervisor_review",
    35: "temporary_suspension",
    20: "contract_review",
}

# ── Digit Insurance tiers (mock) ──────────────────────────────────────────────
INSURANCE_TIERS = [
    {"min": 90, "label": "Platinum",  "discount_pct": 25, "monthly_premium": 180},
    {"min": 75, "label": "Gold",      "discount_pct": 15, "monthly_premium": 250},
    {"min": 60, "label": "Silver",    "discount_pct":  5, "monthly_premium": 380},
    {"min":  0, "label": "Standard",  "discount_pct":  0, "monthly_premium": 480},
]

COACHING_MODULES = {
    "helmet_non_compliance":   "Helmet Safety & ISI Standards (4 min video)",
    "triple_riding":           "Load Limits & Passenger Safety (3 min)",
    "mobile_phone_use":        "Distracted Riding — The Hidden Killer (5 min)",
    "wrong_side_driving":      "Traffic Flow & Right-of-Way Rules (6 min)",
    "red_light_violation":     "Signal Discipline & Intersection Safety (4 min)",
    "registration_spoof":      "Vehicle Documentation Compliance (8 min)",
    "seatbelt_non_compliance": "Seatbelt Physics — Why It Saves Lives (3 min)",
}

@dataclass
class PartnerProfile:
    partner_id: str
    name: str
    zone: str
    vehicle_id: str
    vehicle_type: str = "two_wheeler"
    safety_score: float = 100.0
    total_deliveries: int = 0
    streak_days: int = 0
    last_violation_ts: Optional[str] = None
    violations_this_month: int = 0
    coaching_completed: list = field(default_factory=list)
    badges: list = field(default_factory=list)
    total_fine_avoided_inr: int = 0

    def insurance_tier(self) -> dict:
        for t in INSURANCE_TIERS:
            if self.safety_score >= t["min"]:
                return t
        return INSURANCE_TIERS[-1]

    def intervention(self) -> str:
        for threshold in sorted(INTERVENTIONS, reverse=True):
            if self.safety_score <= threshold:
                return INTERVENTIONS[threshold]
        return "none"

    def to_dict(self):
        d = asdict(self)
        d["insurance_tier"] = self.insurance_tier()
        d["intervention"]   = self.intervention()
        return d

class FleetSafetyEngine:
    def __init__(self):
        self.partners: dict[str, PartnerProfile] = {}
        self.events: list[dict] = []
        self._seed_demo_partners()

    def _seed_demo_partners(self):
        demo = [
            ("EK001", "Ravi Kumar",   "Koregaon Park", "MH12EK0042", 94.0, 312, 14),
            ("EK002", "Suresh Patil", "Hinjewadi",      "MH14AB5678", 61.0, 187,  2),
            ("EK003", "Amit Singh",   "Viman Nagar",    "MH12CD1234", 78.5, 501,  7),
            ("EK004", "Priya Nair",   "Kharadi",        "KA01MJ9999", 45.0,  98,  0),
            ("EK005", "Deepak Rao",   "Wakad",          "MH14EK7890", 88.0, 430, 11),
        ]
        for pid, name, zone, vid, score, deliveries, streak in demo:
            p = PartnerProfile(
                partner_id=pid, name=name, zone=zone,
                vehicle_id=vid, safety_score=score,
                total_deliveries=deliveries, streak_days=streak,
                violations_this_month=max(0, int((100 - score) / 8)),
            )
            # seed badges
            if streak >= 10: p.badges.append("10-Day Streak")
            if score >= 90:  p.badges.append("Platinum Rider")
            if deliveries >= 400: p.badges.append("Road Veteran")
            self.partners[pid] = p

    def register_partner(self, name: str, zone: str, vehicle_id: str,
                         vehicle_type: str = "two_wheeler") -> PartnerProfile:
        pid = f"EK{random.randint(100,999)}"
        p = PartnerProfile(partner_id=pid, name=name, zone=zone,
                           vehicle_id=vehicle_id, vehicle_type=vehicle_type)
        self.partners[pid] = p
        return p

    def record_violation(self, partner_id: str,
                         violation_types: list[str],
                         fine_inr: int,
                         location: str = "",
                         severity: str = "HIGH") -> dict:
        if partner_id not in self.partners:
            return {"error": "partner not found"}

        p = self.partners[partner_id]
        old_score = p.safety_score
        old_tier  = p.insurance_tier()["label"]

        deduction = sum(DEDUCTIONS.get(v, 10) for v in violation_types)
        p.safety_score = max(0, p.safety_score - deduction)
        p.violations_this_month += 1
        p.streak_days = 0
        p.last_violation_ts = time.strftime("%Y-%m-%dT%H:%M:%SZ")

        new_tier = p.insurance_tier()["label"]
        intervention = p.intervention()

        # Coaching modules
        modules = [COACHING_MODULES[v] for v in violation_types if v in COACHING_MODULES]

        # Build notification payload (what partner app receives)
        notification = {
            "type": "violation_alert",
            "title": "Safety alert recorded",
            "body": f"A violation was detected for your vehicle. Your safety score has been updated.",
            "score_change": f"{old_score:.0f} → {p.safety_score:.0f}",
            "deduction": deduction,
            "violations": violation_types,
            "fine_avoided_if_corrected_inr": fine_inr,
            "insurance_tier": f"{old_tier} → {new_tier}" if new_tier != old_tier else old_tier,
            "action_required": intervention.replace("_", " ").title(),
            "coaching_modules": modules,
            "timestamp": p.last_violation_ts,
        }

        event = {
            "event_id": str(uuid.uuid4())[:8],
            "partner_id": partner_id,
            "partner_name": p.name,
            "violations": violation_types,
            "deduction": deduction,
            "score_before": old_score,
            "score_after": p.safety_score,
            "fine_inr": fine_inr,
            "severity": severity,
            "location": location,
            "intervention": intervention,
            "timestamp": p.last_violation_ts,
            "notification": notification,
        }
        self.events.append(event)
        return event

    def record_safe_delivery(self, partner_id: str) -> dict:
        if partner_id not in self.partners:
            return {}
        p = self.partners[partner_id]
        p.total_deliveries += 1
        p.streak_days += 1
        recovery = min(2.0, p.streak_days * 0.1)
        p.safety_score = min(100, p.safety_score + recovery)

        new_badges = []
        for days, badge in [(7,"7-Day Streak"),(14,"14-Day Streak"),(30,"30-Day Streak")]:
            if p.streak_days == days and badge not in p.badges:
                p.badges.append(badge)
                new_badges.append(badge)
        if p.safety_score >= 90 and "Platinum Rider" not in p.badges:
            p.badges.append("Platinum Rider")
            new_badges.append("Platinum Rider")

        return {"partner_id": partner_id, "streak": p.streak_days,
                "score": round(p.safety_score, 1), "new_badges": new_badges,
                "score_recovery": round(recovery, 2)}

    def get_fleet_report(self) -> dict:
        ps = list(self.partners.values())
        if not ps:
            return {}
        scores = [p.safety_score for p in ps]
        at_risk  = sum(1 for s in scores if s < 65)
        critical = sum(1 for s in scores if s < 35)
        total_premium_saved = sum(
            max(0, INSURANCE_TIERS[-1]["monthly_premium"] - p.insurance_tier()["monthly_premium"])
            for p in ps
        ) * len(ps)

        zone_violations: dict[str, int] = {}
        for e in self.events:
            z = e.get("location", "Unknown").split(",")[0].strip()
            zone_violations[z] = zone_violations.get(z, 0) + len(e.get("violations", []))

        return {
            "total_partners":       len(ps),
            "avg_safety_score":     round(sum(scores)/len(scores), 1),
            "partners_at_risk":     at_risk,
            "critical_cases":       critical,
            "total_violations":     len(self.events),
            "monthly_premium_pool_saved_inr": total_premium_saved,
            "top_zones_by_violations": dict(sorted(zone_violations.items(), key=lambda x:-x[1])[:5]),
            "insurance_breakdown":  {
                t["label"]: sum(1 for p in ps if p.insurance_tier()["label"] == t["label"])
                for t in INSURANCE_TIERS
            },
        }

    def get_partner(self, pid: str) -> Optional[PartnerProfile]:
        return self.partners.get(pid)

    def list_partners(self) -> list[dict]:
        return [p.to_dict() for p in self.partners.values()]

# Singleton for the API to import
fleet_engine = FleetSafetyEngine()