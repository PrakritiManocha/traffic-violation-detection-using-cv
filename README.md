# VisionAI: Turning Traffic Images into Actionable Intelligence

## Overview

VisionAI is an end-to-end computer vision platform that automatically analyzes traffic images and video streams, detects traffic violations, classifies offenses, recognizes license plates, generates evidentiary records, and provides actionable analytics.

Developed for the **Flipkart GridLock Hackathon – Automated Photo Identification and Classification for Traffic Violations Using Computer Vision**, VisionAI transforms raw traffic imagery into structured, review-ready traffic violation records with minimal human intervention.


## Problem Statement

Traffic surveillance systems generate enormous amounts of visual data every day, yet a large portion of this data remains unreviewed due to the time, cost, and effort required for manual inspection.

This results in:

* Missed violations
* Delayed enforcement
* Increased operational costs
* Underutilized traffic intelligence

VisionAI addresses this challenge by automatically identifying, classifying, documenting, and reporting traffic violations at scale.


## Key Features

### Image Preprocessing

* CLAHE-based image enhancement
* RGB + Edge representation
* Illumination normalization
* Robustness against glare, shadows, rain, and low-light conditions

### Vehicle & Road User Detection

* Vehicle detection and localization
* Rider and road-user identification
* Vehicle categorization
* Lightweight edge-friendly processing pipeline

### Traffic Violation Detection

* Helmet Non-Compliance
* Seatbelt Non-Compliance
* Triple Riding
* Wrong-Side Driving
* Stop-Line Violations
* Red-Light Violations
* Illegal Parking

### Violation Classification

* Automatic violation categorization
* Confidence score assignment
* Structured infraction records

### License Plate Recognition

* Number plate localization
* OCR-based registration extraction
* Vehicle identity association

### Evidence Generation

* Annotated evidence imagery
* Timestamped violation records
* Legal narratives
* Structured JSON evidence packages
* SHA-256 tamper-evident integrity hashes

### Analytics & Reporting

* Searchable violation records
* Violation statistics
* Traffic trend analysis
* Reporting dashboard


## What Makes VisionAI Different?

Most traffic monitoring systems stop at detecting a violation.

VisionAI focuses on the complete enforcement workflow—from detection to evidence generation.

A key challenge in automated traffic monitoring is the **Courtroom Defensibility Bottleneck**: detecting a violation is not enough; authorities must also be able to explain and justify why it was detected.

VisionAI automatically converts detections into structured evidence packages containing annotated images, violation details, timestamps, confidence scores, legal reasoning, and integrity hashes, making records easier to review, verify, and defend.


## System Workflow

```text
Traffic Image / Video
          ↓
Image Preprocessing
          ↓
Vehicle & Road User Detection
          ↓
Traffic Violation Detection
          ↓
Violation Classification
          ↓
License Plate Recognition
          ↓
Evidence Generation
          ↓
Analytics & Reporting
```

---

## Architecture Highlights

* Edge-Native Processing
* CLAHE-Based Enhancement
* RGB + Edge Visual Representation
* Cognitive Law Engine
* Legal Narrative Generation
* Courtroom-Ready Evidence Packaging
* SHA-256 Tamper-Evident Records
* Modular and Scalable Design


## Project Structure

```text
VisionAI/
│
├── dashboard/
│   │
│   ├── api/
│   │   └── server.py
│   │
│   ├── src/
│   │   ├── detection_engine.py
│   │   └── fleet_engine.py
│   │
│   └── index.html
│
├── live_simulation/
│   │
│   ├── models/
│   │
│   ├── storage/
│   │   ├── EV-XXXXXXXX-NX-WRONG_.jpg
│   │   ├── EV-XXXXXXXX-NX-WRONG_.json
│   │   └── ...
│   │
│   ├── core_pipeline.py
│   ├── database_setup.py
│   ├── vision_guard.py
│   ├── mock_stream.mp4
│   ├── vision_guard.db
│   └── yolo8n.pt
│
├── requirements.txt
│
└── README.md
```


## Why Are Dashboard and Simulation Separate?

The Dashboard and Live Simulation modules were intentionally maintained as separate components to simplify development, testing, and demonstration.

In a production deployment, these modules would communicate through APIs while remaining independently scalable and maintainable.


## Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```


## Running the Dashboard

Navigate to the dashboard directory:

```bash
cd dashboard
```

Start the FastAPI backend:

```bash
uvicorn api.server:app --reload --port 8000
```

Launch the dashboard frontend:

```bash
start index.html
```


## Running the Live Simulation

Navigate to the simulation directory:

```bash
cd live_simulation
```

Place a traffic video named:

```text
mock_stream.mp4
```

You may replace it with any local traffic video while keeping the filename unchanged.

Run the simulation:

```bash
python vision_guard.py
```


## Generated Outputs

For every detected violation, VisionAI generates:

### Annotated Evidence Image

```text
storage/EV-XXXXXXXX.jpg
```

### Structured Evidence Record

```json
{
  "evidence_id": "EV-1782229724-N3-WRONG_",
  "timestamp": 1782229724,
  "node_id": 3,
  "violation_class": "WRONG_SIDE_DRIVING",
  "confidence": 0.89,
  "legal_narrative": "...",
  "annotated_image_path": "...",
  "cryptographic_hash": "SHA-256:..."
}
```

The generated evidence package contains:

* Evidence ID
* Timestamp
* Camera/Node Identifier
* Violation Category
* Confidence Score
* Tracking Signature
* Legal Narrative
* Annotated Evidence Image
* Cryptographic Integrity Hash


## Technology Stack

### Computer Vision

* OpenCV
* YOLOv8
* OCR Pipeline
* Image Processing Techniques

### Backend

* Python
* FastAPI
* SQLite

### Frontend

* HTML
* CSS
* JavaScript

### Security

* SHA-256 Evidence Hashing


## Future Scope

* Cross-Camera Vehicle Re-Identification
* Advanced OCR Pipeline
* Open-Vocabulary Violation Detection
* Grounding DINO Integration
* SAM2-Based Segmentation
* Edge Deployment on Jetson Hardware
* PostgreSQL/PostGIS Analytics
* Smart-City Integrations
* Fleet Safety and Compliance Monitoring


## Vision

**What if every traffic image could automatically explain what happened, who was involved, and why it matters?**

VisionAI aims to transform ordinary traffic cameras into intelligent traffic enforcement systems capable of automatically understanding, documenting, and reporting traffic violations at scale.
