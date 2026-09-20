"""HTTP API for face-recognition attendance and reports.

Run with:
    uvicorn api:app --reload
"""

import os
from datetime import date
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from face_attendance.face_attendance import AttendanceDB, AttendanceSystem
from report import report_for_date, report_range


app = FastAPI(title="Face Attendance API", version="1.0.0")

origins = os.getenv("API_CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in origins],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_UPLOAD_BYTES = 15 * 1024 * 1024
SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"}


def _records(frame) -> list[dict]:
    """Convert a report DataFrame into JSON-safe records."""
    return frame.where(frame.notna(), None).to_dict(orient="records")


def _summary(frame) -> dict:
    present = int((frame["status"] == "Present").sum()) if "status" in frame else 0
    total = len(frame)
    return {
        "present": present,
        "total": total,
        "percentage": round(present / max(total, 1) * 100, 1),
    }


def _system(tolerance: float, scale: float, model: str) -> AttendanceSystem:
    if not 0 < tolerance <= 1:
        raise HTTPException(status_code=400, detail="tolerance must be greater than 0 and at most 1")
    if not 0 < scale <= 1:
        raise HTTPException(status_code=400, detail="scale must be greater than 0 and at most 1")
    if model not in {"hog", "cnn"}:
        raise HTTPException(status_code=400, detail="model must be 'hog' or 'cnn'")
    return AttendanceSystem(tolerance=tolerance, scale_factor=scale, model=model)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/attendance/process")
async def process_attendance(
    image: UploadFile = File(...),
    tolerance: float = Form(0.45),
    scale: float = Form(0.25),
    model: str = Form("hog"),
) -> dict:
    """Recognize all faces in one uploaded image and mark today's attendance."""
    if image.content_type not in SUPPORTED_IMAGE_TYPES:
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, WEBP, BMP, or TIFF image")

    contents = await image.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image must be 15 MB or smaller")

    frame = cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="The uploaded file is not a readable image")

    system = _system(tolerance, scale, model)
    if not system.known_encodings:
        raise HTTPException(status_code=503, detail="No enrolled face encodings found")

    faces = system._process_frame(frame)
    db = system.db
    report = db.today_report()
    recognized = [
        {
            "student_id": sid,
            "name": name,
            "confidence": confidence,
            "recognized": sid != "unknown",
            "location": {"top": top, "right": right, "bottom": bottom, "left": left},
        }
        for top, right, bottom, left, name, sid, confidence in faces
    ]
    exported = db.export_csv()
    return {
        "date": date.today().isoformat(),
        "filename": image.filename,
        "faces": recognized,
        "attendance": _records(report),
        "summary": _summary(report),
        "csv_file": exported.name,
    }


@app.get("/attendance/report")
def attendance_report(
    report_date: Optional[str] = Query(None, alias="date"),
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> dict:
    """Return a report for one date or an inclusive date range."""
    if (start is None) != (end is None):
        raise HTTPException(status_code=400, detail="start and end must be provided together")

    db = AttendanceDB()
    if start and end:
        frame = report_range(db, start, end)
        report_kind = "range"
    else:
        target = report_date or date.today().isoformat()
        frame = report_for_date(db, target)
        report_kind = "date"
    return {"type": report_kind, "attendance": _records(frame), "summary": _summary(frame)}


@app.get("/attendance/report.csv")
def download_report(
    report_date: Optional[str] = Query(None, alias="date"),
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> FileResponse:
    """Generate and download a CSV report for one date or a date range."""
    if (start is None) != (end is None):
        raise HTTPException(status_code=400, detail="start and end must be provided together")

    db = AttendanceDB()
    if start and end:
        frame = report_range(db, start, end)
        filename = f"report_{start}_{end}.csv"
    else:
        target = report_date or date.today().isoformat()
        frame = report_for_date(db, target)
        filename = f"report_{target}.csv"

    output = Path(db.db_path).parent / "reports" / filename
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return FileResponse(output, media_type="text/csv", filename=filename)