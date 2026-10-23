"""HTTP API for face-recognition attendance and reports.

Run with:
    uvicorn api:app --reload
"""

import base64
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
MAX_BATCH_IMAGES = 25
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


def _annotated_image(frame, faces: list[tuple]) -> str:
    """Encode the processed frame with recognition overlays for the UI."""
    for top, right, bottom, left, name, sid, confidence in faces:
        color = (16, 185, 129) if sid != "unknown" else (14, 165, 233)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 3)
        label = f"{name} ({confidence:.1f}%)" if sid != "unknown" else "Unknown"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
        )
        label_top = max(top - text_height - baseline - 8, 0)
        cv2.rectangle(frame, (left, label_top), (left + text_width + 12, top), color, -1)
        cv2.putText(
            frame,
            label,
            (left + 6, top - 6 if top > text_height + baseline + 8 else label_top + text_height + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    encoded, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not encoded:
        raise HTTPException(status_code=500, detail="Could not encode the result image")
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("ascii")


def _system(
    tolerance: float,
    scale: float,
    model: str,
    upsample: int = 1,
    use_haar: bool = True,
) -> AttendanceSystem:
    if not 0 < tolerance <= 1:
        raise HTTPException(status_code=400, detail="tolerance must be greater than 0 and at most 1")
    if not 0 < scale <= 1:
        raise HTTPException(status_code=400, detail="scale must be greater than 0 and at most 1")
    if model not in {"hog", "cnn"}:
        raise HTTPException(status_code=400, detail="model must be 'hog' or 'cnn'")
    if upsample not in {0, 1, 2}:
        raise HTTPException(status_code=400, detail="upsample must be 0, 1, or 2")
    return AttendanceSystem(
        tolerance=tolerance,
        scale_factor=scale,
        model=model,
        upsample=upsample,
        use_haar=use_haar,
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/attendance/process")
async def process_attendance(
    image: list[UploadFile] = File(...),
    tolerance: float = Form(0.50),
    scale: float = Form(1.0),
    model: str = Form("hog"),
    upsample: int = Form(1),
    use_haar: bool = Form(True),
) -> dict:
    """Recognize faces in one or more uploaded images and mark attendance."""
    if not image:
        raise HTTPException(status_code=400, detail="Upload at least one image")
    if len(image) > MAX_BATCH_IMAGES:
        raise HTTPException(status_code=413, detail=f"Upload at most {MAX_BATCH_IMAGES} images at a time")

    system = _system(tolerance, scale, model, upsample=upsample, use_haar=use_haar)
    if not system.known_encodings:
        raise HTTPException(status_code=503, detail="No enrolled face encodings found")

    results = []
    for upload in image:
        if upload.content_type not in SUPPORTED_IMAGE_TYPES:
            raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, WEBP, BMP, or TIFF image")

        contents = await upload.read()
        if not contents:
            raise HTTPException(status_code=400, detail=f"Uploaded image is empty: {upload.filename}")
        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"Image must be 15 MB or smaller: {upload.filename}")

        frame = cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail=f"The uploaded file is not a readable image: {upload.filename}")

        faces = system._process_frame(frame, is_static_image=(scale == 1.0))
        image_faces = [
            {
                "student_id": sid,
                "name": name,
                "confidence": confidence,
                "recognized": sid != "unknown",
                "location": {"top": top, "right": right, "bottom": bottom, "left": left},
            }
            for top, right, bottom, left, name, sid, confidence in faces
        ]
        results.append({
            "filename": upload.filename,
            "faces": image_faces,
            "summary": {
                "total": len(image_faces),
                "recognized": sum(face["recognized"] for face in image_faces),
                "unknown": sum(not face["recognized"] for face in image_faces),
            },
            "result_image": _annotated_image(frame, faces),
        })

    db = system.db
    report = db.today_report()
    exported = db.export_csv()
    first_result = results[0]
    return {
        "date": date.today().isoformat(),
        "filename": first_result["filename"],
        "faces": first_result["faces"],
        "result_image": first_result["result_image"],
        "results": results,
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