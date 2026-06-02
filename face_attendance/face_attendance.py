"""
================================================================================
  Face Recognition Attendance System
  Detects multiple students simultaneously from webcam / video feed
  and logs attendance to CSV + SQLite.
================================================================================
"""

import cv2
import face_recognition
import numpy as np
import pandas as pd
import sqlite3
import os
import json
import time
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Optional
import sys

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("attendance_system.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# Ensure console streams use UTF-8 on platforms (Windows) where the
# default encoding may not support some characters used in log messages.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    # reconfigure may not be available in some environments; ignore.
    pass

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
STUDENTS_DIR    = BASE_DIR / "students"          # one sub-folder per student
ENCODINGS_FILE  = BASE_DIR / "encodings.json"    # cached face encodings
DB_FILE         = BASE_DIR / "attendance.db"
REPORTS_DIR     = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
STUDENTS_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Database helpers
# ══════════════════════════════════════════════════════════════════════════════
class AttendanceDB:
    """Thin wrapper around an SQLite attendance database."""

    def __init__(self, db_path: Path = DB_FILE):
        self.db_path = db_path
        self._init_db()

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._conn() as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS students (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id  TEXT UNIQUE NOT NULL,
                    name        TEXT NOT NULL,
                    created_at  TEXT DEFAULT (datetime('now'))
                )
            """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS attendance (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id  TEXT NOT NULL,
                    date        TEXT NOT NULL,
                    time_in     TEXT NOT NULL,
                    status      TEXT DEFAULT 'Present',
                    UNIQUE(student_id, date)
                )
            """)
        log.info("Database initialised at %s", self.db_path)

    def register_student(self, student_id: str, name: str):
        with self._conn() as con:
            con.execute(
                "INSERT OR IGNORE INTO students (student_id, name) VALUES (?,?)",
                (student_id, name),
            )

    def mark_attendance(self, student_id: str) -> bool:
        """Returns True if this is the first mark today."""
        today    = date.today().isoformat()
        time_now = datetime.now().strftime("%H:%M:%S")
        with self._conn() as con:
            cur = con.execute(
                "SELECT id FROM attendance WHERE student_id=? AND date=?",
                (student_id, today),
            )
            if cur.fetchone():
                return False          # already marked today
            con.execute(
                "INSERT INTO attendance (student_id, date, time_in) VALUES (?,?,?)",
                (student_id, today, time_now),
            )
        return True

    def today_report(self) -> pd.DataFrame:
        today = date.today().isoformat()
        with self._conn() as con:
            df = pd.read_sql_query(
                """SELECT s.student_id, s.name, a.time_in, a.status
                   FROM students s
                   LEFT JOIN attendance a
                          ON s.student_id = a.student_id AND a.date = ?
                   ORDER BY s.name""",
                con,
                params=(today,),
            )
        df["status"] = df["status"].fillna("Absent")
        return df

    def export_csv(self, output_path: Optional[Path] = None) -> Path:
        today  = date.today().isoformat()
        output_path = output_path or REPORTS_DIR / f"attendance_{today}.csv"
        try:
            self.today_report().to_csv(output_path, index=False)
            out = output_path
        except PermissionError:
            # File may be open/locked (e.g. Excel). Fallback to a timestamped file.
            alt = REPORTS_DIR / f"attendance_{today}_{int(time.time())}.csv"
            log.warning("Could not write %s (in use). Writing to %s instead", output_path, alt)
            self.today_report().to_csv(alt, index=False)
            out = alt
        log.info("Attendance exported -> %s", out)
        return out


# ══════════════════════════════════════════════════════════════════════════════
#  Encoding helpers
# ══════════════════════════════════════════════════════════════════════════════
class EncodingManager:
    """Loads, builds, and caches face encodings for all enrolled students."""

    def __init__(self, students_dir: Path = STUDENTS_DIR, cache: Path = ENCODINGS_FILE):
        self.students_dir = students_dir
        self.cache        = cache

    # ── public API ────────────────────────────────────────────────────────────
    def load(self, force_rebuild: bool = False) -> tuple[list, list]:
        """Return (known_encodings, known_ids). Rebuild cache if needed."""
        if not force_rebuild and self.cache.exists():
            data = json.loads(self.cache.read_text())
            encodings = [np.array(e) for e in data["encodings"]]
            ids       = data["ids"]
            log.info("Loaded %d face encoding(s) from cache.", len(ids))
            return encodings, ids

        return self._build_and_cache()

    def _build_and_cache(self) -> tuple[list, list]:
        encodings, ids = [], []
        for student_folder in sorted(self.students_dir.iterdir()):
            if not student_folder.is_dir():
                continue
            sid = student_folder.name          # folder name = student_id
            enc_list = self._encode_folder(student_folder, sid)
            encodings.extend(enc_list)
            ids.extend([sid] * len(enc_list))

        # persist
        self.cache.write_text(
            json.dumps({"encodings": [e.tolist() for e in encodings], "ids": ids})
        )
        log.info("Built encodings for %d image(s) across %d student(s).",
                 len(ids), len(set(ids)))
        return encodings, ids

    @staticmethod
    def _encode_folder(folder: Path, sid: str) -> list:
        results = []
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for img_path in folder.iterdir():
            if img_path.suffix.lower() not in image_extensions:
                continue
            img = face_recognition.load_image_file(str(img_path))
            encs = face_recognition.face_encodings(img)
            if encs:
                results.append(encs[0])
                log.debug("encoded %s for student %s", img_path.name, sid)
            else:
                log.warning("no face found in %s (student %s)", img_path.name, sid)
        return results


# ══════════════════════════════════════════════════════════════════════════════
#  Main recogniser / attendance loop
# ══════════════════════════════════════════════════════════════════════════════
class AttendanceSystem:
    """
    Real-time multi-student face recognition attendance system.

    Parameters
    ----------
    tolerance : float
        Face-match distance threshold (lower = stricter). Default 0.5.
    scale_factor : float
        Resize frame before recognition for speed (0.25 – 1.0). Default 0.25.
    cooldown_seconds : int
        Min seconds between successive marking for the same student. Default 5.
    model : str
        face_recognition detection model: 'hog' (CPU) or 'cnn' (GPU). Default 'hog'.
    """

    def __init__(
        self,
        tolerance: float        = 0.45,
        scale_factor: float     = 0.25,
        cooldown_seconds: int   = 5,
        model: str              = "hog",
    ):
        self.tolerance       = tolerance
        self.scale_factor    = scale_factor
        self.cooldown        = cooldown_seconds
        self.model           = model

        self.db              = AttendanceDB()
        enc_mgr              = EncodingManager()
        self.known_encodings, self.known_ids = enc_mgr.load()

        # Map student_id → name from DB
        with self.db._conn() as con:
            rows = con.execute("SELECT student_id, name FROM students").fetchall()
        self.id_to_name = {r[0]: r[1] for r in rows}

        # Cooldown tracker: student_id → last_marked timestamp
        self._last_seen: dict[str, float] = {}

        # Per-session stats
        self.session_marked: set[str] = set()

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _is_image_file(source) -> bool:
        """Return True when source is a path to a static image file."""
        IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
        return isinstance(source, str) and Path(source).suffix.lower() in IMAGE_EXTS

    # ── core loop ─────────────────────────────────────────────────────────────
    def run(self, source: int | str = 0, window_title: str = "Attendance System"):
        """
        Start recognition from a webcam, video file, or static image.

        Parameters
        ----------
        source : int or str
            • int        → webcam index (0, 1, …)
            • str (image) → path to .jpg / .jpeg / .png / .bmp / .webp
            • str (video) → path to .mp4 / .avi / .mov / etc.
        window_title : str
            Title of the OpenCV display window.
        """
        if not self.known_encodings:
            log.error("No face encodings found. Enrol students first!")
            return

        # ── Route to image mode or video mode ────────────────────────────────
        if self._is_image_file(source):
            self._run_image(source, window_title)
        else:
            self._run_video(source, window_title)

    # ── static image mode ─────────────────────────────────────────────────────
    def _run_image(self, image_path: str, window_title: str):
        """
        Process a single static image for attendance.
        Shows the annotated result and waits for any key press to exit.
        """
        log.info("Image mode -> loading %s", image_path)

        frame = cv2.imread(image_path)
        if frame is None:
            log.error("Could not read image file: %s", image_path)
            log.error("Check the path exists and is a valid image.")
            return

        h, w = frame.shape[:2]
        log.info("Image size: %d × %d px", w, h)

        # Run recognition on the single frame (no scale-skip in image mode)
        faces = self._process_frame(frame)
        self._draw_ui(frame, time.time(), faces)

        # Print & export results immediately
        self._print_report()
        exported = self.db.export_csv()
        log.info("CSV exported -> %s", exported)

        # ── Display annotated image ──────────────────────────────────────────
        # Resize for display if image is very large (keep aspect ratio)
        max_display = 1280
        if w > max_display:
            scale        = max_display / w
            display_frame = cv2.resize(frame, (max_display, int(h * scale)))
        else:
            display_frame = frame

        cv2.imshow(f"{window_title}  |  Press any key to close", display_frame)
        log.info("Displaying result. Press any key in the image window to close.")
        cv2.waitKey(0)          # Wait indefinitely until user presses a key
        cv2.destroyAllWindows()

        # Also save the annotated image next to the source
        out_path = Path(image_path).with_stem(Path(image_path).stem + "_result")
        cv2.imwrite(str(out_path), frame)
        log.info("Annotated image saved -> %s", out_path)

    # ── video / webcam mode ───────────────────────────────────────────────────
    def _run_video(self, source: int | str, window_title: str):
        """Live recognition loop for webcam or video file."""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            log.error("Cannot open video source: %s", source)
            log.error(
                "Tip: for a static image use --source with a .jpg/.png path; "
                "for a webcam use an integer (0, 1, …)."
            )
            return

        log.info("Recognition started. Press [Q] to quit, [R] for report, [E] to export.")

        frame_idx = 0
        fps_time  = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                log.info("End of stream.")
                break

            frame_idx += 1

            # ── Process every other frame for speed ───────────────────────────
            faces = []
            if frame_idx % 2 == 0:
                faces = self._process_frame(frame)

            # ── Overlay UI ───────────────────────────────────────────────────
            self._draw_ui(frame, fps_time, faces)
            fps_time = time.time()

            cv2.imshow(window_title, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                self._print_report()
            elif key == ord("e"):
                path = self.db.export_csv()
                log.info("Exported → %s", path)

        cap.release()
        cv2.destroyAllWindows()
        self._print_report()
        self.db.export_csv()

    # ── frame-level logic ─────────────────────────────────────────────────────
    def _process_frame(self, frame: np.ndarray):
        """Detect all faces in frame, match against known encodings, mark attendance.

        Returns a list of face tuples: (top, right, bottom, left, name, sid, confidence)
        """
        small = cv2.resize(frame, (0, 0), fx=self.scale_factor, fy=self.scale_factor)
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb, model=self.model)
        face_encodings = face_recognition.face_encodings(rgb, face_locations)

        results = []
        inv = 1.0 / self.scale_factor

        for enc, loc in zip(face_encodings, face_locations):
            sid, name, confidence = self._match(enc)
            self._maybe_mark(sid)

            # Scale location back to original frame size
            top, right, bottom, left = [int(v * inv) for v in loc]
            results.append((top, right, bottom, left, name, sid, confidence))

        return results

    def _match(self, encoding: np.ndarray) -> tuple[str, str, float]:
        if not self.known_encodings:
            return "unknown", "Unknown", 0.0
        distances = face_recognition.face_distance(self.known_encodings, encoding)
        idx       = int(np.argmin(distances))
        dist      = float(distances[idx])
        if dist <= self.tolerance:
            sid  = self.known_ids[idx]
            name = self.id_to_name.get(sid, sid)
            return sid, name, round((1 - dist) * 100, 1)
        return "unknown", "Unknown", 0.0

    def _maybe_mark(self, sid: str):
        if sid == "unknown":
            return
        now = time.time()
        if now - self._last_seen.get(sid, 0) < self.cooldown:
            return
        if self.db.mark_attendance(sid):
            log.info("Attendance marked -> %s (%s)", self.id_to_name.get(sid, sid), sid)
            self.session_marked.add(sid)
        self._last_seen[sid] = now

    # ── drawing ───────────────────────────────────────────────────────────────
    def _draw_ui(self, frame: np.ndarray, fps_start: float, faces: list):

        for top, right, bottom, left, name, sid, conf in faces:
            known    = sid != "unknown"
            colour   = (0, 220, 0) if known else (0, 0, 220)
            marked   = sid in self.session_marked

            # Box
            cv2.rectangle(frame, (left, top), (right, bottom), colour, 2)

            # Label background
            label    = f"{name}  {conf:.1f}%" if known else "Unknown"
            lh       = 22
            cv2.rectangle(frame, (left, bottom - lh - 4), (right, bottom), colour, cv2.FILLED)
            cv2.putText(frame, label, (left + 4, bottom - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            # ✔ badge
            if marked:
                cv2.putText(frame, "✔", (left + 4, top - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)

        # HUD
        now_str  = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        fps      = 1.0 / max(time.time() - fps_start, 1e-6)
        mode_lbl = "IMAGE MODE" if fps > 500 else f"FPS {fps:.1f}"
        cv2.putText(frame, f"{mode_lbl}  |  {now_str}", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        cv2.putText(frame, f"Marked today: {len(self.session_marked)}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        cv2.putText(frame, "Q=Quit  R=Report  E=Export", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # ── report ────────────────────────────────────────────────────────────────
    def _print_report(self):
        df = self.db.today_report()
        print("\n" + "=" * 60)
        print(f"  ATTENDANCE REPORT  —  {date.today().isoformat()}")
        print("=" * 60)
        print(df.to_string(index=False))
        present = (df["status"] == "Present").sum()
        total   = len(df)
        print(f"\n  Present: {present}/{total}  ({present/max(total,1)*100:.1f}%)")
        print("=" * 60 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Face Recognition Attendance System")
    parser.add_argument("--source",    default=0,     help="Webcam index or video file path")
    parser.add_argument("--tolerance", type=float, default=0.5,  help="Match tolerance (0–1)")
    parser.add_argument("--scale",     type=float, default=0.25, help="Frame scale for speed")
    parser.add_argument("--model",     default="hog", choices=["hog", "cnn"],
                        help="Detection model: hog (CPU) or cnn (GPU/dlib)")
    parser.add_argument("--rebuild",   action="store_true", help="Force rebuild face encodings")
    args = parser.parse_args()

    # Rebuild encodings if requested
    if args.rebuild:
        EncodingManager().load(force_rebuild=True)

    source = int(args.source) if str(args.source).isdigit() else args.source

    system = AttendanceSystem(
        tolerance     = args.tolerance,
        scale_factor  = args.scale,
        model         = args.model,
    )
    system.run(source=source)