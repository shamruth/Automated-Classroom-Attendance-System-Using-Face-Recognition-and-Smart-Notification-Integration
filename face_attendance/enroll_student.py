"""
enroll_student.py
─────────────────
Register a new student by:
  1. adding them to the SQLite database, and
  2. saving their sample images under  students/<student_id>/

Usage examples
──────────────
# Capture 5 photos from webcam
python enroll_student.py --id S001 --name "Alice Johnson" --capture 5

# Use existing image files
python enroll_student.py --id S002 --name "Bob Smith" --images bob1.jpg bob2.jpg

# After enrolling all students, rebuild the encoding cache
python enroll_student.py --rebuild-only
"""

import cv2
import face_recognition
import shutil
import argparse
import logging
from pathlib import Path

# Import shared components from the main module
import sys
# Ensure parent folder is on sys.path so `face_attendance` can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))
from face_attendance.face_attendance import (
    AttendanceDB, EncodingManager,
    STUDENTS_DIR, log,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ── webcam capture helper ─────────────────────────────────────────────────────
def capture_from_webcam(student_id: str, name: str, count: int = 5) -> list[Path]:
    """Capture `count` face photos from the webcam and save them."""
    save_dir = STUDENTS_DIR / student_id
    save_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        log.error("Cannot open webcam.")
        return []

    print(f"\n  Capturing {count} photos for {name} ({student_id})")
    print("  Press [SPACE] to capture  |  [Q] to quit early\n")

    saved, idx = [], 0
    while idx < count:
        ret, frame = cap.read()
        if not ret:
            break

        # Live preview with guidance
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locs = face_recognition.face_locations(rgb, model="hog")

        colour = (0, 220, 0) if locs else (0, 0, 220)
        status = f"Face detected  [{idx}/{count} saved]" if locs else "No face detected — adjust position"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
        cv2.putText(frame, "SPACE=Capture  Q=Quit", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        for (top, right, bottom, left) in locs:
            cv2.rectangle(frame, (left, top), (right, bottom), colour, 2)

        cv2.imshow(f"Enroll — {name}", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord(" ") and locs:
            path = save_dir / f"sample_{idx:03d}.jpg"
            cv2.imwrite(str(path), frame)
            saved.append(path)
            idx += 1
            print(f"  ✔ Saved {path.name}")

    cap.release()
    cv2.destroyAllWindows()
    return saved


# ── image-file helper ─────────────────────────────────────────────────────────
def copy_images(student_id: str, image_paths: list[str]) -> list[Path]:
    """Copy existing image files into the student folder."""
    save_dir = STUDENTS_DIR / student_id
    save_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for i, src in enumerate(image_paths):
        src_path = Path(src)
        if not src_path.exists():
            log.warning("File not found: %s", src)
            continue
        dst = save_dir / f"sample_{i:03d}{src_path.suffix.lower()}"
        shutil.copy2(src_path, dst)
        saved.append(dst)
        print(f"  ✔ Copied {src_path.name} → {dst}")
    return saved


# ── verify encodings ──────────────────────────────────────────────────────────
def verify_faces(image_paths: list[Path]) -> int:
    good = 0
    for p in image_paths:
        img  = face_recognition.load_image_file(str(p))
        encs = face_recognition.face_encodings(img)
        if encs:
            good += 1
        else:
            log.warning("  ✘ No face detected in %s — consider replacing it", p.name)
    return good


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Enroll a student into the attendance system")
    parser.add_argument("--id",      help="Unique student ID (e.g. S001)")
    parser.add_argument("--name",    help='Student full name (e.g. "Alice Johnson")')
    parser.add_argument("--capture", type=int, default=0,
                        help="Number of webcam photos to capture (0 = use --images)")
    parser.add_argument("--images",  nargs="+", default=[],
                        help="Paths to existing image files")
    parser.add_argument("--rebuild-only", action="store_true",
                        help="Skip enrollment, just rebuild all face encodings")
    args = parser.parse_args()

    if args.rebuild_only:
        log.info("Rebuilding encoding cache for all students …")
        EncodingManager().load(force_rebuild=True)
        log.info("Done.")
        return

    if not args.id or not args.name:
        parser.error("--id and --name are required unless using --rebuild-only")

    db = AttendanceDB()
    db.register_student(args.id, args.name)
    log.info("Student registered in DB: %s — %s", args.id, args.name)

    if args.capture > 0:
        saved = capture_from_webcam(args.id, args.name, args.capture)
    elif args.images:
        saved = copy_images(args.id, args.images)
    else:
        parser.error("Provide --capture N or --images path1 path2 …")
        return

    if not saved:
        log.error("No images saved — enrollment aborted.")
        return

    good = verify_faces(saved)
    log.info("%d/%d images contain a detectable face.", good, len(saved))

    if good == 0:
        log.error("No usable face images. Enrollment incomplete.")
        return

    log.info("Rebuilding face encoding cache …")
    EncodingManager().load(force_rebuild=True)
    log.info("✅ Enrollment complete for %s (%s).", args.name, args.id)


if __name__ == "__main__":
    main()
