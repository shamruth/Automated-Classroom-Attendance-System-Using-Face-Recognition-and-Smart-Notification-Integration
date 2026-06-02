"""
batch_enroll.py
───────────────
Bulk-enroll many students at once from an existing folder.

Expected folder layout
──────────────────────
sample_images/
    S001_Alice_Johnson/
        photo1.jpg
        photo2.jpg
    S002_Bob_Smith/
        img_a.png
        img_b.png
    ...

The folder name format is:  <student_id>_<FirstName>_<LastName>
(extra words after the first underscore-separated token are joined as the name)

Usage
─────
python batch_enroll.py --dir sample_images/
python batch_enroll.py --dir sample_images/ --no-rebuild   # skip encoding rebuild
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # Ensure parent folder is on sys.path so `face_attendance` can be imported
from face_attendance.face_attendance import AttendanceDB, EncodingManager, STUDENTS_DIR
from face_attendance.enroll_student import copy_images, verify_faces

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_folder_name(folder_name: str) -> tuple[str, str]:
    """
    'S001_Alice_Johnson' → ('S001', 'Alice Johnson')
    Falls back to using the whole name as both id and name.
    """
    parts = folder_name.split("_", 1)
    if len(parts) == 2:
        student_id = parts[0]
        name       = parts[1].replace("_", " ")
    else:
        student_id = folder_name
        name       = folder_name
    return student_id, name


def main():
    parser = argparse.ArgumentParser(description="Batch-enroll students from a folder tree")
    parser.add_argument("--dir", required=True, help="Root folder containing student sub-folders")
    parser.add_argument("--no-rebuild", action="store_true",
                        help="Skip rebuilding face encodings after enrollment")
    args = parser.parse_args()

    root = Path(args.dir)
    if not root.exists():
        log.error("Directory not found: %s", root)
        sys.exit(1)

    db = AttendanceDB()
    enrolled = 0

    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue

        student_id, name = parse_folder_name(folder.name)
        images = [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]

        if not images:
            log.warning("No images in %s — skipping.", folder.name)
            continue

        log.info("Enrolling  %-10s  %s  (%d image(s))", student_id, name, len(images))
        db.register_student(student_id, name)
        saved = copy_images(student_id, [str(p) for p in images])
        good  = verify_faces(saved)

        if good == 0:
            log.warning("  ✘ No detectable faces in any image for %s", name)
        else:
            log.info("  ✔ %d/%d faces verified for %s", good, len(saved), name)
            enrolled += 1

    log.info("Enrolled %d student(s) total.", enrolled)

    if not args.no_rebuild:
        log.info("Rebuilding encoding cache …")
        EncodingManager().load(force_rebuild=True)
        log.info("Encoding cache updated.")


if __name__ == "__main__":
    main()
