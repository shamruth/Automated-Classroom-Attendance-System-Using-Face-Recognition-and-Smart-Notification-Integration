# 🎓 Face Recognition Attendance System

Multi-student real-time attendance using face recognition.

---

## 📁 Project Structure

```
face_attendance/
├── face_attendance.py   ← Main recognition loop (run this)
├── enroll_student.py    ← Add one student (webcam or images)
├── batch_enroll.py      ← Bulk add many students from a folder
├── report.py            ← Generate / export attendance reports
├── requirements.txt
│
├── students/            ← Auto-created; one sub-folder per student
│   ├── S001/
│   │   ├── sample_000.jpg
│   │   └── sample_001.jpg
│   └── S002/
│       └── sample_000.jpg
│
├── encodings.json       ← Auto-built cache of face encodings
├── attendance.db        ← SQLite database
└── reports/             ← Exported CSV files
```

---

## 🐍 Create and Activate Virtual Environment

It is recommended to use a virtual environment to avoid dependency conflicts.

### Windows (PowerShell)

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### Windows (Command Prompt)

```cmd
:: Create virtual environment
python -m venv venv

:: Activate virtual environment
venv\Scripts\activate.bat

:: Upgrade pip
python -m pip install --upgrade pip

:: Install dependencies
pip install -r requirements.txt
```

### Linux / macOS

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
python --version
pip --version
```

### Deactivate Virtual Environment

```bash
deactivate
```

## ⚙️ Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# On Ubuntu/Debian you may need system packages first:
sudo apt-get install cmake libdlib-dev libopenblas-dev

# On macOS:
brew install cmake dlib
```

---

## 🚀 Quick Start

### Step 1 — Enroll students

**Option A: Capture photos from webcam**
```bash
python enroll_student.py --id S001 --name "Alice Johnson" --capture 5
python enroll_student.py --id S002 --name "Bob Smith"     --capture 5
```

**Option B: Use existing images**
```bash
python enroll_student.py --id S001 --name "Alice Johnson" \
       --images alice1.jpg alice2.jpg alice3.jpg
```

**Option C: Bulk enroll from a folder**

Organise your images like this:
```
sample_images/
    S001_Alice_Johnson/   ← folder name = ID_FirstName_LastName
        photo1.jpg
        photo2.jpg
    S002_Bob_Smith/
        bob.jpg
```

Then run:
```bash
python face_attendance/batch_enroll.py --dir sample_images/
```

---

### Step 2 — Start the attendance system

```bash
# Default: webcam 0, HOG model (CPU)
python face_attendance/face_attendance.py

# Custom webcam index
python face_attendance/face_attendance.py --source 1

# Use a video file
python face_attendance/face_attendance.py --source class_recording.mp4

# USE PHOTO
# Scale 0.5 = balances speed vs detecting background faces
python face_attendance/face_attendance.py --source .\test_images\test1.jpg --scale 0.5

# If still missing faces, try full resolution (slower but detects everyone)
python face_attendance/face_attendance.py --source ..\test_images\PPRO_TEST_LARGE_TEST.jpeg --scale 1.0

# Best accuracy overall (uses CNN model — needs more RAM, slower on CPU)
python face_attendance/face_attendance.py --source ..\test_images\PPRO_TEST_LARGE_TEST.jpeg --scale 1.0 --model cnn

# GPU mode (requires dlib built with CUDA)
python face_attendance/face_attendance.py --model cnn

# Stricter matching (lower = stricter)
python face_attendance.py --tolerance 0.45

# Force-rebuild face encodings
python face_attendance.py --rebuild
```

**Keyboard shortcuts while running:**
| Key | Action |
|-----|--------|
| `Q` | Quit and save report |
| `R` | Print today's report in terminal |
| `E` | Export today's attendance to CSV |

---

### Step 3 — Generate reports

```bash
# Print today's attendance
python report.py

# Specific date
python report.py --date 2025-08-10

# Date range summary
python report.py --range 2025-08-01 2025-08-31

# Export to CSV
python report.py --export
python report.py --date 2025-08-10 --export --out my_report.csv
```

---

## ⚙️ Configuration Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--tolerance` | `0.5` | Face match threshold. Lower = stricter (0.4–0.6 recommended) |
| `--scale` | `0.25` | Frame downscale before recognition. Higher = more accurate but slower |
| `--model` | `hog` | `hog` = CPU (fast), `cnn` = GPU (more accurate) |
| `--source` | `0` | Webcam index or video file path |
| `--rebuild` | False | Force rebuild encoding cache |

---

## 💡 Tips for Best Accuracy

1. **More sample images = better accuracy** — 5–10 varied photos per student is ideal
2. **Vary conditions** — different angles, lighting, with/without glasses
3. **Good lighting** — face should be well-lit and not backlit
4. **Image resolution** — at least 200×200 px for the face region
5. **Tolerance tuning** — if false positives, lower to 0.45; if misses, raise to 0.55

---

## 🗃️ Database Schema

```sql
-- students table
student_id  TEXT  (e.g. "S001")
name        TEXT  (e.g. "Alice Johnson")
created_at  TEXT

-- attendance table
student_id  TEXT
date        TEXT  (YYYY-MM-DD)
time_in     TEXT  (HH:MM:SS)
status      TEXT  ("Present")
```

---

## 📊 Output CSV format

```
student_id, name, time_in, status
S001, Alice Johnson, 09:03:12, Present
S002, Bob Smith,    , Absent
```
## 📊 EVALUATION
pip install scikit-learn matplotlib seaborn
python face_attendance/evaluation.py
python face_attendance/evaluation.py --test-dir test_images/
python face_attendance/evaluation.py --test-dir test_images/ --sweep
