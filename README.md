# Automated Classroom Attendance System

An end-to-end classroom attendance application that combines multi-face recognition, a
SQLite attendance database, a command-line/OpenCV workflow, a FastAPI HTTP API, and a
browser-based UI. The system can recognize several enrolled students in a webcam feed,
video, static image, or uploaded classroom photograph and record each student once per
day.

> **Important:** This is a local biometric-processing project. Use it only with appropriate
> consent and authorization, protect the student images and database, and validate the
> recognition threshold for your environment before relying on the results.

## What the project provides

- **Enrollment:** capture samples from a webcam, copy existing images, or bulk-enroll
  students from a folder tree.
- **Hybrid recognition:** face-recognition HOG/CNN detection with an optional OpenCV Haar
  Cascade complement for classroom and lower-contrast images.
- **Attendance persistence:** registered students and daily attendance are stored in SQLite.
  A unique `(student_id, date)` constraint prevents duplicate daily entries.
- **Multiple input modes:** webcam, video file, single image, or multiple uploaded images.
- **Reports:** terminal reports, date-range reports, CSV exports, JSON API responses, and a
  browser table with present/total/percentage metrics.
- **Evaluation:** accuracy, macro precision/recall/F1, FAR, FRR, approximate/exact EER
  analysis where applicable, confusion matrices, timing/FPS data, charts, and tolerance
  sweeps.

## Architecture

```text
Student images ──┐
                 ├─> EncodingManager ──> encodings.json cache
Student records ─┘          │
                            v
Webcam / video / image ─> AttendanceSystem
                            │
                            ├─> face detection and encoding
                            ├─> nearest known-face match
                            ├─> attendance mark (once per student/day)
                            └─> annotated output and report

OpenCV CLI ───────────────────────────────┐
FastAPI API ── api.py ────────────────────┼─> shared face_attendance core
Browser UI ── api_ui.html ─> HTTP API ────┘
```

The API and UI do not implement a separate recognition engine. `api.py` creates the
same `AttendanceSystem` used by the desktop workflow, while `api_ui.html` uploads images,
displays annotated results, and refreshes reports through the API.

## Repository layout

```text
.
├── api.py                       FastAPI application
├── api_requirements.txt         API-only dependencies
├── api_ui.html                  Standalone browser frontend
├── API.md                      Short API reference
├── UI_API_README.md             Earlier API/UI setup notes
├── report.py                   Terminal report and CSV command
├── requirements.txt             Main recognition and analytics dependencies
├── face_attendance/
│   ├── __init__.py
│   ├── face_attendance.py       Core DB, detection, matching, and attendance loop
│   ├── enroll_student.py        Single-student enrollment and cache rebuild
│   ├── batch_enroll.py          Folder-based bulk enrollment
│   ├── evaluation.py            Metrics, plots, group mode, and tolerance sweep
│   └── students/                One image folder per enrolled student
├── sample_images/               Example bulk-enrollment input
├── test_images/                 Recognition/evaluation images
├── test_images_row1/            Additional classroom test set
├── attendance.db                Runtime SQLite database when generated
├── encodings.json               Runtime face-encoding cache when generated
├── reports/                     Runtime CSV output directory
└── eval_results*/               Runtime evaluation output directories
```

`venv/`, `.venv/`, Python caches, logs, databases, encoding caches, reports, and
evaluation outputs are local runtime artifacts. They should not be copied between
machines as an installation method. The `.gitignore` excludes the generated categories;
review any real student images before committing them.

## Requirements

- Python 3.10 or newer is recommended.
- A webcam is required only for webcam enrollment or live attendance.
- Windows users may need a PowerShell execution-policy adjustment for virtual
  environment activation.
- `dlib`/`face_recognition` installation can require platform-specific build support.
  On Ubuntu/Debian, install the system packages required by your Python/dlib setup
  (commonly `cmake`, `libdlib-dev`, and `libopenblas-dev`). On macOS, install CMake
  and the dlib prerequisites with Homebrew if pip cannot build them.
- CNN detection is more accurate in some cases but is slower and requires a dlib build
  with suitable CPU/GPU support. HOG is the practical default.

## Installation

Run these commands from the repository root.

### Windows PowerShell

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r api_requirements.txt
```

If PowerShell blocks activation, allow it only for the current shell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
```

### Windows Command Prompt

```cmd
python -m venv venv
venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r api_requirements.txt
```

### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r api_requirements.txt
```

The two requirements files are intentionally separate: `requirements.txt` contains the
recognition, image-processing, data, and evaluation stack; `api_requirements.txt`
contains FastAPI, Uvicorn, and multipart upload support. Installing both enables every
workflow documented here.

Verify the environment:

```powershell
python --version
python -m pip --version
```

Deactivate it when finished:

```powershell
deactivate
```

## First use: enroll students

Students must be registered and have usable face samples before recognition can mark
attendance. Each student is stored under `face_attendance/students/<student_id>/`.
Enrollment also registers the student in SQLite and rebuilds the encoding cache.

### Capture images from a webcam

```powershell
python face_attendance\enroll_student.py --id S001 --name "Alice Johnson" --capture 5
```

During capture, press **Space** to save a detected face and **Q** to stop early.

### Use existing images

```powershell
python face_attendance\enroll_student.py `
  --id S001 `
  --name "Alice Johnson" `
  --images alice1.jpg alice2.jpg alice3.jpg
```

The script verifies that each copied image contains a detectable face. Use several
well-lit images with different angles and expressions; five to ten varied samples per
student is a useful starting point.

### Bulk enrollment

Use one directory per student. The folder name is parsed as
`<student_id>_<name_words>`:

```text
sample_images/
├── S001_Alice_Johnson/
│   ├── photo1.jpg
│   └── photo2.jpg
└── S002_Bob_Smith/
    └── photo1.jpg
```

```powershell
python face_attendance\batch_enroll.py --dir sample_images
```

Add `--no-rebuild` when enrolling several batches and rebuild once at the end:

```powershell
python face_attendance\batch_enroll.py --dir sample_images --no-rebuild
python face_attendance\enroll_student.py --rebuild-only
```

To explicitly rebuild the cache at any time:

```powershell
python face_attendance\face_attendance.py --rebuild
```

## Option A: run the OpenCV desktop workflow

Start with the default webcam (`0`) and the CPU HOG detector:

```powershell
python face_attendance\face_attendance.py
```

Other sources and settings:

```powershell
# Another webcam
python face_attendance\face_attendance.py --source 1

# Video file
python face_attendance\face_attendance.py --source classroom.mp4

# Static image; full resolution is recommended for distant faces
python face_attendance\face_attendance.py --source test_images\row1.jpg --scale 1.0

# More accurate/slower CNN detector
python face_attendance\face_attendance.py --model cnn

# Stricter matching threshold
python face_attendance\face_attendance.py --tolerance 0.45

# More upsampling for small faces
python face_attendance\face_attendance.py --upsample 2

# Disable the Haar complement
python face_attendance\face_attendance.py --no-haar
```

### Desktop keyboard controls

| Key | Action |
|---|---|
| `Q` | Stop the live/video session and save a report |
| `R` | Print today's report |
| `E` | Export today's attendance to CSV |

Static-image mode displays the annotated image, exports attendance, and writes a
`*_result` image beside the source image.

## Option B: run the combined API and browser UI

The API and UI are separate local processes:

1. Uvicorn serves recognition and report endpoints on port `8000`.
2. Python's static server serves `api_ui.html` on port `5500`.
3. The UI calls `http://127.0.0.1:8000`, uploads one or more images, and renders the
   JSON response and base64-encoded annotated images.

### Start the API

```powershell
python -m uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

Check that it is running:

```powershell
curl.exe http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"ok"}
```

Interactive OpenAPI documentation is available at
`http://127.0.0.1:8000/docs`.

### Serve and open the UI

In a second PowerShell window:

```powershell
.\venv\Scripts\Activate.ps1
python -m http.server 5500
```

Open `http://127.0.0.1:5500/api_ui.html`.

The UI supports:

- API connection status and error messages
- one or more classroom image uploads
- immediate previews of loaded files
- scale selection (`100%`, `75%`, or `50%`)
- HOG or hybrid HOG + Haar detection
- upsample level `0`, `1`, or `2`
- match-tolerance adjustment
- annotated result gallery for every uploaded image
- recognized/unknown face counts and confidence values
- today's attendance table and present/total/percentage summary
- report refresh without restarting either server

Stop both servers with `Ctrl+C` in their respective terminals.

### Process an image without the UI

PowerShell's `curl.exe` can call the multipart endpoint directly:

```powershell
curl.exe -X POST `
  -F "image=@test_images\row1.jpg" `
  -F "tolerance=0.50" `
  -F "scale=1.0" `
  -F "model=hog" `
  -F "upsample=1" `
  -F "use_haar=true" `
  http://127.0.0.1:8000/attendance/process
```

The response contains the processing date, the first result for backward compatibility,
`results` for every uploaded image, face locations, recognized IDs/names/confidence,
annotated JPEG data URLs, attendance rows, summary metrics, and the generated CSV
filename. The API accepts at most 25 images per request and limits each image to 15 MB.
Supported types are JPEG, PNG, WEBP, BMP, and TIFF.

## HTTP API reference

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Return `{"status":"ok"}` |
| `POST` | `/attendance/process` | Process one or more `image` multipart files and mark attendance |
| `GET` | `/attendance/report` | Today's JSON report |
| `GET` | `/attendance/report?date=YYYY-MM-DD` | JSON report for one date |
| `GET` | `/attendance/report?start=YYYY-MM-DD&end=YYYY-MM-DD` | Inclusive date-range JSON report |
| `GET` | `/attendance/report.csv` | Download today's CSV report |
| `GET` | `/attendance/report.csv?date=YYYY-MM-DD` | Download one date as CSV |
| `GET` | `/attendance/report.csv?start=YYYY-MM-DD&end=YYYY-MM-DD` | Download a date range as CSV |
| `GET` | `/docs` | Swagger UI generated by FastAPI |

`/attendance/process` accepts optional multipart fields:

| Field | Default | Valid values |
|---|---:|---|
| `tolerance` | `0.50` | greater than `0` and at most `1`; lower is stricter |
| `scale` | `1.0` | greater than `0` and at most `1` |
| `model` | `hog` | `hog` or `cnn` |
| `upsample` | `1` | `0`, `1`, or `2` |
| `use_haar` | `true` | boolean |

Configure CORS for a deployed frontend with a comma-separated environment variable:

```powershell
$env:API_CORS_ORIGINS = "http://127.0.0.1:5500,https://your-frontend.example"
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

The development default is `*`. Do not use a permissive CORS policy or an exposed
development server as-is for a public deployment.

## Reports and persistence

### Terminal reports

```powershell
# Today
python report.py

# One date
python report.py --date 2026-10-26

# Inclusive date range
python report.py --range 2026-10-01 2026-10-31

# Export today's report
python report.py --export

# Export a selected date to a chosen path
python report.py --date 2026-10-26 --export --out reports\class.csv
```

### SQLite schema

The core module creates the database automatically at
`face_attendance/attendance.db`:

```sql
students (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  student_id TEXT UNIQUE NOT NULL,
  name TEXT NOT NULL,
  created_at TEXT
)

attendance (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  student_id TEXT NOT NULL,
  date TEXT NOT NULL,
  time_in TEXT NOT NULL,
  status TEXT DEFAULT 'Present',
  UNIQUE(student_id, date)
)
```

Every report starts with the registered student roster, then left-joins attendance for
the requested date; students without a row are reported as `Absent`. Attendance marking
is idempotent for a student/date, and the runtime cooldown also prevents repeated marks
while the same face remains in a live stream.

### Encoding cache

`face_attendance/encodings.json` stores serialized face vectors and their student IDs.
It is rebuilt from the image folders when enrollment completes, with
`enroll_student.py --rebuild-only`, or with `face_attendance.py --rebuild`. Rebuild it
after adding, replacing, or deleting student images.

## Evaluation

The evaluation script uses the encoding cache and a labelled test directory. Install
the analytics dependencies if they are not already present:

```powershell
python -m pip install scikit-learn matplotlib seaborn
```

### Labelled per-student evaluation

```text
test_images/
├── S001/
│   ├── image1.jpg
│   └── image2.jpg
├── S002/
│   └── image1.jpg
└── unknown/
    └── stranger.jpg
```

```powershell
python face_attendance\evaluation.py --test-dir test_images
python face_attendance\evaluation.py --test-dir test_images --tolerance 0.45 --model hog
python face_attendance\evaluation.py --test-dir test_images --out-dir eval_results
python face_attendance\evaluation.py --test-dir test_images --sweep
```

### Group-photo evaluation

For a flat directory of classroom/group photos, use group mode:

```powershell
python face_attendance\evaluation.py `
  --group-mode `
  --test-dir test_images_row1 `
  --out-dir eval_results_row1 `
  --upsample 1
```

Evaluation outputs include CSV summaries, per-image/per-face results, classification
metrics, confusion matrices, timing statistics, and charts. The tolerance sweep
compares accuracy, F1, FAR, and FRR over multiple thresholds and writes a
`tolerance_sweep.png` chart. Do not feed generated `_result` images back into the
evaluation dataset.

## Recognition tuning

- Start with `tolerance=0.50`; lower it toward `0.45` when false matches are a concern,
  and raise it cautiously when genuine faces are missed.
- Use `scale=1.0` for static classroom images when accuracy matters more than speed.
- Use `scale=0.50` for live video where performance matters, with the full-resolution
  fallback handling frames in which no faces are found.
- Use `upsample=1` for typical classroom images and `2` for small/distant faces; higher
  upsampling increases processing cost.
- Use HOG for normal CPU operation. Use CNN only when the installed dlib build supports
  it and the additional processing time is acceptable.
- Keep multiple samples per student and vary lighting, angle, distance, and glasses.
- Inspect unknown detections and evaluate on representative images before operational use.

## Troubleshooting

### No faces or no encodings

Confirm that the student folder contains supported image extensions (`.jpg`, `.jpeg`,
`.png`, `.bmp`, or `.webp`), that faces are visible, and that the cache has been rebuilt:

```powershell
python face_attendance\enroll_student.py --rebuild-only
```

Try a larger scale or upsample value, improve lighting, and replace samples with images
where the face is not too small or obstructed.

### API reports no enrolled encodings

The API returns HTTP `503` until `encodings.json` contains at least one usable face
encoding. Enroll at least one student and rebuild the cache before processing uploads.

### UI cannot connect

Make sure Uvicorn is still running on port `8000`, open the UI through
`http://127.0.0.1:5500/api_ui.html` rather than directly from `file://`, and verify:

```powershell
curl.exe http://127.0.0.1:8000/health
```

### Webcam or OpenCV window does not open

Try `--source 1` or another camera index, close applications currently using the camera,
and verify that the process is running in a desktop session with camera permission.
Use a file path as `--source` to isolate camera issues from recognition issues.

### Dependency installation fails

Upgrade pip, confirm the active interpreter is the virtual environment, and install the
platform's CMake/dlib prerequisites. The `face_recognition` stack is native-code
dependent and may need a compatible Python version or prebuilt wheel.

## Security, privacy, and deployment notes

- The application has no authentication or authorization layer. Keep the API bound to
  localhost for development and place authentication, TLS, rate limiting, and network
  controls in front of it before deployment.
- Uploaded images and base64-encoded annotated results contain biometric information.
  Use HTTPS in transit, restrict filesystem access, define retention rules, and avoid
  logging image contents or sensitive identifiers.
- Replace the development CORS wildcard with an explicit allowlist.
- Validate date input, report access, enrollment access, and operational audit needs
  before using the system in a real institution.
- Recognition confidence is a matching score, not proof of identity. Combine threshold
  tuning, representative evaluation, and human review for uncertain cases.

## Related documentation

- [API.md](API.md) contains a compact endpoint-focused API reference.
- [UI_API_README.md](UI_API_README.md) contains the original two-terminal UI/API walkthrough.
