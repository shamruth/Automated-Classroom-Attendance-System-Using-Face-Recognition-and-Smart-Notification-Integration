# Face Attendance System (FastAPI + InsightFace)

## Setup (Python 3.10/3.11)
    python -m venv venv && venv\Scripts\activate
    pip install -r requirements.txt
    pip install opencv-python numpy scipy insightface python-dotenv fastapi uvicorn pydantic python-multipart dotenv

## Running Code   
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
## UI ctrl+click
Open http://localhost:8000  (phones on same Wi-Fi: http://<pc-ip>:8000; camera needs https or localhost, otherwise use Upload)

## Data
- `students/<ROLLNO>/<NAME>.jpg` - 1+ clear front-face photos per student (more photos = better accuracy)
- `contacts.csv` - roll,name,email,parent_email
- `.env` - copy from `.env.example`; with no SMTP_HOST, emails are saved as .eml in data/outbox (safe demo mode)

## API
POST /api/attendance (file, subject, period) | GET /api/sessions/{id} | POST /api/sessions/{id}/notify | POST /api/sessions/{id}/resend {rolls?}
POST /api/sessions/{id}/override | POST /api/roster/reload | docs at /docs

## Logs (logs/)
enrollment, capture, detection, recognition, attendance, notification, api - one file per stage.

## Tuning
MATCH_THRESHOLD (default 0.38): raise to cut false matches, lower if known students come out "Unknown".
