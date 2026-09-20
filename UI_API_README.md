# Attendance API and UI

This project now includes:

- `api.py`: API for image-based face recognition, attendance marking, and reports
- `api_ui.html`: simple browser UI for checking the API and viewing attendance
- `api_requirements.txt`: dependencies required by the API

The existing face-recognition and attendance code is used without modification.

## 1. Open the project folder

Open PowerShell in the project folder:

```powershell
cd "C:\Users\Samruth\Downloads\MINI\Automated-Classroom-Attendance-System-Using-Face-Recognition-and-Smart-Notification-Integration"
```

## 2. Activate the virtual environment

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation script, run this once in the same PowerShell window:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
```

## 3. Install API dependencies

```powershell
python -m pip install -r api_requirements.txt
```

The API dependencies include FastAPI, Uvicorn, and multipart upload support.

## 4. Start the API

Keep this PowerShell window open:

```powershell
python -m uvicorn api:app --reload
```

The API runs at:

```text
http://127.0.0.1:8000
```

## 5. Start the UI server

Open a second PowerShell window, activate the environment if needed, and go to the project folder:

```powershell
cd "C:\Users\Samruth\Downloads\MINI\Automated-Classroom-Attendance-System-Using-Face-Recognition-and-Smart-Notification-Integration"
.\venv\Scripts\Activate.ps1
python -m http.server 5500
```

Keep this window open too.

## 6. Open the UI

Open this address in a browser:

```text
http://127.0.0.1:5500/api_ui.html
```

The UI shows:

- API connection status
- Today's attendance report
- Present, total, and attendance percentage summary
- Student attendance rows
- Image upload and face-processing results

Choose a classroom image and click **Process attendance**. The report updates after processing.

## 7. Check the API directly

Health check:

```text
http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"ok"}
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

Today's report:

```text
http://127.0.0.1:8000/attendance/report
```

## 8. Test image processing from PowerShell

Use an existing image from the project:

```powershell
curl.exe -X POST `
  -F "image=@test_images\row1.jpg" `
  http://127.0.0.1:8000/attendance/process
```

The response contains recognized faces, confidence values, attendance rows, a summary, and the generated CSV filename.

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Check that the API is running |
| `POST` | `/attendance/process` | Upload an image and mark attendance |
| `GET` | `/attendance/report` | Return today's report |
| `GET` | `/attendance/report?date=YYYY-MM-DD` | Return a report for one date |
| `GET` | `/attendance/report?start=YYYY-MM-DD&end=YYYY-MM-DD` | Return a date-range report |
| `GET` | `/attendance/report.csv` | Download today's report as CSV |

## Requirements before processing images

Students must already be enrolled. The API also needs the existing face encodings cache or student images so the recognition system can load known faces.

If the UI says the API is unavailable, confirm that the Uvicorn command is still running on port `8000`. If image processing returns that no encodings were found, enroll students and rebuild the encodings before trying again.

## Stop the servers

Press `Ctrl+C` in each PowerShell window running Uvicorn or the UI server.
