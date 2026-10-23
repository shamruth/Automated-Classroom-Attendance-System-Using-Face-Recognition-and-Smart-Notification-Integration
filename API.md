# Attendance API

This is an HTTP adapter around the existing face-recognition attendance system. It does not replace or modify the webcam, enrollment, database, or report code.

## Start the API

```powershell
pip install -r api_requirements.txt
uvicorn api:app --reload
```

The server listens on `http://127.0.0.1:8000` by default. The frontend may use `http://localhost:8000` as its API base URL.

## Process an uploaded picture

`POST /attendance/process` accepts one or more multipart form uploads named `image`.

```javascript
const form = new FormData();
files.forEach((file) => form.append("image", file));

const response = await fetch("http://127.0.0.1:8000/attendance/process", {
  method: "POST",
  body: form,
});
const result = await response.json();
```

Optional multipart fields are `tolerance`, `scale`, and `model` (`hog` or `cnn`). The response includes today's attendance rows, a present/total summary, the generated CSV filename, and `results`, containing one item per uploaded image. Each result has its filename, recognized faces, and `result_image`, a base64-encoded JPEG data URL containing the recognition overlays.

## Reports

- `GET /attendance/report` returns today's report.
- `GET /attendance/report?date=2026-09-20` returns one date.
- `GET /attendance/report?start=2026-09-01&end=2026-09-20` returns a date range.
- `GET /attendance/report.csv?date=2026-09-20` downloads a CSV report.
- `GET /health` checks that the API is running.

Set `API_CORS_ORIGINS` to a comma-separated list of frontend origins in production. The default is `*` for local development.