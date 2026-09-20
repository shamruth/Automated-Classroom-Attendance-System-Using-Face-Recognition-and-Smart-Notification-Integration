# Attendance API

This is an HTTP adapter around the existing face-recognition attendance system. It does not replace or modify the webcam, enrollment, database, or report code.

## Start the API

```powershell
pip install -r api_requirements.txt
uvicorn api:app --reload
```

The server listens on `http://127.0.0.1:8000` by default. The frontend may use `http://localhost:8000` as its API base URL.

## Process an uploaded picture

`POST /attendance/process` accepts a multipart form upload named `image`.

```javascript
const form = new FormData();
form.append("image", fileFromInput);

const response = await fetch("http://127.0.0.1:8000/attendance/process", {
  method: "POST",
  body: form,
});
const result = await response.json();
```

Optional multipart fields are `tolerance`, `scale`, and `model` (`hog` or `cnn`). The response includes recognized faces, today's attendance rows, a present/total summary, and the generated CSV filename.

## Reports

- `GET /attendance/report` returns today's report.
- `GET /attendance/report?date=2026-09-20` returns one date.
- `GET /attendance/report?start=2026-09-01&end=2026-09-20` returns a date range.
- `GET /attendance/report.csv?date=2026-09-20` downloads a CSV report.
- `GET /health` checks that the API is running.

Set `API_CORS_ORIGINS` to a comma-separated list of frontend origins in production. The default is `*` for local development.