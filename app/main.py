import json, os, time, uuid, cv2, numpy as np
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
from .logger import get_logger
from .face_engine import FaceEngine, annotate
from . import notifier

api_log, cap_log, att_log = get_logger("api"), get_logger("capture"), get_logger("attendance")
SESS = ROOT / "data" / "sessions"; SESS.mkdir(parents=True, exist_ok=True)
engine = FaceEngine(ROOT / "students", float(os.getenv("MATCH_THRESHOLD", 0.38)))
app = FastAPI(title="Face Attendance API")

@app.middleware("http")
async def log_requests(req: Request, call_next):
    t = time.time(); res = await call_next(req)
    if req.url.path.startswith("/api"):
        api_log.info(f"{req.method} {req.url.path} -> {res.status_code} ({(time.time()-t)*1000:.0f} ms)")
    return res

def _load(sid):
    p = SESS / sid / "result.json"
    if not p.exists(): raise HTTPException(404, "Session not found")
    return json.loads(p.read_text(encoding="utf-8"))
def _save(r): (SESS / r["id"] / "result.json").write_text(json.dumps(r, indent=1), encoding="utf-8")

def _split(r, present_rolls):
    r["present"] = [dict(roll=x, name=engine.names.get(x, x)) for x in engine.rolls if x in present_rolls]
    r["absent"] = [dict(roll=x, name=engine.names.get(x, x)) for x in engine.rolls if x not in present_rolls]
    r["summary"] = dict(total=len(engine.rolls), present=len(r["present"]), absent=len(r["absent"]),
                        unknown_faces=sum(1 for f in r["faces"] if not f["roll"]))

@app.post("/api/attendance")
async def mark_attendance(file: UploadFile = File(...), subject: str = Form("General"), period: str = Form("")):
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None: raise HTTPException(400, "Not a valid image")
    now = datetime.now(); sid = now.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    (SESS / sid).mkdir(parents=True)
    cv2.imwrite(str(SESS / sid / "original.jpg"), img)
    cap_log.info(f"[{sid}] received {file.filename} {len(data)//1024} KB {img.shape[1]}x{img.shape[0]} subject='{subject}'")
    t = time.time(); faces = engine.identify(img)
    cv2.imwrite(str(SESS / sid / "annotated.jpg"), annotate(img, faces))
    h, w = img.shape[:2]
    for f in faces:  # percentage boxes for the UI overlay
        f["box_pct"] = [f["box"][0]/w*100, f["box"][1]/h*100, (f["box"][2]-f["box"][0])/w*100, (f["box"][3]-f["box"][1])/h*100]
    r = dict(id=sid, subject=subject, period=period, date=now.strftime("%d %b %Y"), time=now.strftime("%I:%M %p"),
             faces=faces, notifications={}, manual_overrides=[], processing_sec=round(time.time()-t, 2))
    _split(r, {f["roll"] for f in faces if f["roll"]}); _save(r)
    att_log.info(f"[{sid}] present={r['summary']['present']} absent={r['summary']['absent']} unknown={r['summary']['unknown_faces']} "
                 f"absentees={[a['roll'] for a in r['absent']]}")
    return r

@app.get("/api/sessions")
def sessions():
    out = []
    for d in sorted(SESS.iterdir(), reverse=True)[:15]:
        if (d / "result.json").exists():
            r = json.loads((d / "result.json").read_text(encoding="utf-8"))
            out.append({k: r[k] for k in ("id", "subject", "date", "time", "summary")})
    return out

@app.get("/api/sessions/{sid}")
def session(sid: str): return _load(sid)

@app.get("/api/sessions/{sid}/image/{kind}")
def image(sid: str, kind: str):
    if kind not in ("original", "annotated"): raise HTTPException(404)
    return FileResponse(SESS / sid / f"{kind}.jpg")

class Override(BaseModel):
    roll: str; present: bool

@app.post("/api/sessions/{sid}/override")
def override(sid: str, o: Override):
    r = _load(sid); cur = {p["roll"] for p in r["present"]}
    (cur.add if o.present else cur.discard)(o.roll)
    r["manual_overrides"].append(dict(roll=o.roll, present=o.present, at=datetime.now().isoformat(timespec="seconds")))
    _split(r, cur); _save(r)
    att_log.info(f"[{sid}] MANUAL override {o.roll} -> {'present' if o.present else 'absent'}")
    return r

class Notify(BaseModel):
    rolls: list[str] | None = None   # None = all absentees

def _notify(sid, rolls, force):
    r = _load(sid); contacts = notifier.load_contacts(); results = []
    for s in [a for a in r["absent"] if rolls is None or a["roll"] in rolls]:
        if not force and r["notifications"].get(s["roll"], {}).get("status") in ("sent", "dry_run"):
            continue
        status, detail = notifier.send(s, r, contacts.get(s["roll"], {}))
        r["notifications"][s["roll"]] = dict(status=status, detail=detail, at=datetime.now().isoformat(timespec="seconds"))
        results.append(dict(roll=s["roll"], status=status))
    _save(r); return dict(sent=results, notifications=r["notifications"])

@app.post("/api/sessions/{sid}/notify")        # first send: skips anyone already notified
def notify(sid: str, n: Notify = Notify()): return _notify(sid, n.rolls, force=False)

@app.post("/api/sessions/{sid}/resend")        # resend: forces re-delivery (all absentees or chosen rolls)
def resend(sid: str, n: Notify = Notify()): return _notify(sid, n.rolls, force=True)

def _notify_staff(sid, force):
    r = _load(sid); prev = r.setdefault("staff_notifications", {})
    prev.update(notifier.send_staff(r, (SESS / sid / "annotated.jpg").read_bytes(), prev, force))
    _save(r); return dict(staff_notifications=prev, provider=notifier.provider())

@app.post("/api/sessions/{sid}/notify-staff")   # class advisor + HOD + subject staff (skips already-sent)
def notify_staff(sid: str): return _notify_staff(sid, force=False)

@app.post("/api/sessions/{sid}/resend-staff")   # force re-delivery to staff
def resend_staff(sid: str): return _notify_staff(sid, force=True)

@app.post("/api/roster/reload")
def reload_roster(): engine.reload(); return dict(students=len(engine.rolls))

@app.get("/api/roster")
def roster(): return [dict(roll=r, name=engine.names[r]) for r in engine.rolls]

app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="ui")
