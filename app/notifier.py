"""Email delivery. Provider is chosen automatically:
   RESEND_API_KEY set -> Resend API | else SMTP_HOST set -> SMTP | else DRY-RUN (.eml files in data/outbox)."""
import base64, csv, json, os, smtplib, ssl, urllib.error, urllib.request
from email.message import EmailMessage
from pathlib import Path
from .logger import get_logger

log = get_logger("notification")
ROOT = Path(__file__).resolve().parent.parent
OUTBOX = ROOT / "data" / "outbox"; OUTBOX.mkdir(parents=True, exist_ok=True)
OK = ("sent", "dry_run")

def provider():
    return "resend" if _resend_api_key() else "smtp" if os.getenv("SMTP_HOST") else "dry_run"


def _resend_api_key():
    # Keep the older variable name working for existing .env files.
    return os.getenv("RESEND_API_KEY") or os.getenv("RESEND_API")

def load_contacts():
    p = ROOT / "contacts.csv"
    if not p.exists():
        return {}
    with open(p, newline="", encoding="utf-8") as f:
        return {r["roll"].strip(): r for r in csv.DictReader(f)}

def load_staff():
    p = ROOT / "staff.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

def deliver(to, subject, text, html=None, attachments=None, tag="mail"):
    """Send ONE email. attachments = [(filename, bytes)]. Returns (status, detail)."""
    prov, sender = provider(), os.getenv("MAIL_FROM", "Attendance System <onboarding@resend.dev>")
    try:
        if prov == "resend":
            api_key = _resend_api_key()
            body = {"from": sender, "to": [to], "subject": subject, "text": text}
            if html: body["html"] = html
            if attachments:
                body["attachments"] = [{"filename": f, "content": base64.b64encode(d).decode()} for f, d in attachments]
            req = urllib.request.Request("https://api.resend.com/emails", data=json.dumps(body).encode(), method="POST",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
                         "User-Agent": "face-attendance/1.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                rid = json.loads(r.read()).get("id")
            log.info(f"[{tag}] RESEND ok -> {to} (id {rid})"); return "sent", f"{to} (id {rid})"
        m = EmailMessage(); m["Subject"], m["From"], m["To"] = subject, sender, to; m.set_content(text)
        if html: m.add_alternative(html, subtype="html")
        for fn, data in attachments or []: m.add_attachment(data, maintype="image", subtype="jpeg", filename=fn)
        if prov == "smtp":
            port = int(os.getenv("SMTP_PORT", 587))
            smtp_class = smtplib.SMTP_SSL if port == 465 else smtplib.SMTP
            with smtp_class(os.getenv("SMTP_HOST"), port, timeout=20) as s:
                if port != 465:
                    s.starttls(context=ssl.create_default_context())
                s.login(os.getenv("SMTP_USER"), os.getenv("SMTP_PASSWORD")); s.send_message(m)
            log.info(f"[{tag}] SMTP ok -> {to}"); return "sent", to
        (OUTBOX / f"{tag}_{to}.eml").write_bytes(bytes(m))
        log.info(f"[{tag}] DRY-RUN -> {to} (saved in data/outbox)"); return "dry_run", to
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="ignore")[:300]; log.error(f"[{tag}] FAILED -> {to}: HTTP {e.code} {detail}")
        return "failed", f"HTTP {e.code}: {detail}"
    except Exception as e:
        log.error(f"[{tag}] FAILED -> {to}: {e}"); return "failed", str(e)

# ---------- student / guardian absence notice ----------
def send(student, session, contact):
    recipients = [e for e in (contact.get("email"), contact.get("parent_email")) if e]
    if not recipients:
        log.warning(f"{student['roll']}: no email on file"); return "no_contact", "no email in contacts.csv"
    subj = f"Absence notice: {student['name']} - {session['subject']} ({session['date']})"
    text = (f"Dear Sir/Madam,\n\n{student['name']} (Roll No. {student['roll']}) was marked ABSENT for "
            f"{session['subject']} on {session['date']} at {session['time']}.\n\n"
            f"If this is incorrect, please contact the class advisor.\n\n- Automated Attendance System")
    res = [deliver(to, subj, text, tag=f"{session['id']}_{student['roll']}") for to in recipients]
    bad = [d for s, d in res if s == "failed"]
    return ("failed", "; ".join(bad)) if bad else (res[0][0], ", ".join(recipients))

# ---------- class advisor / HOD / subject staff summary ----------
def staff_recipients(subject):
    cfg = load_staff(); out = {}
    for role in ("class_advisor", "hod"):
        if cfg.get(role, {}).get("email"): out[role] = cfg[role]
    subs = {k.strip().lower(): v for k, v in cfg.get("subjects", {}).items()}
    if subject.strip().lower() in subs: out["subject_staff"] = subs[subject.strip().lower()]
    else: log.warning(f"No subject staff configured for '{subject}' in staff.json")
    return out

def send_staff(session, image_bytes, previous, force=False):
    s = session["summary"]; results = {}
    subj = f"Attendance: {session['subject']} - {session['date']} ({s['present']}/{s['total']} present)"
    rows = lambda L, c: "".join(f"<tr><td style='padding:3px 10px'>{x['roll']}</td><td style='padding:3px 10px'>{x['name']}</td></tr>" for x in L) or f"<tr><td colspan=2>None</td></tr>"
    html = (f"<p>Attendance for <b>{session['subject']}</b> on {session['date']}, {session['time']}.</p>"
            f"<p><b>Present:</b> {s['present']} &nbsp; <b>Absent:</b> {s['absent']} &nbsp; <b>Total:</b> {s['total']}</p>"
            f"<h4>Absent ({s['absent']})</h4><table>{rows(session['absent'], 0)}</table>"
            f"<h4>Present ({s['present']})</h4><table>{rows(session['present'], 0)}</table>"
            f"<p>The annotated class photo is attached.</p>")
    text = (f"Attendance for {session['subject']} on {session['date']}, {session['time']}\n"
            f"Present {s['present']} / Absent {s['absent']} / Total {s['total']}\n\nABSENT:\n" +
            "\n".join(f"  {x['roll']}  {x['name']}" for x in session["absent"]) +
            "\n\nPRESENT:\n" + "\n".join(f"  {x['roll']}  {x['name']}" for x in session["present"]))
    for role, p in staff_recipients(session["subject"]).items():
        if not force and previous.get(role, {}).get("status") in OK:
            continue
        st, detail = deliver(p["email"], subj, f"Dear {p.get('name', 'Sir/Madam')},\n\n" + text, html,
                             [("class_photo.jpg", image_bytes)], tag=f"{session['id']}_{role}")
        from datetime import datetime
        results[role] = dict(name=p.get("name", ""), email=p["email"], status=st, detail=detail,
                             at=datetime.now().isoformat(timespec="seconds"))
    return results
