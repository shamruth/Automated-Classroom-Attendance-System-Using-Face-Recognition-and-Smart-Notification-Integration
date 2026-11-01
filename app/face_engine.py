import cv2, numpy as np
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from insightface.app import FaceAnalysis
from .logger import get_logger
from .notifier import load_contacts

enroll_log, det_log, rec_log = get_logger("enrollment"), get_logger("detection"), get_logger("recognition")
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

class FaceEngine:
    def __init__(self, students_dir: Path, threshold: float = 0.38):
        self.dir, self.threshold = Path(students_dir), threshold
        self.model = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        self.reload()

    # ---------- enrollment ----------
    def reload(self):
        self.model.prepare(ctx_id=-1, det_size=(640, 640))
        self.names, self.labels, embs = {}, [], []
        contacts = load_contacts()   # names from contacts.csv win over photo filenames
        for folder in sorted(p for p in self.dir.iterdir() if p.is_dir()):
            roll, got = folder.name, 0
            if contacts.get(roll, {}).get("name"):
                self.names[roll] = contacts[roll]["name"].strip().upper()
            for img in sorted(folder.iterdir()):
                if img.suffix.lower() not in EXTS:
                    continue
                im = cv2.imread(str(img))
                faces = self.model.get(im) if im is not None else []
                if not faces:
                    enroll_log.warning(f"{roll}/{img.name}: no face found - skipped")
                    continue
                f = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                embs.append(f.normed_embedding); self.labels.append(roll)
                self.names.setdefault(roll, img.stem.upper()); got += 1
            self.names.setdefault(roll, roll)
            enroll_log.info(f"{roll} ({self.names[roll]}): {got} face(s) enrolled")
        self.gallery = np.array(embs) if embs else np.zeros((0, 512))
        self.rolls = sorted(self.names)
        enroll_log.info(f"Roster ready: {len(self.rolls)} students, {len(embs)} embeddings")

    # ---------- recognition ----------
    def identify(self, image_bgr):
        h, w = image_bgr.shape[:2]
        scale = max(1.0, 1600 / max(h, w))            # upscale small group photos so tiny faces are detectable
        im = cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale > 1 else image_bgr
        self.model.prepare(ctx_id=-1, det_size=(1280, 1280), det_thresh=0.35)
        faces = self.model.get(im)
        det_log.info(f"Image {w}x{h} (scale {scale:.2f}) -> {len(faces)} faces detected")
        if not faces or len(self.gallery) == 0:
            return []
        E = np.array([f.normed_embedding for f in faces])
        sims = E @ self.gallery.T                      # faces x gallery
        S = np.full((len(self.rolls), len(faces)), -1.0)   # students x faces (best photo per student)
        for gi, roll in enumerate(self.labels):
            ri = self.rolls.index(roll)
            S[ri] = np.maximum(S[ri], sims[:, gi])
        rows, cols = linear_sum_assignment(-S)         # one student <-> one face, globally optimal
        match = {c: (r, S[r, c]) for r, c in zip(rows, cols) if S[r, c] >= self.threshold}
        out = []
        for i, f in enumerate(faces):
            x1, y1, x2, y2 = (f.bbox / scale).tolist()
            best = self.rolls[int(S[:, i].argmax())]
            if i in match:
                r, s = match[i]
                rec_log.info(f"face {i}: MATCH {self.rolls[r]} score={s:.3f}")
                out.append(dict(box=[x1, y1, x2, y2], roll=self.rolls[r], name=self.names[self.rolls[r]], score=round(float(s), 3)))
            else:
                rec_log.info(f"face {i}: UNKNOWN (closest {best} score={S[:, i].max():.3f} < {self.threshold})")
                out.append(dict(box=[x1, y1, x2, y2], roll=None, name="Unknown", score=round(float(S[:, i].max()), 3)))
        return out

def annotate(image_bgr, faces):
    im = image_bgr.copy()
    t = max(2, im.shape[1] // 400)
    for f in faces:
        x1, y1, x2, y2 = map(int, f["box"])
        col = (90, 160, 30) if f["roll"] else (60, 60, 210)
        cv2.rectangle(im, (x1, y1), (x2, y2), col, t)
        label = f["roll"] or "Unknown"
        fs = max(0.4, im.shape[1] / 1400)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        cv2.rectangle(im, (x1, y1 - th - 8), (x1 + tw + 6, y1), col, -1)
        cv2.putText(im, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
    return im
