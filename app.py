import io, os, base64, datetime, jwt
from typing import List, Optional, Tuple
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from passlib.hash import bcrypt
from sqlalchemy import create_engine, Column, Integer, String, LargeBinary, DateTime, Text, ForeignKey, Boolean
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
import numpy as np
import cv2
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

JWT_SECRET = os.environ.get("JWT_SECRET", "dev_secret_change_me")
ALGORITHM = "HS256"

# ---------- DB ----------
DB_URL = "sqlite:///./hms.db"
engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, default="doctor")  # admin/doctor/receptionist
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    # one face sample for simplicity
    face_image = Column(LargeBinary, nullable=True)  # stored as JPEG bytes

class Patient(Base):
    __tablename__ = "patients"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    age = Column(Integer, nullable=True)
    gender = Column(String, nullable=True)
    phone = Column(String, nullable=True)
    emergency_flag = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    face_image = Column(LargeBinary, nullable=True)  # for face search
    treatments = relationship("Treatment", back_populates="patient", cascade="all, delete-orphan")

class Treatment(Base):
    __tablename__ = "treatments"
    id = Column(Integer, primary_key=True)
    patient_id = Column(Integer, ForeignKey("patients.id"))
    added_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    diagnosis = Column(Text, nullable=True)
    prescription = Column(Text, nullable=True)
    notes = Column(Text, nullable=True)
    ts = Column(DateTime, default=datetime.datetime.utcnow)
    patient = relationship("Patient", back_populates="treatments")

Base.metadata.create_all(engine)

# ---------- App ----------
app = FastAPI(title="HMS Face Backend", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Utils ----------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def hash_pw(p): return bcrypt.hash(p)
def verify_pw(p, h): return bcrypt.verify(p, h)

def create_token(user: User):
    payload = {
        "sub": str(user.id),
        "role": user.role,
        "name": user.name,
        "email": user.email,
        "iat": int(datetime.datetime.utcnow().timestamp()),
        "exp": int((datetime.datetime.utcnow() + datetime.timedelta(hours=12)).timestamp()),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=ALGORITHM)

def decode_token(token: str):
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

def current_user(token: str = Depends(lambda authorization: authorization),
                 db=Depends(get_db)):
        # Simple Bearer token extraction
        # FastAPI dependency injection trick: header isn't auto-passed, so use request workaround:
        from fastapi import Request
        from fastapi.params import Depends as _Depends
        async def _extract(req: Request):
            auth = req.headers.get("Authorization", "")
            if not auth.startswith("Bearer "):
                raise HTTPException(status_code=401, detail="Missing token")
            return auth.split(" ", 1)[1]
        return _Depends(_extract)

# Re-define with proper dependency since we can't reference Depends inside itself cleanly
from fastapi import Request
async def get_token(req: Request):
    auth = req.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing token")
    return auth.split(" ", 1)[1]

def require_roles(*roles):
    def _inner(payload=Depends(lambda token=Depends(get_token): decode_token(token))):
        if payload.get("role") not in roles:
            raise HTTPException(status_code=403, detail="Forbidden")
        return payload
    return _inner

# ---------- Face helpers ----------
# use Haar cascade for detection; LBPH for recognition
CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def read_image_from_upload(file_bytes: bytes) -> Optional[np.ndarray]:
    img_arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    return img

def extract_face_gray(img: np.ndarray, size: Tuple[int,int]=(200,200)) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = CASCADE.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60,60))
    if len(faces) == 0:
        return None
    x,y,w,h = sorted(faces, key=lambda f:f[2]*f[3], reverse=True)[0]
    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, size)
    return face

def train_lbph(images: List[np.ndarray], labels: List[int]):
    if len(images) == 0:
        return None
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, np.array(labels))
    return recognizer

def predict_lbph(recognizer, face_img: np.ndarray) -> Tuple[int, float]:
    # returns (label, confidence) lower is better in LBPH
    return recognizer.predict(face_img)

# ---------- Startup: ensure admin ----------
def ensure_admin():
    db = SessionLocal()
    try:
        admin_email = os.environ.get("ADMIN_EMAIL", "admin@clinic.local")
        pwd = os.environ.get("ADMIN_PASSWORD", "Admin@123")
        u = db.query(User).filter_by(email=admin_email).first()
        if not u:
            u = User(name="Admin", email=admin_email, password_hash=hash_pw(pwd), role="admin")
            db.add(u)
            db.commit()
    finally:
        db.close()
ensure_admin()

# ---------- Schemas ----------
class RegisterUser(BaseModel):
    name: str
    email: str
    password: str
    role: str = "doctor"

class LoginPassword(BaseModel):
    email: str
    password: str

class PatientIn(BaseModel):
    name: str
    age: Optional[int] = None
    gender: Optional[str] = None
    phone: Optional[str] = None

class TreatmentIn(BaseModel):
    diagnosis: Optional[str] = None
    prescription: Optional[str] = None
    notes: Optional[str] = None

# ---------- Auth & Users ----------
@app.post("/api/register_user")
async def register_user(
    name: str = Form(...), email: str = Form(...),
    password: str = Form(...), role: str = Form("doctor"),
    face: Optional[UploadFile] = File(None), db=Depends(get_db)):
    if db.query(User).filter_by(email=email).first():
        raise HTTPException(400, "Email already registered")
    face_bytes = await face.read() if face else None
    # validate face if provided
    if face_bytes:
        img = read_image_from_upload(face_bytes)
        if img is None:
            raise HTTPException(400, "Invalid face image")
        face_gray = extract_face_gray(img)
        if face_gray is None:
            # allow registration without face (you asked for fallback)
            face_bytes = None
        else:
            # store normalized jpeg
            _, enc = cv2.imencode(".jpg", face_gray)
            face_bytes = enc.tobytes()
    u = User(name=name, email=email, password_hash=hash_pw(password), role=role, face_image=face_bytes)
    db.add(u); db.commit()
    return {"ok": True, "message": "User registered"}

@app.post("/api/login_password")
async def login_password(body: LoginPassword, db=Depends(get_db)):
    u = db.query(User).filter_by(email=body.email).first()
    if not u or not verify_pw(body.password, u.password_hash):
        raise HTTPException(401, "Invalid credentials")
    return {"ok": True, "token": create_token(u), "role": u.role, "name": u.name}

@app.post("/api/login_face")
async def login_face(img: UploadFile = File(...), db=Depends(get_db)):
    file_bytes = await img.read()
    img_cv = read_image_from_upload(file_bytes)
    if img_cv is None:
        raise HTTPException(400, "Invalid image")
    face_gray = extract_face_gray(img_cv)
    if face_gray is None:
        # IMPORTANT: fallback info
        raise HTTPException(422, "Face not detected. Please use password login.")
    # Build dataset from users who have face
    users = db.query(User).filter(User.face_image != None).all()
    if not users:
        raise HTTPException(422, "No face data registered. Use password login.")
    train_images, labels, id_map = [], [], {}
    for idx, u in enumerate(users):
        arr = np.frombuffer(u.face_image, np.uint8)
        face_img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        train_images.append(face_img); labels.append(idx); id_map[idx] = u.id
    recognizer = train_lbph(train_images, labels)
    pred_label, confidence = predict_lbph(recognizer, face_gray)
    # LBPH confidence: lower = better; empirical threshold:
    if confidence > 90:  # adjust if needed
        raise HTTPException(401, f"Face not recognized (conf={confidence:.1f}). Use password login.")
    user_id = id_map[pred_label]
    u = db.get(User, user_id)
    return {"ok": True, "token": create_token(u), "role": u.role, "name": u.name, "confidence": confidence}

@app.get("/api/whoami")
async def whoami(payload=Depends(lambda token=Depends(get_token): decode_token(token))):
    return payload

# ---------- Patients ----------
@app.post("/api/patients")
async def add_patient(body: PatientIn, payload=Depends(require_roles("admin","doctor","receptionist")), db=Depends(get_db)):
    p = Patient(**body.model_dump())
    db.add(p); db.commit()
    return {"ok": True, "id": p.id}

@app.get("/api/patients")
async def list_patients(q: Optional[str]=None, payload=Depends(require_roles("admin","doctor","receptionist")), db=Depends(get_db)):
    query = db.query(Patient)
    if q:
        query = query.filter(Patient.name.ilike(f"%{q}%"))
    data = [{"id":p.id,"name":p.name,"age":p.age,"gender":p.gender,"phone":p.phone,"emergency":p.emergency_flag} for p in query.order_by(Patient.id.desc()).all()]
    return {"ok": True, "items": data}

@app.get("/api/patients/{pid}")
async def get_patient(pid:int, payload=Depends(require_roles("admin","doctor","receptionist")), db=Depends(get_db)):
    p = db.get(Patient, pid)
    if not p: raise HTTPException(404, "Not found")
    tr = [{"id":t.id,"diagnosis":t.diagnosis,"prescription":t.prescription,"notes":t.notes,"ts":t.ts.isoformat()} for t in p.treatments]
    return {"ok": True, "patient":{"id":p.id,"name":p.name,"age":p.age,"gender":p.gender,"phone":p.phone,"emergency":p.emergency_flag}, "treatments": tr}

@app.post("/api/patients/{pid}/treatments")
async def add_treatment(pid:int, body: TreatmentIn, payload=Depends(require_roles("admin","doctor")), db=Depends(get_db)):
    p = db.get(Patient, pid)
    if not p: raise HTTPException(404, "Not found")
    u_id = int(decode_token((await get_token)).get("sub")) if False else None  # kept simple
    t = Treatment(patient_id=pid, diagnosis=body.diagnosis, prescription=body.prescription, notes=body.notes, added_by=None)
    db.add(t); db.commit()
    return {"ok": True, "id": t.id}

# ---------- Emergency Mode ----------
@app.post("/api/emergency_add")
async def emergency_add(name: str = Form(...), phone: str = Form(None), img: Optional[UploadFile] = File(None), db=Depends(get_db)):
    face_bytes = None
    if img:
        b = await img.read()
        im = read_image_from_upload(b)
        if im is not None:
            face = extract_face_gray(im)
            if face is not None:
                _, enc = cv2.imencode(".jpg", face)
                face_bytes = enc.tobytes()
    p = Patient(name=name, phone=phone, emergency_flag=True, face_image=face_bytes)
    db.add(p); db.commit()
    return {"ok": True, "id": p.id, "emergency": True}

# ---------- Face-based Patient Search ----------
@app.post("/api/patient_search_face")
async def patient_search_face(img: UploadFile = File(...), payload=Depends(require_roles("admin","doctor","receptionist")), db=Depends(get_db)):
    b = await img.read()
    im = read_image_from_upload(b)
    if im is None: raise HTTPException(400, "Invalid image")
    face = extract_face_gray(im)
    if face is None: raise HTTPException(422, "No face detected")
    # Train on patients with face
    patients = db.query(Patient).filter(Patient.face_image != None).all()
    if not patients: raise HTTPException(404, "No patient face data available")
    imgs, labels, id_map = [], [], {}
    for idx, p in enumerate(patients):
        arr = np.frombuffer(p.face_image, np.uint8)
        gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        imgs.append(gray); labels.append(idx); id_map[idx]=p.id
    rec = train_lbph(imgs, labels)
    lbl, conf = predict_lbph(rec, face)
    if conf > 90:
        raise HTTPException(404, f"No close match (conf={conf:.1f})")
    pid = id_map[lbl]
    p = db.get(Patient, pid)
    return {"ok": True, "match":{"id":p.id,"name":p.name,"phone":p.phone,"emergency":p.emergency_flag}, "confidence": conf}

# ---------- Attach/Update Patient Face ----------
@app.post("/api/patients/{pid}/face")
async def set_patient_face(pid:int, img: UploadFile = File(...), payload=Depends(require_roles("admin","doctor","receptionist")), db=Depends(get_db)):
    p = db.get(Patient, pid)
    if not p: raise HTTPException(404, "Not found")
    b = await img.read()
    im = read_image_from_upload(b)
    if im is None: raise HTTPException(400, "Invalid image")
    face = extract_face_gray(im)
    if face is None: raise HTTPException(422, "No face detected")
    _, enc = cv2.imencode(".jpg", face)
    p.face_image = enc.tobytes()
    db.commit()
    return {"ok": True}

# ---------- PDF Report ----------
@app.get("/api/patients/{pid}/report.pdf")
async def report_pdf(pid:int, payload=Depends(require_roles("admin","doctor","receptionist")), db=Depends(get_db)):
    p = db.get(Patient, pid)
    if not p: raise HTTPException(404, "Not found")
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    y = h - 50
    c.setFont("Helvetica-Bold", 16); c.drawString(50, y, "Patient Report"); y -= 30
    c.setFont("Helvetica", 12)
    c.drawString(50, y, f"Name: {p.name}"); y -= 18
    c.drawString(50, y, f"Age: {p.age or '-'}    Gender: {p.gender or '-'}    Phone: {p.phone or '-'}"); y -= 18
    c.drawString(50, y, f"Emergency: {'YES' if p.emergency_flag else 'No'}"); y -= 24
    c.setFont("Helvetica-Bold", 13); c.drawString(50, y, "Treatments:"); y -= 20
    c.setFont("Helvetica", 11)
    for t in p.treatments[:40]:
        lines = [
            f"- {t.ts.strftime('%Y-%m-%d %H:%M')}   Diagnosis: {t.diagnosis or '-'}",
            f"  Prescription: {t.prescription or '-'}",
            f"  Notes: {t.notes or '-'}",
        ]
        for ln in lines:
            c.drawString(60, y, ln); y -= 16
            if y < 80:
                c.showPage(); y = h - 50; c.setFont("Helvetica", 11)
    c.showPage(); c.save()
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/pdf", headers={"Content-Disposition": f'inline; filename="patient_{pid}_report.pdf"'})
