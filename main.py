from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel, EmailStr
from typing import Optional
import json
import math
from datetime import datetime, timedelta
import jwt
from passlib.context import CryptContext
import sqlite3
import os

# Base directory for resolving paths in serverless environment
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join("/tmp", "cardiopredict.db")

def get_db():
    """Get a database connection. Uses /tmp/ for Vercel compatibility."""
    return sqlite3.connect(DB_PATH)

# Configuración
SECRET_KEY = os.environ.get("SECRET_KEY", "tu-clave-secreta-super-segura-cambiar-en-produccion")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

app = FastAPI(
    title="CardioPredict S.A.",
    description="Sistema de Predicción de Riesgo Cardiovascular con IA",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Montar archivos estáticos
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# Seguridad
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Cargar modelo XGBoost desde JSON (inferencia pura sin dependencia xgboost)
MODEL_PATH = os.path.join(BASE_DIR, "model", "cardio_model.json")
FEATURE_NAMES = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
                 'FastingBS', 'MaxHR', 'ExerciseAngina', 'ST_Depression', 'NumMajorVessels']

def _predict_tree(tree, features):
    """Recorre un arbol de decision y retorna el valor de la hoja."""
    node = 0
    left = tree['left_children']
    right = tree['right_children']
    split_idx = tree['split_indices']
    split_cond = tree['split_conditions']
    while left[node] != -1:
        if features[split_idx[node]] < split_cond[node]:
            node = left[node]
        else:
            node = right[node]
    return split_cond[node]

def model_predict(features_list):
    """Predice probabilidad de riesgo cardiovascular usando el modelo cargado.
    features_list: lista con los 10 valores clinicos en orden de FEATURE_NAMES.
    Retorna probabilidad entre 0 y 1."""
    raw_sum = model_data['base_score_logit']
    for tree in model_data['trees']:
        raw_sum += _predict_tree(tree, features_list)
    return 1.0 / (1.0 + math.exp(-raw_sum))

try:
    with open(MODEL_PATH) as f:
        _raw = json.load(f)
    _base_score = float(_raw['learner']['learner_model_param']['base_score'])
    model_data = {
        'trees': _raw['learner']['gradient_booster']['model']['trees'],
        'base_score_logit': math.log(_base_score / (1 - _base_score)),
    }
    del _raw
    model = True
    print("Modelo cargado correctamente")
except Exception as e:
    print(f"Error cargando modelo: {e}")
    model = None

# === MODELOS PYDANTIC ===

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    organization: str  # Nombre del consultorio/clínica
    plan: str = "micro"  # micro, basico, empresarial

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class PatientInfo(BaseModel):
    """Información del paciente (NO del proveedor)"""
    patient_name: str
    patient_id: str  # Cédula o ID del paciente
    patient_email: Optional[EmailStr] = None
    patient_phone: Optional[str] = None

class PatientData(BaseModel):
    # Datos del paciente
    patient_info: PatientInfo
    
    # Datos clínicos para predicción
    age: int
    sex: int  # 0=Femenino, 1=Masculino
    chest_pain_type: int  # 0-3
    resting_bp: int
    cholesterol: int
    fasting_bs: int  # 0=No, 1=Sí
    max_hr: int
    exercise_angina: int  # 0=No, 1=Sí
    st_depression: float
    num_major_vessels: int  # 0-3

class PredictionResponse(BaseModel):
    prediction_id: int
    patient_info: PatientInfo
    probability: float
    risk_level: str
    risk_percentage: float
    recommendations: list
    factors: list
    timestamp: str
    cost: float  # Costo de esta predicción

class UsageStats(BaseModel):
    """Estadísticas de uso para facturación"""
    provider_name: str
    organization: str
    plan: str
    current_period_start: str
    current_period_end: str
    predictions_this_period: int
    predictions_limit: int  # -1 = ilimitado
    total_cost: float
    predictions_included: int
    overage_predictions: int
    overage_cost: float

# === BASE DE DATOS ===

def init_db():
    """Inicializa la base de datos SQLite"""
    conn = get_db()
    c = conn.cursor()
    
    # Tabla de usuarios (PROVEEDORES DE SERVICIOS)
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            hashed_password TEXT NOT NULL,
            full_name TEXT NOT NULL,
            organization TEXT NOT NULL,
            plan TEXT DEFAULT 'micro',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT 1,
            current_period_start DATE DEFAULT CURRENT_DATE,
            current_period_end DATE DEFAULT (date(CURRENT_DATE, '+1 month'))
        )
    ''')
    
    # Tabla de predicciones con datos del PACIENTE
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            patient_name TEXT NOT NULL,
            patient_id TEXT NOT NULL,
            patient_age INTEGER,
            patient_email TEXT,
            patient_phone TEXT,
            clinical_data TEXT,
            probability REAL,
            risk_level TEXT,
            cost REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Tabla de facturación (billing)
    c.execute('''
        CREATE TABLE IF NOT EXISTS billing (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            period_start DATE,
            period_end DATE,
            predictions_count INTEGER,
            base_cost REAL,
            overage_cost REAL,
            total_cost REAL,
            status TEXT DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()

# Inicializar DB al arrancar
init_db()

# === FUNCIONES DE AUTENTICACIÓN ===

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_user_from_db(email: str):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE email = ?", (email,))
    user = c.fetchone()
    conn.close()
    return user

def create_user_in_db(email: str, hashed_password: str, full_name: str, organization: str, plan: str):
    conn = get_db()
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO users (email, hashed_password, full_name, organization, plan) VALUES (?, ?, ?, ?, ?)",
            (email, hashed_password, full_name, organization, plan)
        )
        conn.commit()
        user_id = c.lastrowid
        conn.close()
        return user_id
    except sqlite3.IntegrityError:
        conn.close()
        return None

# === FUNCIONES DE PRICING ===

PLANS = {
    'micro': {
        'name': 'Micro',
        'description': 'Consultorios de bajo volumen',
        'price_per_prediction': 0.75,
        'monthly_fee': 0,
        'included_predictions': 0,
        'limit': -1  # -1 = sin límite, paga por uso
    },
    'basico': {
        'name': 'Básico',
        'description': 'Clínicas pequeñas',
        'price_per_prediction': 0,
        'monthly_fee': 99,
        'included_predictions': 150,
        'overage_price': 0.65,  # Precio por predicción extra
        'limit': -1
    },
    'empresarial': {
        'name': 'Empresarial',
        'description': 'Hospitales y aseguradoras',
        'price_per_prediction': 0,
        'monthly_fee': 399,
        'included_predictions': -1,  # ilimitado
        'limit': -1
    }
}

def get_plan_details(plan_name: str):
    """Obtiene detalles del plan"""
    return PLANS.get(plan_name.lower(), PLANS['micro'])

def calculate_prediction_cost(user_plan: str, predictions_this_period: int):
    """Calcula el costo de una predicción según el plan"""
    plan = get_plan_details(user_plan)
    
    if user_plan == 'micro':
        # Paga por uso
        return plan['price_per_prediction']
    
    elif user_plan == 'basico':
        # Incluye 150, después paga extra
        if predictions_this_period < plan['included_predictions']:
            return 0  # Incluido en la mensualidad
        else:
            return plan['overage_price']
    
    elif user_plan == 'empresarial':
        # Ilimitado
        return 0
    
    return 0

def get_user_usage_stats(user_id: int):
    """Obtiene estadísticas de uso del proveedor"""
    conn = get_db()
    c = conn.cursor()
    
    # Obtener info del usuario
    c.execute("SELECT full_name, organization, plan, current_period_start, current_period_end FROM users WHERE id = ?", (user_id,))
    user_info = c.fetchone()
    
    if not user_info:
        conn.close()
        return None
    
    full_name, organization, plan, period_start, period_end = user_info
    plan_details = get_plan_details(plan)
    
    # Contar predicciones en el período actual
    c.execute("""
        SELECT COUNT(*), SUM(cost) 
        FROM predictions 
        WHERE user_id = ? AND created_at >= ? AND created_at <= ?
    """, (user_id, period_start, period_end))
    
    count_result = c.fetchone()
    predictions_count = count_result[0] if count_result[0] else 0
    total_cost = count_result[1] if count_result[1] else 0
    
    conn.close()
    
    # Calcular costos
    if plan == 'micro':
        # Todo es overage
        base_cost = 0
        overage_predictions = predictions_count
        overage_cost = total_cost
    elif plan == 'basico':
        base_cost = plan_details['monthly_fee']
        included = plan_details['included_predictions']
        overage_predictions = max(0, predictions_count - included)
        overage_cost = overage_predictions * plan_details['overage_price']
    else:  # empresarial
        base_cost = plan_details['monthly_fee']
        overage_predictions = 0
        overage_cost = 0
    
    return {
        'provider_name': full_name,
        'organization': organization,
        'plan': plan,
        'plan_name': plan_details['name'],
        'current_period_start': period_start,
        'current_period_end': period_end,
        'predictions_this_period': predictions_count,
        'predictions_limit': plan_details['included_predictions'],
        'predictions_included': plan_details['included_predictions'],
        'overage_predictions': overage_predictions,
        'base_cost': base_cost,
        'overage_cost': overage_cost,
        'total_cost': base_cost + overage_cost
    }

def check_usage_limit(user_id: int, user_plan: str):
    """Verifica si el usuario puede hacer más predicciones"""
    # Solo el plan básico tiene límite antes del cobro extra
    # Los demás planes permiten uso
    return True  # Por ahora siempre permitimos, pero registramos el costo

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="No se pudo validar las credenciales",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
    
    user = get_user_from_db(email)
    if user is None:
        raise credentials_exception
    return user

# === FUNCIONES DE PREDICCIÓN ===

def interpret_risk(probability: float):
    """Interpreta la probabilidad de riesgo"""
    if probability < 0.3:
        return {
            'level': 'BAJO',
            'recommendations': [
                'Mantener hábitos saludables actuales',
                'Realizar chequeos preventivos anuales',
                'Continuar con actividad física regular'
            ]
        }
    elif probability < 0.6:
        return {
            'level': 'MODERADO',
            'recommendations': [
                'Consultar con cardiólogo para evaluación',
                'Modificar estilo de vida (dieta y ejercicio)',
                'Monitoreo regular de presión y colesterol'
            ]
        }
    else:
        return {
            'level': 'ALTO',
            'recommendations': [
                'Consulta médica URGENTE recomendada',
                'Evaluación cardiológica completa necesaria',
                'Implementar cambios inmediatos en estilo de vida',
                'Posible necesidad de tratamiento farmacológico'
            ]
        }

def detect_risk_factors(data: dict):
    """Detecta factores de riesgo"""
    factors = []
    if data['age'] > 55:
        factors.append("Edad avanzada (>55 años)")
    if data['resting_bp'] > 140:
        factors.append("Hipertensión (presión >140 mm Hg)")
    if data['cholesterol'] > 240:
        factors.append("Colesterol alto (>240 mg/dL)")
    if data['fasting_bs'] == 1:
        factors.append("Glucosa elevada en ayunas")
    if data['exercise_angina'] == 1:
        factors.append("Angina inducida por ejercicio")
    if data['st_depression'] > 2.0:
        factors.append("Depresión ST significativa")
    return factors

# === ENDPOINTS ===

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Página principal"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Dashboard principal (requiere autenticación)"""
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.post("/register")
async def register(user: UserCreate):
    """Registro de nuevo proveedor de servicios"""
    # Verificar si el usuario ya existe
    existing_user = get_user_from_db(user.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El email ya está registrado"
        )
    
    # Validar plan
    if user.plan.lower() not in PLANS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Plan inválido. Opciones: micro, basico, empresarial"
        )
    
    # Crear usuario
    hashed_password = get_password_hash(user.password)
    user_id = create_user_in_db(user.email, hashed_password, user.full_name, user.organization, user.plan)
    
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error al crear usuario"
        )
    
    plan_details = get_plan_details(user.plan)
    
    return {
        "message": "Proveedor registrado exitosamente",
        "user_id": user_id,
        "email": user.email,
        "organization": user.organization,
        "plan": plan_details['name'],
        "monthly_fee": plan_details['monthly_fee'],
        "price_per_prediction": plan_details['price_per_prediction']
    }

@app.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login y generación de token JWT"""
    user = get_user_from_db(form_data.username)  # username es el email
    if not user or not verify_password(form_data.password, user[2]):  # user[2] = hashed_password
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email o contraseña incorrectos",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user[1]},  # user[1] = email
        expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer"
    }

@app.get("/me")
async def read_users_me(current_user: tuple = Depends(get_current_user)):
    """Obtener información del proveedor actual"""
    return {
        "id": current_user[0],
        "email": current_user[1],
        "full_name": current_user[3],
        "organization": current_user[4],
        "plan": current_user[5],
        "created_at": current_user[6]
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    data: PatientData,
    current_user: tuple = Depends(get_current_user)
):
    """Realizar predicción de riesgo cardiovascular para un PACIENTE"""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo no disponible"
        )
    
    user_id = current_user[0]
    user_plan = current_user[5]  # plan está en posición 5
    
    # Verificar límites
    if not check_usage_limit(user_id, user_plan):
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="Límite de predicciones alcanzado para su plan"
        )
    
    try:
        # Contar predicciones actuales para calcular costo
        conn = get_db()
        c = conn.cursor()
        c.execute("""
            SELECT COUNT(*) FROM predictions 
            WHERE user_id = ? 
            AND created_at >= (SELECT current_period_start FROM users WHERE id = ?)
        """, (user_id, user_id))
        predictions_count = c.fetchone()[0]
        conn.close()
        
        # Calcular costo de esta predicción
        prediction_cost = calculate_prediction_cost(user_plan, predictions_count)
        
        # Preparar datos para el modelo
        features = [data.age, data.sex, data.chest_pain_type, data.resting_bp,
                    data.cholesterol, data.fasting_bs, data.max_hr,
                    data.exercise_angina, data.st_depression, data.num_major_vessels]

        # Realizar predicción
        probability = float(model_predict(features))
        
        # Interpretar resultado
        interpretation = interpret_risk(probability)
        factors = detect_risk_factors(data.dict())
        
        # Guardar predicción con datos del PACIENTE en la base de datos
        conn = get_db()
        c = conn.cursor()
        c.execute("""
            INSERT INTO predictions 
            (user_id, patient_name, patient_id, patient_age, patient_email, patient_phone, 
             clinical_data, probability, risk_level, cost) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            data.patient_info.patient_name,
            data.patient_info.patient_id,
            data.age,  # Edad viene de datos clínicos
            data.patient_info.patient_email,
            data.patient_info.patient_phone,
            str(data.dict()),
            probability,
            interpretation['level'],
            prediction_cost
        ))
        conn.commit()
        prediction_id = c.lastrowid
        conn.close()
        
        return {
            "prediction_id": prediction_id,
            "patient_info": data.patient_info,
            "probability": probability,
            "risk_level": interpretation['level'],
            "risk_percentage": probability * 100,
            "recommendations": interpretation['recommendations'],
            "factors": factors,
            "timestamp": datetime.now().isoformat(),
            "cost": prediction_cost
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en predicción: {str(e)}"
        )

@app.get("/history")
async def get_history(current_user: tuple = Depends(get_current_user)):
    """Obtener historial de predicciones del proveedor"""
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        SELECT id, patient_name, patient_id, probability, risk_level, cost, created_at
        FROM predictions 
        WHERE user_id = ? 
        ORDER BY created_at DESC 
        LIMIT 20
    """, (current_user[0],))
    predictions = c.fetchall()
    conn.close()
    
    return {
        "predictions": [
            {
                "id": p[0],
                "patient_name": p[1],
                "patient_id": p[2],
                "probability": p[3],
                "risk_level": p[4],
                "cost": p[5],
                "created_at": p[6]
            }
            for p in predictions
        ]
    }

@app.get("/usage", response_model=UsageStats)
async def get_usage_stats(current_user: tuple = Depends(get_current_user)):
    """Obtener estadísticas de uso y facturación del proveedor"""
    stats = get_user_usage_stats(current_user[0])
    
    if not stats:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Estadísticas no encontradas"
        )
    
    return stats

@app.get("/billing/generate")
async def generate_billing_report(current_user: tuple = Depends(get_current_user)):
    """Generar reporte de facturación para el período actual"""
    user_id = current_user[0]
    stats = get_user_usage_stats(user_id)
    
    if not stats:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No se pudo generar reporte"
        )
    
    # Guardar en tabla de billing
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        INSERT INTO billing 
        (user_id, period_start, period_end, predictions_count, base_cost, overage_cost, total_cost, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id,
        stats['current_period_start'],
        stats['current_period_end'],
        stats['predictions_this_period'],
        stats['base_cost'],
        stats['overage_cost'],
        stats['total_cost'],
        'pending'
    ))
    conn.commit()
    billing_id = c.lastrowid
    conn.close()
    
    return {
        "billing_id": billing_id,
        "status": "generated",
        "stats": stats,
        "invoice_details": {
            "provider": stats['provider_name'],
            "organization": stats['organization'],
            "plan": stats['plan_name'],
            "period": f"{stats['current_period_start']} a {stats['current_period_end']}",
            "predictions_count": stats['predictions_this_period'],
            "base_fee": f"${stats['base_cost']:.2f}",
            "overage_fee": f"${stats['overage_cost']:.2f}",
            "total": f"${stats['total_cost']:.2f}"
        }
    }

@app.get("/usage/report")
async def get_usage_report(current_user: tuple = Depends(get_current_user)):
    """Generar informe detallado de uso (no factura, solo resumen + detalle)"""
    user_id = current_user[0]
    stats = get_user_usage_stats(user_id)
    
    if not stats:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No se pudo generar informe"
        )
    
    # Obtener detalle de cada predicción del período
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        SELECT id, patient_name, patient_id, patient_age, risk_level, cost, created_at
        FROM predictions 
        WHERE user_id = ? 
        AND created_at >= ? 
        AND created_at <= ?
        ORDER BY created_at DESC
    """, (user_id, stats['current_period_start'], stats['current_period_end']))
    
    predictions = c.fetchall()
    conn.close()
    
    return {
        "report_type": "usage_report",
        "summary": stats,
        "details": [
            {
                "id": p[0],
                "patient_name": p[1],
                "patient_id": p[2],
                "patient_age": p[3],
                "risk_level": p[4],
                "cost": p[5],
                "date": p[6]
            }
            for p in predictions
        ]
    }

@app.get("/prediction/{prediction_id}/patient-report")
async def get_patient_report(
    prediction_id: int,
    current_user: tuple = Depends(get_current_user)
):
    """Generar reporte amigable para el paciente (sin costos)"""
    user_id = current_user[0]
    
    # Obtener predicción
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        SELECT patient_name, patient_id, patient_age, patient_email, patient_phone,
               clinical_data, probability, risk_level, created_at
        FROM predictions 
        WHERE id = ? AND user_id = ?
    """, (prediction_id, user_id))
    
    pred = c.fetchone()
    conn.close()
    
    if not pred:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Predicción no encontrada"
        )
    
    # Parsear datos clínicos
    import ast
    clinical_data = ast.literal_eval(pred[5])
    
    # Interpretar riesgo
    probability = pred[6]
    interpretation = interpret_risk(probability)
    factors = detect_risk_factors(clinical_data)
    
    # Generar explicaciones detalladas
    risk_explanation = generate_patient_explanation(probability, interpretation['level'])
    
    return {
        "report_type": "patient_report",
        "patient_info": {
            "name": pred[0],
            "id": pred[1],
            "age": pred[2],
            "email": pred[3],
            "phone": pred[4]
        },
        "analysis_date": pred[8],
        "risk_level": interpretation['level'],
        "risk_percentage": probability * 100,
        "risk_explanation": risk_explanation,
        "clinical_parameters": {
            "blood_pressure": f"{clinical_data.get('resting_bp', 'N/A')} mm Hg",
            "cholesterol": f"{clinical_data.get('cholesterol', 'N/A')} mg/dL",
            "max_heart_rate": f"{clinical_data.get('max_hr', 'N/A')} bpm",
            "fasting_blood_sugar": "Elevado" if clinical_data.get('fasting_bs') == 1 else "Normal",
            "exercise_angina": "Sí" if clinical_data.get('exercise_angina') == 1 else "No"
        },
        "risk_factors": factors,
        "recommendations": interpretation['recommendations'],
        "detailed_recommendations": generate_detailed_recommendations(interpretation['level']),
        "lifestyle_tips": generate_lifestyle_tips(),
        "when_to_see_doctor": generate_when_to_see_doctor(interpretation['level']),
        "disclaimer": "Este reporte es generado por un sistema de inteligencia artificial y tiene fines informativos únicamente. No reemplaza la consulta médica profesional. Consulte siempre con su médico para un diagnóstico y tratamiento adecuados."
    }

@app.post("/billing/{billing_id}/pay")
async def simulate_payment(
    billing_id: int,
    current_user: tuple = Depends(get_current_user)
):
    """Simular pago de factura (Checkout simulado)"""
    user_id = current_user[0]
    
    # Obtener factura
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        SELECT total_cost, status 
        FROM billing 
        WHERE id = ? AND user_id = ?
    """, (billing_id, user_id))
    
    billing = c.fetchone()
    
    if not billing:
        conn.close()
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Factura no encontrada"
        )
    
    if billing[1] == 'paid':
        conn.close()
        return {
            "message": "Esta factura ya fue pagada",
            "status": "already_paid"
        }
    
    # Simular procesamiento de pago
    # En producción aquí iría la integración con Stripe/PayPal/Checkout.com
    c.execute("""
        UPDATE billing 
        SET status = 'paid' 
        WHERE id = ?
    """, (billing_id,))
    conn.commit()
    conn.close()
    
    return {
        "message": "Pago procesado exitosamente (SIMULADO)",
        "billing_id": billing_id,
        "amount_paid": billing[0],
        "payment_method": "Simulación - Tarjeta de crédito",
        "transaction_id": f"SIM-{billing_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "status": "paid",
        "note": "Este es un pago simulado con fines académicos"
    }

# === FUNCIONES AUXILIARES PARA REPORTES ===

def generate_patient_explanation(probability: float, risk_level: str):
    """Genera explicación detallada para el paciente"""
    if risk_level == "BAJO":
        return {
            "summary": "Sus resultados muestran un riesgo bajo de enfermedad cardiovascular.",
            "what_it_means": "Esto significa que según los parámetros evaluados, la probabilidad de desarrollar problemas cardiovasculares en el corto plazo es reducida. Sus valores están dentro de rangos favorables.",
            "good_news": "¡Felicitaciones! Está en buen camino. Mantener estos resultados es importante para su salud a largo plazo.",
            "next_steps": "Continue con sus hábitos saludables actuales y realice chequeos preventivos anuales."
        }
    elif risk_level == "MODERADO":
        return {
            "summary": "Sus resultados indican un riesgo moderado de enfermedad cardiovascular.",
            "what_it_means": "Esto significa que existen algunos factores que podrían aumentar su riesgo de problemas cardíacos. No es motivo de alarma inmediata, pero sí requiere atención.",
            "good_news": "La buena noticia es que muchos de estos factores pueden mejorarse con cambios en el estilo de vida y, en algunos casos, con tratamiento médico adecuado.",
            "next_steps": "Se recomienda consultar con un cardiólogo para una evaluación más detallada y establecer un plan de acción personalizado."
        }
    else:  # ALTO
        return {
            "summary": "Sus resultados muestran un riesgo elevado de enfermedad cardiovascular.",
            "what_it_means": "Los parámetros evaluados sugieren que existe una probabilidad aumentada de problemas cardíacos. Esto requiere atención médica profesional pronto.",
            "good_news": "Con el tratamiento y cambios adecuados, muchas personas logran reducir significativamente su riesgo cardiovascular.",
            "next_steps": "Es importante que consulte con un cardiólogo lo antes posible para una evaluación completa y comenzar un plan de tratamiento si es necesario."
        }

def generate_detailed_recommendations(risk_level: str):
    """Genera recomendaciones detalladas según nivel de riesgo"""
    base_recommendations = [
        {
            "category": "Alimentación",
            "title": "Dieta Saludable para el Corazón",
            "tips": [
                "Aumente el consumo de frutas y vegetales (5 porciones al día)",
                "Elija granos enteros en lugar de refinados",
                "Limite la sal a menos de 5g al día (1 cucharadita)",
                "Reduzca las grasas saturadas (carnes rojas, productos lácteos enteros)",
                "Incluya pescado graso (salmón, atún) 2 veces por semana",
                "Limite el consumo de azúcares añadidos"
            ]
        },
        {
            "category": "Actividad Física",
            "title": "Ejercicio Regular",
            "tips": [
                "Realice al menos 150 minutos de actividad moderada por semana",
                "Puede dividirlo en sesiones de 30 minutos, 5 días a la semana",
                "Incluya ejercicios de fuerza 2 veces por semana",
                "Caminar, nadar, andar en bicicleta son excelentes opciones",
                "Comience gradualmente si no está acostumbrado al ejercicio"
            ]
        }
    ]
    
    if risk_level in ["MODERADO", "ALTO"]:
        base_recommendations.extend([
            {
                "category": "Control Médico",
                "title": "Monitoreo Regular",
                "tips": [
                    "Mida su presión arterial regularmente en casa",
                    "Realice análisis de sangre según lo indique su médico",
                    "Mantenga un registro de sus valores",
                    "No suspenda medicamentos sin consultar a su médico",
                    "Asista a todas sus citas de seguimiento"
                ]
            }
        ])
    
    if risk_level == "ALTO":
        base_recommendations.append({
            "category": "Urgente",
            "title": "Atención Inmediata",
            "tips": [
                "Programe cita con cardiólogo en las próximas 2 semanas",
                "Si experimenta dolor en el pecho, dificultad para respirar o mareos, busque atención médica inmediata",
                "Considere reducir o eliminar el consumo de alcohol y tabaco",
                "Evite situaciones de estrés extremo",
                "Informe a familiares cercanos sobre su situación"
            ]
        })
    
    return base_recommendations

def generate_lifestyle_tips():
    """Genera consejos generales de estilo de vida"""
    return [
        {
            "icon": "🚭",
            "title": "No Fumar",
            "description": "Si fuma, dejar de fumar es lo más importante que puede hacer por su corazón. Pida ayuda a su médico."
        },
        {
            "icon": "😴",
            "title": "Dormir Bien",
            "description": "Duerma 7-8 horas cada noche. El sueño insuficiente se asocia con mayor riesgo cardiovascular."
        },
        {
            "icon": "🧘",
            "title": "Manejo del Estrés",
            "description": "Practique técnicas de relajación como meditación, yoga o respiración profunda."
        },
        {
            "icon": "💊",
            "title": "Medicamentos",
            "description": "Si tiene medicamentos prescritos, tómelos exactamente como lo indique su médico."
        },
        {
            "icon": "⚖️",
            "title": "Peso Saludable",
            "description": "Mantener un peso adecuado reduce la carga sobre su corazón."
        },
        {
            "icon": "🍷",
            "title": "Alcohol Moderado",
            "description": "Si bebe alcohol, hágalo con moderación (máximo 1-2 bebidas al día)."
        }
    ]

def generate_when_to_see_doctor(risk_level: str):
    """Genera información sobre cuándo buscar atención médica"""
    emergency_signs = {
        "title": "🚨 Busque Atención Médica INMEDIATA si experimenta:",
        "signs": [
            "Dolor o presión en el pecho que dura más de unos minutos",
            "Dolor que se extiende al brazo, cuello, mandíbula o espalda",
            "Dificultad para respirar severa",
            "Náuseas o vómitos junto con malestar en el pecho",
            "Sudoración fría repentina",
            "Mareos o desmayos",
            "Latidos cardíacos muy rápidos o irregulares"
        ]
    }
    
    routine_visit = {
        "title": "📅 Programe una cita regular si nota:",
        "signs": [
            "Fatiga inusual o falta de energía",
            "Hinchazón en piernas, tobillos o pies",
            "Tos persistente, especialmente al acostarse",
            "Necesidad de orinar frecuentemente por la noche",
            "Palpitaciones ocasionales",
            "Cambios en su capacidad para hacer ejercicio"
        ]
    }
    
    if risk_level == "ALTO":
        return {
            "urgency": "alta",
            "recommendation": "Dado su nivel de riesgo, se recomienda consultar con un cardiólogo en las próximas 1-2 semanas, incluso si no tiene síntomas.",
            "emergency_signs": emergency_signs,
            "routine_visit": routine_visit
        }
    elif risk_level == "MODERADO":
        return {
            "urgency": "moderada",
            "recommendation": "Se recomienda agendar una consulta con su médico o cardiólogo en el próximo mes para evaluación.",
            "emergency_signs": emergency_signs,
            "routine_visit": routine_visit
        }
    else:
        return {
            "urgency": "baja",
            "recommendation": "Continue con sus chequeos anuales de rutina.",
            "emergency_signs": emergency_signs,
            "routine_visit": routine_visit
        }

@app.get("/plans")
async def get_available_plans():
    """Obtener planes disponibles y sus precios"""
    return {
        "plans": [
            {
                "id": "micro",
                "name": plan['name'],
                "description": plan['description'],
                "monthly_fee": plan['monthly_fee'],
                "price_per_prediction": plan.get('price_per_prediction', 0),
                "included_predictions": plan['included_predictions'],
                "overage_price": plan.get('overage_price', 0),
                "recommended_for": "Consultorios con <50 análisis/mes" if key == 'micro' else
                                  "Clínicas con 50-150 análisis/mes" if key == 'basico' else
                                  "Hospitales y aseguradoras con >150 análisis/mes"
            }
            for key, plan in PLANS.items()
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
