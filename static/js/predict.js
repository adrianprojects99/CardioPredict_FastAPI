// Verificar autenticación al cargar
window.addEventListener('load', async () => {
    const token = localStorage.getItem('token');
    
    if (!token) {
        window.location.href = '/';
        return;
    }
    
    try {
        const response = await fetch('/me', {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        if (response.ok) {
            const user = await response.json();
            document.getElementById('userName').textContent = user.full_name;
            loadHistory();
        } else {
            localStorage.removeItem('token');
            window.location.href = '/';
        }
    } catch (error) {
        console.error('Error verificando autenticación:', error);
        window.location.href = '/';
    }
});

// Logout
function logout() {
    localStorage.removeItem('token');
    window.location.href = '/';
}

// Actualizar valor de slider
function updateVesselsValue(value) {
    document.getElementById('vesselsValue').textContent = value;
}

// Cargar historial
async function loadHistory() {
    const token = localStorage.getItem('token');
    
    try {
        const response = await fetch('/history', {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        const data = await response.json();
        const historyDiv = document.getElementById('history');
        
        if (data.predictions.length === 0) {
            historyDiv.innerHTML = '<p>No hay análisis previos</p>';
            return;
        }
        
        historyDiv.innerHTML = data.predictions.map(p => `
            <div class="history-item">
                <strong>${p.risk_level}</strong><br>
                <small>${new Date(p.created_at).toLocaleDateString('es')}</small><br>
                <small>${(p.probability * 100).toFixed(1)}%</small>
            </div>
        `).join('');
    } catch (error) {
        console.error('Error cargando historial:', error);
    }
}

// Enviar predicción
document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const token = localStorage.getItem('token');
    const data = {
        age: parseInt(document.getElementById('age').value),
        sex: parseInt(document.getElementById('sex').value),
        chest_pain_type: parseInt(document.getElementById('chestPainType').value),
        resting_bp: parseInt(document.getElementById('restingBP').value),
        cholesterol: parseInt(document.getElementById('cholesterol').value),
        fasting_bs: parseInt(document.getElementById('fastingBS').value),
        max_hr: parseInt(document.getElementById('maxHR').value),
        exercise_angina: parseInt(document.getElementById('exerciseAngina').value),
        st_depression: parseFloat(document.getElementById('stDepression').value),
        num_major_vessels: parseInt(document.getElementById('numMajorVessels').value)
    };
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        if (response.ok) {
            displayResult(result);
            loadHistory();
        } else {
            alert('Error: ' + result.detail);
        }
    } catch (error) {
        alert('Error de conexión');
    }
});

// Mostrar resultado
function displayResult(result) {
    const resultDiv = document.getElementById('result');
    const resultContent = document.getElementById('resultContent');
    
    const riskClass = result.risk_level === 'BAJO' ? 'risk-low' : 
                     result.risk_level === 'MODERADO' ? 'risk-moderate' : 'risk-high';
    
    const factorsHTML = result.factors.length > 0 ? 
        `<div style="margin-top: 20px;">
            <h3>⚕️ Factores de Riesgo Detectados:</h3>
            <ul>
                ${result.factors.map(f => `<li>${f}</li>`).join('')}
            </ul>
        </div>` : 
        '<p>✅ No se detectaron factores de riesgo significativos</p>';
    
    resultContent.innerHTML = `
        <div class="${riskClass}" style="padding: 30px; border-radius: 15px;">
            <h2>RIESGO ${result.risk_level}</h2>
            <p style="font-size: 24px; font-weight: bold; margin: 20px 0;">
                Probabilidad: ${result.risk_percentage.toFixed(1)}%
            </p>
            
            <h3>💡 Recomendaciones:</h3>
            <ol>
                ${result.recommendations.map(r => `<li>${r}</li>`).join('')}
            </ol>
            
            ${factorsHTML}
            
            <div style="margin-top: 30px; padding: 15px; background: rgba(255,255,255,0.3); border-radius: 10px;">
                <strong>⚠️ Importante:</strong> Esta es una herramienta de apoyo al diagnóstico. 
                Los resultados deben ser interpretados por un profesional médico calificado.
                <br><br>
                <strong>📊 Precisión del Modelo:</strong> 81.97% (XGBoost)
            </div>
        </div>
    `;
    
    resultDiv.style.display = 'block';
    resultDiv.scrollIntoView({ behavior: 'smooth' });
}