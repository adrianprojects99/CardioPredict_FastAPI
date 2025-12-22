// Cambiar entre tabs de login y registro
function showTab(tab) {
    const loginForm = document.getElementById('loginForm');
    const registerForm = document.getElementById('registerForm');
    const tabs = document.querySelectorAll('.tab');
    
    if (tab === 'login') {
        loginForm.style.display = 'block';
        registerForm.style.display = 'none';
        tabs[0].classList.add('active');
        tabs[1].classList.remove('active');
    } else {
        loginForm.style.display = 'none';
        registerForm.style.display = 'block';
        tabs[0].classList.remove('active');
        tabs[1].classList.add('active');
    }
}

// Mostrar mensaje
function showMessage(message, type) {
    const messageDiv = document.getElementById('message');
    messageDiv.textContent = message;
    messageDiv.className = `message ${type}`;
    
    setTimeout(() => {
        messageDiv.style.display = 'none';
    }, 5000);
}

// Login
document.getElementById('loginForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const email = document.getElementById('loginEmail').value;
    const password = document.getElementById('loginPassword').value;
    
    try {
        const formData = new FormData();
        formData.append('username', email);
        formData.append('password', password);
        
        const response = await fetch('/token', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok) {
            localStorage.setItem('token', data.access_token);
            window.location.href = '/dashboard';
        } else {
            showMessage(data.detail || 'Error al iniciar sesión', 'error');
        }
    } catch (error) {
        showMessage('Error de conexión', 'error');
    }
});

// Register
document.getElementById('registerForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const name = document.getElementById('registerName').value;
    const email = document.getElementById('registerEmail').value;
    const password = document.getElementById('registerPassword').value;
    const confirmPassword = document.getElementById('registerConfirmPassword').value;
    
    if (password !== confirmPassword) {
        showMessage('Las contraseñas no coinciden', 'error');
        return;
    }
    
    try {
        const response = await fetch('/register', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                email: email,
                password: password,
                full_name: name
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            showMessage('¡Registro exitoso! Ahora puedes iniciar sesión', 'success');
            setTimeout(() => {
                showTab('login');
            }, 2000);
        } else {
            showMessage(data.detail || 'Error al registrarse', 'error');
        }
    } catch (error) {
        showMessage('Error de conexión', 'error');
    }
});

// Recuperar contraseña
function recoverPassword() {
    alert('Funcionalidad de recuperación de contraseña en desarrollo. Por favor contacta al administrador.');
}