"""
FastAPI application for Iris flower classification
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator
import joblib
import numpy as np
from typing import List
import os

# Initialize FastAPI app
app = FastAPI(
    title="Iris Flower Classification API",
    description="A machine learning API to classify Iris flowers based on their measurements",
    version="1.0.0"
)

# Load the trained model at startup
try:
    model = joblib.load("model.pkl")
    class_names = ["setosa", "versicolor", "virginica"]
except FileNotFoundError:
    raise RuntimeError("Model file 'model.pkl' not found. Please train the model first.")

# Pydantic models for request and response
class IrisInput(BaseModel):
    """Input data model for Iris flower measurements"""
    sepal_length: float = Field(..., description="Sepal length in cm", ge=0, le=10)
    sepal_width: float = Field(..., description="Sepal width in cm", ge=0, le=10)
    petal_length: float = Field(..., description="Petal length in cm", ge=0, le=10)
    petal_width: float = Field(..., description="Petal width in cm", ge=0, le=10)
    
    @field_validator('sepal_length', 'sepal_width', 'petal_length', 'petal_width')
    @classmethod
    def validate_measurements(cls, v):
        if v < 0 or v > 10:
            raise ValueError('Measurements must be between 0 and 10 cm')
        return v

class PredictionOutput(BaseModel):
    """Output data model for prediction results"""
    species: str = Field(..., description="Predicted Iris species")
    confidence: float = Field(..., description="Confidence score (0-1)")
    probabilities: dict = Field(..., description="Probabilities for each class")

class HealthCheck(BaseModel):
    """Health check response model"""
    status: str
    is_model_loaded: bool = Field(..., description="Whether the model is loaded")

# API Endpoints
@app.get("/", response_class=HTMLResponse, summary="Modern Iris Classification Dashboard")
async def root():
    """Modern dashboard interface for Iris flower classification with dark theme and advanced UI"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🧬 Iris AI - Neural Classification</title>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <style>
            :root {
                --primary: #00d4ff;
                --secondary: #7c3aed;
                --accent: #f59e0b;
                --success: #10b981;
                --danger: #ef4444;
                --dark: #0f172a;
                --dark-light: #1e293b;
                --dark-lighter: #334155;
                --text: #f8fafc;
                --text-muted: #94a3b8;
                --glass: rgba(15, 23, 42, 0.8);
            }
            
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
                min-height: 100vh;
                color: var(--text);
                overflow-x: hidden;
            }
            
            .background-pattern {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-image: 
                    radial-gradient(circle at 20% 80%, rgba(0, 212, 255, 0.1) 0%, transparent 50%),
                    radial-gradient(circle at 80% 20%, rgba(124, 58, 237, 0.1) 0%, transparent 50%),
                    radial-gradient(circle at 40% 40%, rgba(245, 158, 11, 0.05) 0%, transparent 50%);
                z-index: -1;
            }
            
            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }
            
            .header {
                text-align: center;
                margin-bottom: 40px;
                position: relative;
            }
            
            .header::before {
                content: '';
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                width: 200px;
                height: 200px;
                background: radial-gradient(circle, var(--primary) 0%, transparent 70%);
                opacity: 0.1;
                border-radius: 50%;
                z-index: -1;
            }
            
            .header h1 {
                font-size: 3.5rem;
                font-weight: 800;
                background: linear-gradient(45deg, var(--primary), var(--secondary));
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                margin-bottom: 10px;
                text-shadow: 0 0 30px rgba(0, 212, 255, 0.3);
            }
            
            .header p {
                font-size: 1.2rem;
                color: var(--text-muted);
                font-weight: 300;
            }
            
            .dashboard {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 30px;
                margin-bottom: 30px;
            }
            
            .card {
                background: var(--glass);
                backdrop-filter: blur(20px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 20px;
                padding: 30px;
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
                transition: all 0.3s ease;
            }
            
            .card:hover {
                transform: translateY(-5px);
                box-shadow: 0 30px 60px rgba(0, 0, 0, 0.3);
                border-color: rgba(0, 212, 255, 0.3);
            }
            
            .card-header {
                display: flex;
                align-items: center;
                margin-bottom: 25px;
            }
            
            .card-header i {
                font-size: 1.5rem;
                margin-right: 15px;
                color: var(--primary);
            }
            
            .card-header h3 {
                font-size: 1.4rem;
                font-weight: 600;
            }
            
            .input-container {
                position: relative;
                margin-bottom: 20px;
            }
            
            .input-container label {
                display: block;
                font-size: 0.9rem;
                font-weight: 500;
                color: var(--text-muted);
                margin-bottom: 8px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            
            .input-container input {
                width: 100%;
                padding: 15px 20px;
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 12px;
                color: var(--text);
                font-size: 1rem;
                transition: all 0.3s ease;
            }
            
            .input-container input:focus {
                outline: none;
                border-color: var(--primary);
                box-shadow: 0 0 0 3px rgba(0, 212, 255, 0.1);
                background: rgba(255, 255, 255, 0.08);
            }
            
            .input-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 15px;
                margin-bottom: 25px;
            }
            
            .btn {
                background: linear-gradient(45deg, var(--primary), var(--secondary));
                color: white;
                border: none;
                padding: 16px 32px;
                border-radius: 12px;
                font-size: 1.1rem;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            
            .btn::before {
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
                transition: left 0.5s;
            }
            
            .btn:hover::before {
                left: 100%;
            }
            
            .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 15px 30px rgba(0, 212, 255, 0.4);
            }
            
            .btn:disabled {
                opacity: 0.6;
                cursor: not-allowed;
                transform: none;
            }
            
            .quick-actions {
                display: flex;
                gap: 10px;
                margin-top: 20px;
                flex-wrap: wrap;
            }
            
            .quick-btn {
                background: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.2);
                color: var(--text);
                padding: 8px 16px;
                border-radius: 8px;
                font-size: 0.9rem;
                cursor: pointer;
                transition: all 0.3s ease;
            }
            
            .quick-btn:hover {
                background: rgba(255, 255, 255, 0.2);
                transform: scale(1.05);
            }
            
            .result-card {
                grid-column: 1 / -1;
                display: none;
                animation: slideUp 0.5s ease;
            }
            
            @keyframes slideUp {
                from { opacity: 0; transform: translateY(30px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            .species-display {
                text-align: center;
                margin-bottom: 30px;
            }
            
            .species-icon {
                font-size: 4rem;
                margin-bottom: 15px;
                display: block;
            }
            
            .species-name {
                font-size: 2.5rem;
                font-weight: 700;
                margin-bottom: 10px;
            }
            
            .confidence-display {
                margin: 30px 0;
            }
            
            .confidence-label {
                display: flex;
                justify-content: space-between;
                margin-bottom: 10px;
                font-weight: 600;
            }
            
            .confidence-bar {
                height: 12px;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 10px;
                overflow: hidden;
                position: relative;
            }
            
            .confidence-fill {
                height: 100%;
                background: linear-gradient(90deg, var(--success), var(--accent));
                border-radius: 10px;
                transition: width 1s ease;
                position: relative;
            }
            
            .confidence-fill::after {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
                animation: shimmer 2s infinite;
            }
            
            @keyframes shimmer {
                0% { transform: translateX(-100%); }
                100% { transform: translateX(100%); }
            }
            
            .probabilities-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-top: 25px;
            }
            
            .prob-item {
                background: rgba(255, 255, 255, 0.05);
                padding: 20px;
                border-radius: 12px;
                text-align: center;
                border: 1px solid rgba(255, 255, 255, 0.1);
                transition: all 0.3s ease;
            }
            
            .prob-item:hover {
                background: rgba(255, 255, 255, 0.1);
                transform: scale(1.02);
            }
            
            .prob-icon {
                font-size: 2rem;
                margin-bottom: 10px;
                display: block;
            }
            
            .prob-name {
                font-weight: 600;
                margin-bottom: 5px;
                text-transform: capitalize;
            }
            
            .prob-value {
                font-size: 1.2rem;
                font-weight: 700;
                color: var(--accent);
            }
            
            .loading {
                display: none;
                text-align: center;
                padding: 40px;
            }
            
            .spinner {
                width: 50px;
                height: 50px;
                border: 3px solid rgba(255, 255, 255, 0.1);
                border-top: 3px solid var(--primary);
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin: 0 auto 20px;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .error {
                background: rgba(239, 68, 68, 0.1);
                border: 1px solid rgba(239, 68, 68, 0.3);
                color: #fecaca;
                padding: 20px;
                border-radius: 12px;
                text-align: center;
            }
            
            @media (max-width: 768px) {
                .dashboard {
                    grid-template-columns: 1fr;
                }
                
                .input-grid {
                    grid-template-columns: 1fr;
                }
                
                .header h1 {
                    font-size: 2.5rem;
                }
                
                .quick-actions {
                    justify-content: center;
                }
            }
        </style>
    </head>
    <body>
        <div class="background-pattern"></div>
        
        <div class="container">
            <div class="header">
                <h1><i class="fas fa-brain"></i> Iris AI</h1>
                <p>Neural Network-Powered Species Classification Dashboard</p>
            </div>
            
            <div class="dashboard">
                <div class="card">
                    <div class="card-header">
                        <i class="fas fa-microscope"></i>
                        <h3>Measurement Input</h3>
                    </div>
                    
                    <form id="irisForm">
                        <div class="input-grid">
                            <div class="input-container">
                                <label for="sepal_length"><i class="fas fa-leaf"></i> Sepal Length (cm)</label>
                                <input type="number" id="sepal_length" step="0.1" min="0" max="10" required>
                            </div>
                            <div class="input-container">
                                <label for="sepal_width"><i class="fas fa-leaf"></i> Sepal Width (cm)</label>
                                <input type="number" id="sepal_width" step="0.1" min="0" max="10" required>
                            </div>
                            <div class="input-container">
                                <label for="petal_length"><i class="fas fa-spa"></i> Petal Length (cm)</label>
                                <input type="number" id="petal_length" step="0.1" min="0" max="10" required>
                            </div>
                            <div class="input-container">
                                <label for="petal_width"><i class="fas fa-spa"></i> Petal Width (cm)</label>
                                <input type="number" id="petal_width" step="0.1" min="0" max="10" required>
                            </div>
                        </div>
                        
                        <button type="submit" class="btn" id="predictBtn">
                            <i class="fas fa-rocket"></i> Analyze Species
                        </button>
                    </form>
                </div>
                
                <div class="card">
                    <div class="card-header">
                        <i class="fas fa-flask"></i>
                        <h3>Quick Samples</h3>
                    </div>
                    
                    <p style="color: var(--text-muted); margin-bottom: 20px;">
                        Load pre-configured measurements for testing
                    </p>
                    
                    <div class="quick-actions">
                        <button class="quick-btn" onclick="fillExample(5.1, 3.5, 1.4, 0.2)">
                            🌸 Setosa Sample
                        </button>
                        <button class="quick-btn" onclick="fillExample(7.0, 3.2, 4.7, 1.4)">
                            🌼 Versicolor Sample
                        </button>
                        <button class="quick-btn" onclick="fillExample(6.3, 3.3, 6.0, 2.5)">
                            🌺 Virginica Sample
                        </button>
                        <button class="quick-btn" onclick="randomSample()">
                            🎲 Random Sample
                        </button>
                    </div>
                    
                    <div style="margin-top: 30px; padding: 20px; background: rgba(255,255,255,0.05); border-radius: 12px;">
                        <h4 style="margin-bottom: 15px; color: var(--primary);">
                            <i class="fas fa-info-circle"></i> About the Model
                        </h4>
                        <p style="color: var(--text-muted); font-size: 0.9rem; line-height: 1.5;">
                            This AI model uses machine learning to classify Iris flowers into three species based on their sepal and petal measurements. The model has been trained on the classic Iris dataset and provides confidence scores for predictions.
                        </p>
                    </div>
                </div>
                
                <div class="card result-card" id="resultCard">
                    <div class="card-header">
                        <i class="fas fa-chart-bar"></i>
                        <h3>Classification Results</h3>
                    </div>
                    
                    <div class="loading" id="loading">
                        <div class="spinner"></div>
                        <p><i class="fas fa-cog fa-spin"></i> Processing neural network...</p>
                    </div>
                    
                    <div id="results" style="display: none;">
                        <div class="species-display">
                            <span class="species-icon" id="speciesIcon">🌸</span>
                            <div class="species-name" id="speciesName">Iris Setosa</div>
                        </div>
                        
                        <div class="confidence-display">
                            <div class="confidence-label">
                                <span><i class="fas fa-percentage"></i> Confidence Level</span>
                                <span id="confidenceText">95.2%</span>
                            </div>
                            <div class="confidence-bar">
                                <div class="confidence-fill" id="confidenceFill" style="width: 0%"></div>
                            </div>
                        </div>
                        
                        <div class="probabilities-grid" id="probabilitiesGrid">
                            <!-- Probability items will be inserted here -->
                        </div>
                    </div>
                    
                    <div id="errorDisplay" class="error" style="display: none;">
                        <i class="fas fa-exclamation-triangle"></i>
                        <h4>Analysis Failed</h4>
                        <p id="errorMessage">An error occurred during classification.</p>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            const speciesData = {
                'setosa': { icon: '🌸', color: '#ff6b6b' },
                'versicolor': { icon: '🌼', color: '#4ecdc4' },
                'virginica': { icon: '🌺', color: '#45b7d1' }
            };
            
            function fillExample(sl, sw, pl, pw) {
                document.getElementById('sepal_length').value = sl;
                document.getElementById('sepal_width').value = sw;
                document.getElementById('petal_length').value = pl;
                document.getElementById('petal_width').value = pw;
                
                // Add visual feedback
                const inputs = [document.getElementById('sepal_length'), document.getElementById('sepal_width'), document.getElementById('petal_length'), document.getElementById('petal_width')];
                inputs.forEach(input => {
                    input.style.borderColor = 'var(--primary)';
                    setTimeout(() => {
                        input.style.borderColor = 'rgba(255, 255, 255, 0.1)';
                    }, 1000);
                });
            }
            
            function randomSample() {
                const ranges = {
                    sepal_length: [4.3, 7.9],
                    sepal_width: [2.0, 4.4],
                    petal_length: [1.0, 6.9],
                    petal_width: [0.1, 2.5]
                };
                
                Object.keys(ranges).forEach(key => {
                    const [min, max] = ranges[key];
                    const value = (Math.random() * (max - min) + min).toFixed(1);
                    document.getElementById(key).value = value;
                });
            }
            
            document.getElementById('irisForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const formData = {
                    sepal_length: parseFloat(document.getElementById('sepal_length').value),
                    sepal_width: parseFloat(document.getElementById('sepal_width').value),
                    petal_length: parseFloat(document.getElementById('petal_length').value),
                    petal_width: parseFloat(document.getElementById('petal_width').value)
                };
                
                // Show loading
                document.getElementById('resultCard').style.display = 'block';
                document.getElementById('loading').style.display = 'block';
                document.getElementById('results').style.display = 'none';
                document.getElementById('errorDisplay').style.display = 'none';
                document.getElementById('predictBtn').disabled = true;
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(formData)
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        displayResult(data);
                    } else {
                        displayError(data.detail || 'An error occurred');
                    }
                } catch (error) {
                    displayError('Network error: ' + error.message);
                } finally {
                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('predictBtn').disabled = false;
                }
            });
            
            function displayResult(data) {
                const species = data.species;
                const confidence = (data.confidence * 100).toFixed(1);
                const speciesInfo = speciesData[species];
                
                // Update species display
                document.getElementById('speciesIcon').textContent = speciesInfo.icon;
                document.getElementById('speciesName').textContent = `Iris ${species.charAt(0).toUpperCase() + species.slice(1)}`;
                document.getElementById('speciesName').style.color = speciesInfo.color;
                
                // Update confidence
                document.getElementById('confidenceText').textContent = confidence + '%';
                setTimeout(() => {
                    document.getElementById('confidenceFill').style.width = confidence + '%';
                }, 100);
                
                // Update probabilities
                const probabilitiesGrid = document.getElementById('probabilitiesGrid');
                probabilitiesGrid.innerHTML = '';
                
                Object.entries(data.probabilities).forEach(([spec, prob]) => {
                    const probPercent = (prob * 100).toFixed(1);
                    const specInfo = speciesData[spec];
                    
                    const probItem = document.createElement('div');
                    probItem.className = 'prob-item';
                    probItem.innerHTML = `
                        <span class="prob-icon">${specInfo.icon}</span>
                        <div class="prob-name">${spec}</div>
                        <div class="prob-value">${probPercent}%</div>
                    `;
                    
                    if (spec === species) {
                        probItem.style.borderColor = specInfo.color;
                        probItem.style.background = `${specInfo.color}20`;
                    }
                    
                    probabilitiesGrid.appendChild(probItem);
                });
                
                document.getElementById('results').style.display = 'block';
            }
            
            function displayError(message) {
                document.getElementById('errorMessage').textContent = message;
                document.getElementById('errorDisplay').style.display = 'block';
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health", response_model=HealthCheck, summary="Health check")
async def health_check():
    """Check if the API and model are working properly"""
    return HealthCheck(
        status="healthy",
        is_model_loaded=model is not None
    )

@app.post("/predict", response_model=PredictionOutput, summary="Predict Iris species")
async def predict_iris(input_data: IrisInput):
    """
    Predict the Iris flower species based on sepal and petal measurements.
    
    - **sepal_length**: Length of the sepal in cm (0-10)
    - **sepal_width**: Width of the sepal in cm (0-10)
    - **petal_length**: Length of the petal in cm (0-10)
    - **petal_width**: Width of the petal in cm (0-10)
    
    Returns the predicted species with confidence score and all class probabilities.
    """
    try:
        # Prepare the input features
        features = np.array([[
            input_data.sepal_length,
            input_data.sepal_width,
            input_data.petal_length,
            input_data.petal_width
        ]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        # Prepare response
        predicted_species = class_names[prediction]
        confidence = float(probabilities[prediction])
        
        # Create probabilities dictionary
        prob_dict = {
            class_names[i]: float(probabilities[i]) 
            for i in range(len(class_names))
        }
        
        return PredictionOutput(
            species=predicted_species,
            confidence=confidence,
            probabilities=prob_dict
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction error: {str(e)}"
        )

@app.post("/predict/batch", summary="Batch prediction")
async def predict_batch(input_list: List[IrisInput]):
    """
    Predict multiple Iris flowers at once.
    
    Takes a list of flower measurements and returns predictions for each.
    """
    try:
        predictions = []
        
        for input_data in input_list:
            # Prepare features
            features = np.array([[
                input_data.sepal_length,
                input_data.sepal_width,
                input_data.petal_length,
                input_data.petal_width
            ]])
            
            # Make prediction
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            
            # Prepare response
            predicted_species = class_names[prediction]
            confidence = float(probabilities[prediction])
            
            prob_dict = {
                class_names[i]: float(probabilities[i]) 
                for i in range(len(class_names))
            }
            
            predictions.append(PredictionOutput(
                species=predicted_species,
                confidence=confidence,
                probabilities=prob_dict
            ))
        
        return {"predictions": predictions}
    
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Batch prediction error: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
