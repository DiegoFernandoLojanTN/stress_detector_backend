from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import re
import json
from datetime import datetime
import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Agregar el directorio de modelos al path
sys.path.append(os.path.join(os.path.dirname(__file__), "models"))

from models.model_simulator import stress_detector_simulator

app = Flask(__name__)
CORS(app, 
     origins=["https://diegofernandolojantn.github.io"],
     methods=["GET", "POST", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization"],
     supports_credentials=True)

# Configuración global
app.config["SECRET_KEY"] = "stress_detector_2024"

class StressDetector:
    def __init__(self):
        self.simulator = stress_detector_simulator
        print("Simulador de modelo SVM cargado exitosamente")
        
    def predict_stress(self, text):
        """Predice si el texto contiene estrés usando el simulador"""
        try:
            result = self.simulator.predict_stress(text)
            return {
                "prediction": result["prediction"],
                "probability": result["probability"],
                "confidence": result["confidence"],
                "cleaned_text": result["cleaned_text"]
            }
        except Exception as e:
            print(f"Error en predicción: {e}")
            # Fallback simple
            return {
                "prediction": 0,
                "probability": 0.1,
                "confidence": 0.1,
                "cleaned_text": text.lower()
            }

# Instancia global del detector
detector = StressDetector()

# Datos de métricas del modelo (basados en los archivos analizados)
MODEL_METRICS = {
    "accuracy": 0.9263,
    "precision": 0.9280,
    "recall": 0.9241,
    "f1_score": 0.9260,
    "training_samples": 3800,
    "validation_samples": 475,
    "test_samples": 475,
    "optimization_time": 592.07,
    "training_time": 3.04,
    "support_vectors": 1717,
    "best_params": {
        "C": 21.3104,
        "gamma": "scale",
        "kernel": "rbf",
        "class_weight": "balanced"
    }
}

# Ejemplos de tweets para pruebas rápidas
SAMPLE_TWEETS = [
    {
        "text": "Que no era hasta el 20 los apagones Bobita? Sábado 21/12/24 Ambato totalmente sin luz, ya ni por sectores, es a nivel nacional e incluso en hospital y seguro social, que infamia burlarse así de un pueblo",
        "expected": "Estrés",
        "date": "21 dic. 2024"
    },
    {
        "text": "12:07 y no hay luz 💡 se supone que a las 12 nos daría, que pasó es el apagón que tanto hablan o se durmieron en el sector del Olímpico , ya es un problema de estrés sin opciones😡😡😡😡😡😡😡",
        "expected": "Estrés",
        "date": "16 nov. 2024"
    },
    {
        "text": "14 horas sin luz y no puedo estar más molesto, sobre todo por los que más sufren esta crisis.",
        "expected": "Estrés",
        "date": "26 oct. 2024"
    },
    {
        "text": "Soy el único al que le da ansiedad la posibilidad de otro apagón nacional que dure días? 😢",
        "expected": "Estrés",
        "date": "30 ago. 2024"
    },
    {
        "text": "DIOS MIO PROVOCA UNA CERVEZA PARA LIBERAR ESTE ESTRÉS POST APAGÓN",
        "expected": "Estrés",
        "date": "30 ago. 2024"
    },
    {
        "text": "Hoy es un día hermoso, el sol brilla y todo está perfecto.",
        "expected": "No Estrés",
        "date": "Ejemplo"
    },
    {
        "text": "Gracias por todo el apoyo, me siento muy agradecido.",
        "expected": "No Estrés",
        "date": "Ejemplo"
    }
]

@app.route("/api/predict", methods=["POST"])
def predict():
    """API para predecir estrés en texto"""
    try:
        data = request.get_json()
        text = data.get("text", "").strip()
        
        if not text:
            return jsonify({"error": "Texto vacío"}), 400
        
        # Realizar predicción
        result = detector.predict_stress(text)
        
        # Formatear respuesta
        response = {
            "success": True,
            "prediction": "Estrés" if result["prediction"] == 1 else "No Estrés",
            "probability": round(result["probability"], 4),
            "confidence": round(result["confidence"], 4),
            "cleaned_text": result["cleaned_text"],
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/metrics")
def get_metrics():
    """API para obtener métricas del modelo"""
    return jsonify(MODEL_METRICS)

@app.route("/api/samples")
def get_samples():
    """API para obtener tweets de ejemplo"""
    return jsonify(SAMPLE_TWEETS)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


