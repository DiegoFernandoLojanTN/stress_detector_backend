"""
Simulador del modelo SVM para detección de estrés
Este módulo simula el comportamiento del modelo real basándose en los resultados del análisis
"""

import re
import numpy as np
from typing import Dict, List, Tuple
import json
import os

class SVMStressDetectorSimulator:
    """
    Simulador del modelo SVM optimizado para detección de estrés en tweets
    Basado en los resultados reales del entrenamiento con Optuna
    """
    
    def __init__(self):
        self.model_metrics = {
            'accuracy': 0.9263,
            'precision': 0.9280,
            'recall': 0.9241,
            'f1_score': 0.9260,
            'support_vectors': 1717,
            'training_samples': 3800
        }
        
        # Términos de estrés identificados en el análisis real
        self.stress_terms = {
            # Términos principales (mayor peso)
            'high_weight': [
                'estres', 'estrés', 'estresado', 'estresada', 'estresante',
                'ansiedad', 'ansioso', 'ansiosa', 'preocupado', 'preocupada',
                'tension', 'tensión', 'tenso', 'tensa', 'insoportable',
                'harto', 'harta', 'agotado', 'agotada', 'cansado', 'cansada',
                'frustrado', 'frustrada', 'desesperado', 'desesperada',
                'colapsado', 'colapsada', 'colapso', 'hartazgo'
            ],
            # Términos contextuales (peso medio)
            'medium_weight': [
                'trabajo', 'trabajar', 'trabajando', 'laboral',
                'no aguanto', 'hasta cuando', 'que infamia', 'burlarse',
                'molesto', 'molesta', 'irritado', 'irritada',
                'problema', 'problemas', 'crisis', 'sufrir', 'sufriendo'
            ],
            # Términos de apoyo (peso bajo)
            'low_weight': [
                'sin luz', 'apagon', 'apagón', 'electricidad',
                'hospital', 'seguro social', 'pueblo', 'nacional',
                'sectores', 'horas', 'días', 'tiempo'
            ]
        }
        
        # Patrones negativos (reducen probabilidad de estrés)
        self.negative_patterns = [
            'gracias', 'agradecido', 'agradecida', 'feliz', 'contento', 'contenta',
            'bien', 'bueno', 'buena', 'excelente', 'perfecto', 'perfecta',
            'hermoso', 'hermosa', 'maravilloso', 'maravillosa'
        ]
        
        # Pesos para el cálculo
        self.weights = {
            'high_weight': 0.3,
            'medium_weight': 0.2,
            'low_weight': 0.1,
            'negative': -0.25,
            'length_factor': 0.05,
            'exclamation_factor': 0.1,
            'caps_factor': 0.1
        }
    
    def clean_text(self, text: str) -> str:
        """Limpia el texto siguiendo el mismo proceso del modelo real"""
        if not isinstance(text, str):
            return ""
        
        # Convertir a minúsculas para análisis
        text_lower = text.lower()
        
        # Eliminar URLs
        text_lower = re.sub(r'http\S+|www\S+|https\S+', '', text_lower, flags=re.MULTILINE)
        # Eliminar menciones
        text_lower = re.sub(r'@\w+', '', text_lower)
        # Eliminar hashtags
        text_lower = re.sub(r'#\S+', '', text_lower)
        # Eliminar emojis básicos (mantener algunos caracteres para análisis)
        text_lower = re.sub(r'[😀-🙏💀-🗿]', '', text_lower)
        # Normalizar espacios
        text_lower = re.sub(r'\s+', ' ', text_lower).strip()
        
        return text_lower
    
    def extract_features(self, text: str) -> Dict[str, float]:
        """Extrae características del texto para simular el proceso de FastText + SVM"""
        cleaned_text = self.clean_text(text)
        original_text = text
        
        features = {
            'stress_score': 0.0,
            'text_length': len(cleaned_text),
            'exclamation_count': original_text.count('!'),
            'caps_ratio': sum(1 for c in original_text if c.isupper()) / max(len(original_text), 1),
            'negative_score': 0.0
        }
        
        # Calcular puntuación de estrés basada en términos
        for term in self.stress_terms['high_weight']:
            if term in cleaned_text:
                features['stress_score'] += self.weights['high_weight']
        
        for term in self.stress_terms['medium_weight']:
            if term in cleaned_text:
                features['stress_score'] += self.weights['medium_weight']
        
        for term in self.stress_terms['low_weight']:
            if term in cleaned_text:
                features['stress_score'] += self.weights['low_weight']
        
        # Calcular puntuación negativa (reduce estrés)
        for pattern in self.negative_patterns:
            if pattern in cleaned_text:
                features['negative_score'] += self.weights['negative']
        
        return features
    
    def predict_stress(self, text: str) -> Dict[str, float]:
        """
        Predice la probabilidad de estrés en el texto
        Simula el comportamiento del modelo SVM real
        """
        if not text or not text.strip():
            return {
                'prediction': 0,
                'probability': 0.1,
                'confidence': 0.1,
                'features': {}
            }
        
        # Extraer características
        features = self.extract_features(text)
        
        # Calcular probabilidad base
        base_probability = 0.1  # Probabilidad base baja
        
        # Agregar puntuación de estrés
        base_probability += features['stress_score']
        
        # Agregar factores adicionales
        if features['exclamation_count'] > 0:
            base_probability += min(features['exclamation_count'] * self.weights['exclamation_factor'], 0.2)
        
        if features['caps_ratio'] > 0.3:  # Más del 30% en mayúsculas
            base_probability += self.weights['caps_factor']
        
        if features['text_length'] > 100:  # Textos largos pueden tener más contexto
            base_probability += self.weights['length_factor']
        
        # Aplicar factores negativos
        base_probability += features['negative_score']
        
        # Normalizar probabilidad entre 0.05 y 0.95
        probability = max(0.05, min(0.95, base_probability))
        
        # Calcular confianza basada en la distancia del umbral 0.5
        confidence = abs(probability - 0.5) * 2
        
        # Agregar algo de ruido realista basado en las métricas del modelo real
        # El modelo real tiene 92.6% de precisión, así que agregamos incertidumbre
        noise_factor = np.random.normal(0, 0.05)  # 5% de ruido
        probability = max(0.05, min(0.95, probability + noise_factor))
        
        # Determinar predicción
        prediction = 1 if probability > 0.5 else 0
        
        return {
            'prediction': prediction,
            'probability': probability,
            'confidence': confidence,
            'features': features,
            'cleaned_text': self.clean_text(text)
        }
    
    def get_model_info(self) -> Dict:
        """Retorna información del modelo simulado"""
        return {
            'model_type': 'SVM (Simulado)',
            'kernel': 'RBF',
            'optimization': 'Optuna TPE',
            'metrics': self.model_metrics,
            'features': 'FastText 128D + Preprocessing',
            'training_data': 'Tweets en español sobre apagones eléctricos',
            'class_balance': 'SMOTE balancing'
        }

# Instancia global del simulador
stress_detector_simulator = SVMStressDetectorSimulator()

def load_model():
    """Función para cargar el modelo (simulado)"""
    return stress_detector_simulator

def predict_stress(text: str) -> Dict:
    """Función de conveniencia para predicción"""
    return stress_detector_simulator.predict_stress(text)

