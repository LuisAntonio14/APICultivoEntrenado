from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

# --- CONFIGURACIÓN INICIAL DE LA API ---
app = Flask(__name__)
CORS(app) # Habilitar CORS para permitir peticiones externas (desde Ionic)

# --- CARGA DE MODELOS Y ARTEFACTOS SERIALIZADOS ---
print("Cargando modelo y herramientas de preprocesamiento...")
modelo = joblib.load('modelo_cultivos_rf.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')
print("Modelos cargados correctamente.")

# --- BASE DE CONOCIMIENTO (PROPIEDADES DEL SUELO) ---
# Diccionario para mapear el input categórico del usuario a valores numéricos aproximados
suelo_mapping = {
    "Arcillosa": {"N": 60.0, "P": 45.0, "K": 35.0, "ph": 7.5},
    "Arenosa":   {"N": 20.0, "P": 15.0, "K": 25.0, "ph": 6.0},
    "Limosa":    {"N": 80.0, "P": 50.0, "K": 50.0, "ph": 6.8},
    "Negra":     {"N": 100.0, "P": 60.0, "K": 50.0, "ph": 6.5},
    "Roja":      {"N": 40.0, "P": 55.0, "K": 40.0, "ph": 5.5}
}

# --- DEFINICIÓN DE ENDPOINTS ---
@app.route('/predecir', methods=['POST'])
def predecir():
    try:
        # 1. Obtener los datos de la petición (JSON)
        datos = request.get_json()
        
        # Extracción de variables enviadas por el cliente
        tierra = datos.get('tierra')
        temp = float(datos.get('temp'))
        hum = float(datos.get('hum'))
        lluvia = float(datos.get('lluvia'))

        # 2. Asignar valores químicos (N, P, K, pH) según el tipo de tierra
        if tierra in suelo_mapping:
            nutrientes = suelo_mapping[tierra]
            N = nutrientes['N']
            P = nutrientes['P']
            K = nutrientes['K']
            ph = nutrientes['ph']
        else:
            return jsonify({'error': 'Tipo de tierra no reconocido'}), 400

        # 3. Construir el vector de características (Feature Vector)
        # Nota: El orden debe coincidir estrictamente con el usado en el entrenamiento
        # Orden: [N, P, K, temperature, humidity, ph, rainfall]
        features = np.array([[N, P, K, temp, hum, ph, lluvia]])

        # 4. Estandarización de datos (Scaling)
        # Aplicar la misma transformación usada en el entrenamiento
        features_scaled = scaler.transform(features)

        # 5. Cálculo de probabilidades (Inferencia)
        # predict_proba retorna un array con la probabilidad para cada una de las 22 clases
        probabilidades = modelo.predict_proba(features_scaled)[0]
        
        # 6. Mapeo de probabilidades a etiquetas de clase
        nombres_cultivos = label_encoder.classes_

        # Crear una lista de diccionarios con el par {cultivo, viabilidad}
        resultados = []
        for nombre, prob in zip(nombres_cultivos, probabilidades):
            resultados.append({
                "cultivo": nombre,
                "viabilidad": round(prob * 100, 2) # Convertir a porcentaje con 2 decimales
            })

        # 7. Ordenamiento de resultados
        # Ordenar de mayor a menor probabilidad
        resultados_ordenados = sorted(resultados, key=lambda x: x['viabilidad'], reverse=True)

        # 8. Selección del Top 3
        top_3 = resultados_ordenados[:3]

        # 9. Retornar respuesta al cliente
        return jsonify({
            'top_3_recomendaciones': top_3,
            'mensaje': 'Análisis predictivo completado exitosamente.'
        })

    except Exception as e:
        # Manejo de errores internos del servidor
        return jsonify({'error': str(e)}), 500

# --- EJECUCIÓN DEL SERVIDOR ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)